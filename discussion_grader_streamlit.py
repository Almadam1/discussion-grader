import streamlit as st
import requests
import re
import pandas as pd

# ────────────────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────────────────
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
HF_TOKEN   = "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"   # <-- hard-coded

# ────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ────────────────────────────────────────────────────────────────────────────
if "graded_data"      not in st.session_state: st.session_state.graded_data      = []
if "internal_prompt"  not in st.session_state: st.session_state.internal_prompt  = ""
if "display_prompt"   not in st.session_state: st.session_state.display_prompt   = ""
if "char_threshold"   not in st.session_state: st.session_state.char_threshold   = 25
if "pass_grade"       not in st.session_state: st.session_state.pass_grade       = 100
if "fail_grade"       not in st.session_state: st.session_state.fail_grade       = 0

# ────────────────────────────────────────────────────────────────────────────
# CALL HUGGING FACE LLM
# ────────────────────────────────────────────────────────────────────────────
def call_llm(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type" : "application/json",
    }
    try:
        r = requests.post(HF_API_URL, headers=headers, json={"inputs": prompt}, timeout=30)
        out = r.json()
        if isinstance(out, list):
            return out[0].get("generated_text", "").strip()
        return out.get("generated_text", "").strip()
    except Exception as e:
        return f"LLM error: {e}"

# ────────────────────────────────────────────────────────────────────────────
# PROMPT TEMPLATES
# ────────────────────────────────────────────────────────────────────────────
def build_internal_prompt(criteria: str) -> str:
    return (
        "You are a grading assistant for student discussion posts. "
        "Follow the instructor criteria strictly. You only output the reason.\n\n"
        f"Instructor Grading Criteria:\n{criteria.strip()}\n\n"
        "Post:\n{{POST}}"
    )

# ────────────────────────────────────────────────────────────────────────────
# SIMPLE SPAM / GIBBERISH CHECK
# ────────────────────────────────────────────────────────────────────────────
def is_spam(post: str) -> bool:
    patterns = [
        r"^(.)\1{10,}$",          # same char repeated
        r"^[^a-zA-Z0-9\s]{5,}$",  # only symbols
        r"^(a{10,}|[.?!]{5,})$",  # aaa… or ???!!!
    ]
    return any(re.fullmatch(p, post.strip()) for p in patterns)

# ────────────────────────────────────────────────────────────────────────────
# MAIN GRADING FUNCTION
# ────────────────────────────────────────────────────────────────────────────
def grade_post(post: str):
    post_clean = post.strip()
    n_chars    = len(post_clean)

    if n_chars == 0:
        return st.session_state.fail_grade, "(0 characters) Empty post."
    if is_spam(post_clean):
        return st.session_state.fail_grade, f"({n_chars} characters) Detected as spam/gibberish."

    meets_len = n_chars >= int(st.session_state.char_threshold)
    grade     = st.session_state.pass_grade if meets_len else st.session_state.fail_grade

    prompt    = st.session_state.internal_prompt.replace("{{POST}}", post_clean)
    reason_llm = call_llm(prompt)
    reason    = re.sub(r"^Reason:\s*", "", reason_llm, flags=re.I).strip() or "No explanation."

    return str(grade), f"({n_chars} characters) {reason}"

# ────────────────────────────────────────────────────────────────────────────
# STREAMLIT PAGE
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Discussion Post Grader", page_icon="📝")
st.title("📝 Instructor-Guided Discussion Post Grader")

# --- Step 1: instructor rubric ------------------------------------------------
st.header("📌 Step 1 – Enter Your Grading Criteria")
example = (
    "Example: Posts must be at least 25 characters, avoid spam, and be clear. "
    "Grade 100 if criteria met, 0 otherwise."
)
criteria_raw = st.text_area("Instructor grading criteria:", placeholder=example, height=120)

if st.button("✨ Finalize & Optimize"):
    if not criteria_raw.strip():
        st.error("Please enter criteria first.")
    else:
        needs_grade = not re.search(r"grade.*\d+", criteria_raw, re.I)
        needs_len   = not re.search(r"\d+\s*characters", criteria_raw, re.I)

        final_criteria = criteria_raw.strip()

        if needs_grade:
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.pass_grade = st.text_input("Grade for acceptable posts", "100")
            with col2:
                st.session_state.fail_grade = st.text_input("Grade for unacceptable posts", "0")
            final_criteria += (
                f" Grade {st.session_state.pass_grade} if criteria met, "
                f"{st.session_state.fail_grade} otherwise."
            )

        if needs_len:
            st.session_state.char_threshold = st.number_input("Minimum characters", 25, step=1)
            final_criteria += f" Posts must be at least {st.session_state.char_threshold} characters."

        optimized = call_llm(
            "Rewrite these grading criteria clearly and concisely, keeping the logic intact:\n\n"
            + final_criteria
        )

        st.session_state.display_prompt  = optimized
        st.session_state.internal_prompt = build_internal_prompt(optimized)
        st.success("Criteria optimized and saved.")
        st.code(optimized)

# --- Step 2: CSV upload -------------------------------------------------------
st.header("📁 Step 2 – Upload CSV of Posts")
st.markdown("CSV **must** include a `DiscussionPost` column (one post per row).")
uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    if not st.session_state.internal_prompt:
        st.error("First finalize your grading criteria above.")
    else:
        df = pd.read_csv(uploaded)
        if "DiscussionPost" not in df.columns:
            st.error("CSV is missing the `DiscussionPost` column.")
        else:
            st.success("CSV accepted. Grading…")
            placeholder = st.empty()
            progress    = st.progress(0)
            total       = len(df)
            for i, post in enumerate(df["DiscussionPost"], 1):
                grade, reason = grade_post(post)
                st.session_state.graded_data.append({"Post": post, "Grade": grade, "Reason": reason})
                placeholder.dataframe(pd.DataFrame(st.session_state.graded_data), use_container_width=True)
                progress.progress(i / total)
            st.success("✅ All posts graded.")

# --- Results table ------------------------------------------------------------
st.header("📋 Graded Results")
if st.session_state.graded_data:
    st.dataframe(pd.DataFrame(st.session_state.graded_data), use_container_width=True)
    if st.button("🗑️ Reset"):
        st.session_state.clear()
        st.experimental_rerun()
else:
    st.info("No posts graded yet.")
