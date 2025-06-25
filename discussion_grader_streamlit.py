import streamlit as st
import requests
import re
import pandas as pd

# ────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────────────────────
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
HF_TOKEN   = "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"  # ← your new token

# ────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ────────────────────────────────────────────────────────────────────────────
if "graded_data"      not in st.session_state: st.session_state.graded_data      = []
if "internal_prompt"  not in st.session_state: st.session_state.internal_prompt  = ""
if "display_prompt"   not in st.session_state: st.session_state.display_prompt   = ""
if "char_threshold"   not in st.session_state: st.session_state.char_threshold   = 25
if "pass_grade"       not in st.session_state: st.session_state.pass_grade       = 100
if "fail_grade"       not in st.session_state: st.session_state.fail_grade       = 0

# ────────────────────────────────────────────────────────────────────────────
# LLM INVOCATION
# ────────────────────────────────────────────────────────────────────────────
def call_llm(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(
            HF_API_URL,
            headers=headers,
            json={"inputs": prompt},
            timeout=30
        )
        data = resp.json()
        if isinstance(data, list) and data:
            return data[0].get("generated_text", "").strip()
        if isinstance(data, dict):
            return data.get("generated_text", "").strip()
        return ""
    except Exception as e:
        return f"LLM error: {e}"

# ────────────────────────────────────────────────────────────────────────────
# PROMPT GENERATOR
# ────────────────────────────────────────────────────────────────────────────
def build_internal_prompt(criteria: str) -> str:
    return (
        "You are an expert teaching assistant and discussion-board grader.\n"
        "Strictly follow the instructor’s prompt below. Do NOT assign grades yourself—"
        "the application will handle numeric grading based on length/spam.\n\n"
        f"Instructor Prompt:\n{criteria.strip()}\n\n"
        "Now evaluate the following student post and respond ONLY with:\n"
        "Reason: <one concise sentence explaining why it meets or fails the prompt>\n\n"
        "Post:\n{{POST}}"
    )

# ────────────────────────────────────────────────────────────────────────────
# SIMPLE SPAM/GIBBERISH DETECTION
# ────────────────────────────────────────────────────────────────────────────
def is_spam(text: str) -> bool:
    text = text.strip()
    patterns = [
        r"^(.)\1{10,}$",         # same char repeated
        r"^[^A-Za-z0-9\s]{5,}$", # only symbols
        r"^(a{10,}|[.?!]{5,})$", # aaa… or ???!!!
    ]
    return any(re.fullmatch(p, text) for p in patterns)

# ────────────────────────────────────────────────────────────────────────────
# GRADING LOGIC
# ────────────────────────────────────────────────────────────────────────────
def grade_post(raw_post: str):
    post = str(raw_post or "").strip()
    length = len(post)

    if length == 0:
        return st.session_state.fail_grade, "(0 characters) Empty post."
    if is_spam(post):
        return st.session_state.fail_grade, f"({length} characters) Detected as spam or gibberish."

    meets_length = length >= int(st.session_state.char_threshold)
    grade = st.session_state.pass_grade if meets_length else st.session_state.fail_grade

    prompt = st.session_state.internal_prompt.replace("{{POST}}", post)
    llm_resp = call_llm(prompt)
    reason = re.sub(r"^Reason:\s*", "", llm_resp, flags=re.I).strip()
    if not reason:
        reason = "No explanation returned."

    return str(grade), f"({length} characters) {reason}"

# ────────────────────────────────────────────────────────────────────────────
# PAGE LAYOUT
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Discussion Post Grader", page_icon="📝")
st.title("📝 Instructor-Guided Discussion Post Grader")
st.markdown(
    "1️⃣ Define your prompt → 2️⃣ Optimize with AI → 3️⃣ Grade via CSV → 4️⃣ Download results"
)

# --- STEP 1: Define & Optimize Prompt ---
st.header("📌 Step 1: Enter & Optimize Grading Prompt")
example = "E.g. Posts ≥25 chars, coherent English, relevant to topic. Accept=100, Reject=0."
raw = st.text_area("Instructor prompt:", placeholder=example, height=100)

if st.button("✨ Optimize Prompt"):
    if not raw.strip():
        st.error("Please enter a prompt first.")
    else:
        with st.spinner("Optimizing your prompt..."):
            needs_grade = not re.search(r"\d+\s*if.*\d+", raw, re.I)
            needs_len   = not re.search(r"\d+\s*characters", raw, re.I)
            final = raw.strip()

            if needs_grade:
                pcol, fcol = st.columns(2)
                with pcol:
                    st.session_state.pass_grade = st.text_input("Accept grade", str(st.session_state.pass_grade))
                with fcol:
                    st.session_state.fail_grade = st.text_input("Reject grade", str(st.session_state.fail_grade))
                final += f" Accept={st.session_state.pass_grade}, Reject={st.session_state.fail_grade}."

            if needs_len:
                st.session_state.char_threshold = st.number_input(
                    "Min char threshold", min_value=1, value=st.session_state.char_threshold
                )
                final += f" MinChars={st.session_state.char_threshold}."

            rewrite_prompt = (
                "Please rewrite the following prompt so that a grading assistant AI "
                "will interpret it correctly and concisely. Keep the same logic.\n\n"
                f"Prompt:\n{final}"
            )
            optimized = call_llm(rewrite_prompt)
            st.session_state.display_prompt  = optimized
            st.session_state.internal_prompt = build_internal_prompt(optimized)

        st.success("Prompt optimized!")
        # Immediately display the optimized prompt under the success message
        st.subheader("🔍 Optimized Prompt")
        st.code(st.session_state.display_prompt, language="text")

# --- STEP 2: Grade via CSV ---
st.header("📁 Step 2: Upload CSV of Discussion Posts")
st.markdown("✅ CSV must include a `DiscussionPost` column.")
with st.expander("See format example"):
    st.write(pd.DataFrame({"DiscussionPost": ["Example post that meets your prompt."]}))

uploaded = st.file_uploader("Choose CSV", type="csv")
if uploaded:
    if not st.session_state.internal_prompt:
        st.error("Please optimize your prompt in Step 1 first.")
    else:
        df = pd.read_csv(uploaded)
        if "DiscussionPost" not in df.columns:
            st.error("CSV is missing the `DiscussionPost` column.")
        else:
            df["DiscussionPost"] = df["DiscussionPost"].fillna("").astype(str)
            st.success("Grading started…")
            placeholder = st.empty()
            prog = st.progress(0)
            total = len(df)
            for idx, post in enumerate(df["DiscussionPost"], start=1):
                grade, reason = grade_post(post)
                st.session_state.graded_data.append({
                    "Post": post,
                    "Grade": grade,
                    "Reason": reason
                })
                placeholder.dataframe(
                    pd.DataFrame(st.session_state.graded_data),
                    use_container_width=True
                )
                prog.progress(idx / total)
            st.success("✅ Grading complete!")

# --- FINAL RESULTS & DOWNLOAD ---
st.header("📋 Graded Results")
if st.session_state.graded_data:
    result_df = pd.DataFrame(st.session_state.graded_data)
    st.dataframe(result_df, use_container_width=True)
    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, "graded_posts.csv", "text/csv")
    if st.button("🗑️ Reset All"):
        st.session_state.clear()
        st.experimental_rerun()
else:
    st.info("No posts graded yet. Complete Step 1 & 2.")
