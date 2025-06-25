import streamlit as st
import requests
import re
import pandas as pd

# ────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────────────────────
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
HF_TOKEN   = "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"  # your new token

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
# HELPER: Call Hugging Face Inference
# ────────────────────────────────────────────────────────────────────────────
def call_llm(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(HF_API_URL, headers=headers, json={"inputs": prompt}, timeout=30)
        out = resp.json()
        if isinstance(out, list) and out:
            return out[0].get("generated_text","").strip()
        if isinstance(out, dict):
            return out.get("generated_text","").strip()
        return ""
    except Exception as e:
        return f"LLM error: {e}"

# ────────────────────────────────────────────────────────────────────────────
# BUILDER: Wrap user-entered prompt into a full scoring prompt
# ────────────────────────────────────────────────────────────────────────────
def build_internal_prompt(criteria: str) -> str:
    return (
        "You are an expert teaching assistant and discussion-board grader.\n"
        "Strictly follow the instructor’s prompt below.  "
        "Do NOT assign the numeric grade yourself—"
        "the app will handle Pass/Fail by character count and spam.\n\n"
        f"Instructor Prompt:\n{criteria.strip()}\n\n"
        "Now evaluate the following student post and respond ONLY with:\n"
        "Reason: <one concise sentence explaining why it meets or fails the prompt>\n\n"
        "Post:\n{{POST}}"
    )

# ────────────────────────────────────────────────────────────────────────────
# SPAM DETECTION (short-circuit gibberish)
# ────────────────────────────────────────────────────────────────────────────
def is_spam(text: str) -> bool:
    t = text.strip()
    patterns = [
        r"^(.)\1{10,}$",          # same character repeated
        r"^[^A-Za-z0-9\s]{5,}$",  # only symbols
        r"^(a{10,}|[.?!]{5,})$",  # aaa… or ???!!!
    ]
    return any(re.fullmatch(p, t) for p in patterns)

# ────────────────────────────────────────────────────────────────────────────
# CORE: Grade one post (spam & length + LLM reason)
# ────────────────────────────────────────────────────────────────────────────
def grade_post(raw_post: str):
    post = str(raw_post or "").strip()
    length = len(post)

    if length == 0:
        return st.session_state.fail_grade, "(0 characters) Empty post."
    if is_spam(post):
        return st.session_state.fail_grade, f"({length} characters) Detected as spam or gibberish."

    meets_len = length >= int(st.session_state.char_threshold)
    grade = st.session_state.pass_grade if meets_len else st.session_state.fail_grade

    prompt = st.session_state.internal_prompt.replace("{{POST}}", post)
    llm_resp = call_llm(prompt)
    reason = re.sub(r"^Reason:\s*", "", llm_resp, flags=re.I).strip()
    if not reason:
        reason = "(no optimized prompt returned)"

    return str(grade), f"({length} characters) {reason}"

# ────────────────────────────────────────────────────────────────────────────
# PAGE LAYOUT
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Discussion Post Grader", page_icon="📝")

st.title("📝 Instructor-Guided Discussion Post Grader")
st.markdown(
    "1️⃣ Enter & Optimize Prompt → 2️⃣ Upload CSV → 3️⃣ Download Results"
)

# ----- STEP 1: Enter & Optimize Prompt -----
st.header("📌 Step 1: Enter & Optimize Grading Prompt")
example = "E.g. Posts ≥10 chars; Accept=100 if ok, Reject=0 if not."
raw = st.text_area("Instructor prompt:", placeholder=example, height=100)

if st.button("✨ Optimize Prompt"):
    if not raw.strip():
        st.error("Please enter a prompt first.")
    else:
        with st.spinner("Optimizing your prompt..."):
            needs_grade = not re.search(r"\d+\s*(Accept|Pass)", raw, re.I)
            needs_len   = not re.search(r"\d+\s*chars?", raw, re.I)
            final = raw.strip()

            if needs_grade:
                c1, c2 = st.columns(2)
                with c1:
                    st.session_state.pass_grade = st.text_input("Accept grade", str(st.session_state.pass_grade))
                with c2:
                    st.session_state.fail_grade = st.text_input("Reject grade", str(st.session_state.fail_grade))
                final += f" Accept={st.session_state.pass_grade}, Reject={st.session_state.fail_grade}."

            if needs_len:
                st.session_state.char_threshold = st.number_input(
                    "Min char threshold", min_value=1, value=st.session_state.char_threshold
                )
                final += f" MinChars={st.session_state.char_threshold}."

            rewrite = (
                "Rewrite this prompt so that a grading assistant AI will interpret it exactly "
                "as given, in clear, concise English. Do not change its logic.\n\n"
                f"Prompt:\n{final}"
            )
            optimized = call_llm(rewrite)

            st.session_state.display_prompt  = optimized
            st.session_state.internal_prompt = build_internal_prompt(optimized)

        st.success("Prompt optimized!")

# ----- ALWAYS SHOW the optimized prompt if available -----
if st.session_state.display_prompt:
    st.subheader("🔍 Optimized Prompt")
    st.code(st.session_state.display_prompt, language="text")

# ----- STEP 2: Upload & Grade CSV -----
st.header("📁 Step 2: Upload CSV of Discussion Posts")
st.markdown("✅ CSV must include a `DiscussionPost` column.")
with st.expander("See format example"):
    st.write(pd.DataFrame({"DiscussionPost": ["Example post that meets your prompt."]}))

uploaded = st.file_uploader("Upload CSV", type="csv")
if uploaded:
    if not st.session_state.internal_prompt:
        st.error("Please optimize your prompt in Step 1 first.")
    else:
        df = pd.read_csv(uploaded)
        if "DiscussionPost" not in df.columns:
            st.error("CSV missing `DiscussionPost` column.")
        else:
            df["DiscussionPost"] = df["DiscussionPost"].fillna("").astype(str)
            st.success("Grading started…")
            placeholder = st.empty()
            prog = st.progress(0)
            total = len(df)
            for i, post in enumerate(df["DiscussionPost"], start=1):
                grade, reason = grade_post(post)
                st.session_state.graded_data.append({
                    "Post": post,
                    "Grade": grade,
                    "Reason": reason
                })
                placeholder.dataframe(pd.DataFrame(st.session_state.graded_data),
                                      use_container_width=True)
                prog.progress(i / total)
            st.success("✅ Grading complete!")

# ----- FINAL RESULTS & DOWNLOAD -----
st.header("📋 Graded Results")
if st.session_state.graded_data:
    results = pd.DataFrame(st.session_state.graded_data)
    st.dataframe(results, use_container_width=True)
    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, "graded_posts.csv", "text/csv")
    if st.button("🗑️ Reset All"):
        st.session_state.clear()
        st.experimental_rerun()
else:
    st.info("No posts graded yet. Complete Step 1 & 2.")
