import streamlit as st
import requests
import re
import pandas as pd
import os

# === CONFIG ===
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
HF_TOKEN = st.secrets["HUGGINGFACE_TOKEN"]  # Requires .streamlit/secrets.toml

# === SESSION STATE ===
if "graded_data" not in st.session_state:
    st.session_state.graded_data = []
if "internal_prompt" not in st.session_state:
    st.session_state.internal_prompt = ""
if "display_prompt" not in st.session_state:
    st.session_state.display_prompt = ""
if "char_threshold" not in st.session_state:
    st.session_state.char_threshold = 25
if "pass_grade" not in st.session_state:
    st.session_state.pass_grade = 100
if "fail_grade" not in st.session_state:
    st.session_state.fail_grade = 0

# === LLM CALL ===
def call_llm(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json={"inputs": prompt},
            timeout=30,
        )
        output = response.json()
        if isinstance(output, list):
            return output[0].get("generated_text", "").strip()
        return output.get("generated_text", "").strip()
    except Exception as e:
        return f"LLM error: {e}"

# === PROMPT BUILDING ===
def build_internal_prompt(criteria: str) -> str:
    return (
        "You are a grading assistant for student discussion posts. "
        "Follow the instructor criteria **strictly**. Do NOT guess the grade. "
        "You only return the reason.\n\n"
        f"Instructor Grading Criteria:\n{criteria.strip()}\n\n"
        "Post:\n{{POST}}"
    )

# === SPAM DETECTOR ===
def is_spam(post: str) -> bool:
    spam_patterns = [
        r"^(.)\1{10,}$",             # repeated characters
        r"^[^a-zA-Z0-9\s]{5,}$",     # only symbols
        r"^(a{10,}|[.?!]{5,})$",     # common spam fillers
    ]
    for pattern in spam_patterns:
        if re.fullmatch(pattern, post.strip()):
            return True
    return False

# === GRADING ===
def grade_post(post: str):
    cleaned = post.strip()
    char_len = len(cleaned)

    if char_len == 0:
        return st.session_state.fail_grade, "(0 characters) Empty post."

    if is_spam(cleaned):
        return st.session_state.fail_grade, f"({char_len} characters) Detected as spam or gibberish."

    meets_char_limit = char_len >= int(st.session_state.char_threshold)
    grade = st.session_state.pass_grade if meets_char_limit else st.session_state.fail_grade

    prompt = st.session_state.internal_prompt.replace("{{POST}}", cleaned)
    response = call_llm(prompt)

    # Extract only the reason portion from response
    reason_match = re.search(r"(Reason:\s*)?(.*)", response, re.IGNORECASE)
    reason = reason_match.group(2).strip() if reason_match else "No explanation."
    return str(grade), f"({char_len} characters) {reason}"

# === PAGE SETUP ===
st.set_page_config(page_title="Discussion Post Grader", page_icon="📝")
st.title("📝 Instructor-Guided Discussion Post Grader")
st.markdown("Define your grading rubric, optimize it, and grade discussion posts manually or via CSV upload.")

# === STEP 1: GRADING CRITERIA ===
st.header("📌 Step 1 – Enter Your Grading Criteria")
example = (
    "Example: Posts must be at least 25 characters, avoid spam, and be clear and relevant. "
    "Grade 100 for acceptable responses, and 0 for unacceptable ones."
)
raw_criteria = st.text_area("Instructor grading criteria:", placeholder=example, height=120)

if st.button("✨ Finalize and Optimize Criteria"):
    if not raw_criteria.strip():
        st.error("Please enter your grading criteria.")
    else:
        with st.spinner("Optimizing your criteria..."):
            needs_grade = not re.search(r"grade.*\d+", raw_criteria, re.I)
            needs_chars = not re.search(r"\d+\s*characters", raw_criteria, re.I)
            final_criteria = raw_criteria.strip()

            if needs_grade:
                col_pass, col_fail = st.columns(2)
                with col_pass:
                    st.session_state.pass_grade = st.text_input("Grade for acceptable posts", "100")
                with col_fail:
                    st.session_state.fail_grade = st.text_input("Grade for unacceptable posts", "0")
                final_criteria += f" Grade {st.session_state.pass_grade} if criteria met, {st.session_state.fail_grade} otherwise."

            if needs_chars:
                st.session_state.char_threshold = st.number_input("Minimum characters", value=25, step=1)
                final_criteria += f" Posts must be at least {st.session_state.char_threshold} characters."

            optimize_prompt = (
                "Rewrite the following grading criteria clearly, concisely, and strictly. "
                "Do not change the logic.\n\nCRITERIA:\n"
                f"{final_criteria}"
            )
            optimized = call_llm(optimize_prompt)

            st.session_state.display_prompt = optimized
            st.session_state.internal_prompt = build_internal_prompt(optimized)

            st.success("Grading criteria optimized.")
            st.code(optimized, language="text")

# === STEP 2: CSV UPLOAD ===
st.header("📁 Step 2 – Upload Discussion Posts (CSV)")
st.markdown("**Your CSV must include a `DiscussionPost` column.** One row = one post.")

example_df = pd.DataFrame({"DiscussionPost": ["This is a valid example post with enough clarity and content."]})
with st.expander("See CSV Format Example"):
    st.write(example_df)

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    if not st.session_state.internal_prompt:
        st.error("Please finalize grading criteria first.")
    else:
        df = pd.read_csv(uploaded)
        if "DiscussionPost" not in df.columns:
            st.error("The uploaded CSV must have a 'DiscussionPost' column.")
        else:
            st.success("CSV accepted. Grading in progress...")
            live_output = st.empty()
            progress = st.progress(0)
            total = len(df)
            for idx, post in enumerate(df["DiscussionPost"], start=1):
                grade, reason = grade_post(post)
                st.session_state.graded_data.append({"Post": post, "Grade": grade, "Reason": reason})
                live_output.dataframe(pd.DataFrame(st.session_state.graded_data), use_container_width=True)
                progress.progress(idx / total)
            st.success("✅ All posts graded.")

# === TABLE OUTPUT ===
st.header("📋 Graded Results")
if st.session_state.graded_data:
    st.dataframe(pd.DataFrame(st.session_state.graded_data), use_container_width=True)
    if st.button("🗑️ Reset All"):
        st.session_state.graded_data.clear()
        st.session_state.internal_prompt = ""
        st.session_state.display_prompt = ""
        st.experimental_rerun()
else:
    st.info("No posts graded yet.")
