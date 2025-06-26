import streamlit as st
import requests
import re
import pandas as pd

# ────────────────────────────────────────────────────────────────────────────
# CONFIGURATION (Use a public Instruction-tuned model)
# ────────────────────────────────────────────────────────────────────────────
HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
HF_TOKEN   = "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"  # your token

# ────────────────────────────────────────────────────────────────────────────
# SESSION STATE DEFAULTS
# ────────────────────────────────────────────────────────────────────────────
for key, default in {
    "graded_data": [],
    "internal_prompt": "",
    "display_prompt": "",
    "char_threshold": 25,
    "pass_grade": 100,
    "fail_grade": 0
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ────────────────────────────────────────────────────────────────────────────
# HELPER: Call Hugging Face with wait_for_model
# ────────────────────────────────────────────────────────────────────────────
def call_llm(prompt: str) -> (str, dict):
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 256},
        "options": {"use_cache": False, "wait_for_model": True}
    }
    try:
        r = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)
        data = r.json()
    except Exception as e:
        return f"LLM error: {e}", {}

    # HF error?
    if isinstance(data, dict) and data.get("error"):
        return f"LLM error: {data['error']}", data

    # list response?
    if isinstance(data, list) and data and "generated_text" in data[0]:
        return data[0]["generated_text"].strip(), data

    # dict response?
    if isinstance(data, dict) and data.get("generated_text") is not None:
        return data["generated_text"].strip(), data

    return "", data

# ────────────────────────────────────────────────────────────────────────────
# BUILD THE FULL PROMPT TEMPLATE
# ────────────────────────────────────────────────────────────────────────────
def build_internal_prompt(criteria: str) -> str:
    return (
        "You are an expert teaching assistant and discussion-board grader.\n"
        "Strictly follow the instructor’s prompt below. You only return a one-sentence Reason.\n\n"
        f"Instructor Prompt:\n{criteria.strip()}\n\n"
        "Now evaluate this student post and respond ONLY with:\n"
        "Reason: <concise sentence explaining why it meets or fails>\n\n"
        "Post:\n{{POST}}"
    )

# ────────────────────────────────────────────────────────────────────────────
# SPAM DETECTION
# ────────────────────────────────────────────────────────────────────────────
def is_spam(text: str) -> bool:
    t = text.strip()
    patterns = [
        r"^(.)\1{10,}$",         
        r"^[^A-Za-z0-9\s]{5,}$", 
        r"^(a{10,}|[.?!]{5,})$", 
    ]
    return any(re.fullmatch(p, t) for p in patterns)

# ────────────────────────────────────────────────────────────────────────────
# GRADE ONE POST
# ────────────────────────────────────────────────────────────────────────────
def grade_post(raw_post: str):
    post   = str(raw_post or "").strip()
    length = len(post)
    if length == 0:
        return st.session_state.fail_grade, "(0 chars) Empty post."
    if is_spam(post):
        return st.session_state.fail_grade, f"({length} chars) Spam detected."

    meets_len = length >= int(st.session_state.char_threshold)
    grade     = st.session_state.pass_grade if meets_len else st.session_state.fail_grade

    tpl   = st.session_state.internal_prompt.replace("{{POST}}", post)
    out, raw = call_llm(tpl)
    reason = re.sub(r"^Reason:\s*", "", out, flags=re.I).strip() or "(no explanation)"

    return str(grade), f"({length} chars) {reason}"

# ────────────────────────────────────────────────────────────────────────────
# UI LAYOUT
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Discussion Post Grader", page_icon="📝")
st.title("📝 Instructor-Guided Discussion Post Grader")

# Step 1: Optimize Prompt
st.header("📌 Step 1: Enter & Optimize Grading Prompt")
example = "E.g. Posts ≥10 chars; Accept=100 if ok, Reject=0 if not."
raw     = st.text_area("Instructor prompt:", placeholder=example, height=100)

if st.button("✨ Optimize Prompt"):
    if not raw.strip():
        st.error("Please enter a prompt.")
    else:
        with st.spinner("Optimizing…"):
            needs_grade = not re.search(r"\d+\s*(Accept|Pass)", raw, re.I)
            needs_len   = not re.search(r"\d+\s*chars?", raw, re.I)
            final       = raw.strip()

            if needs_grade:
                c1, c2 = st.columns(2)
                with c1:
                    st.session_state.pass_grade = st.text_input("Accept grade",  str(st.session_state.pass_grade))
                with c2:
                    st.session_state.fail_grade = st.text_input("Reject grade",  str(st.session_state.fail_grade))
                final += f" Accept={st.session_state.pass_grade}, Reject={st.session_state.fail_grade}."

            if needs_len:
                st.session_state.char_threshold = st.number_input(
                    "Min char threshold", min_value=1, value=st.session_state.char_threshold
                )
                final += f" MinChars={st.session_state.char_threshold}."

            rewrite = (
                "Rewrite this grading prompt clearly and concisely so an AI will interpret it exactly.\n\n"
                f"Prompt:\n{final}"
            )
            opt, raw_json = call_llm(rewrite)
            st.session_state.display_prompt  = opt or "(no optimized prompt returned)"
            st.session_state.internal_prompt = build_internal_prompt(st.session_state.display_prompt)

        st.success("Prompt optimized!")

# Always show optimized prompt
if st.session_state.display_prompt:
    st.subheader("🔍 Optimized Prompt")
    st.code(st.session_state.display_prompt)
    if st.session_state.display_prompt.startswith("(no optimized"):
        with st.expander("🔧 Raw API response"):
            st.json(raw_json)

# Step 2: CSV grading
st.header("📁 Step 2: Upload CSV of Posts")
st.markdown("✅ Must include a `DiscussionPost` column.")
with st.expander("See example"):
    st.write(pd.DataFrame({"DiscussionPost": ["Example post that meets your prompt."]}))

uploaded = st.file_uploader("Upload CSV", type="csv")
if uploaded:
    if not st.session_state.internal_prompt:
        st.error("Optimize prompt first.")
    else:
        df = pd.read_csv(uploaded)
        if "DiscussionPost" not in df.columns:
            st.error("Missing `DiscussionPost` column.")
        else:
            df["DiscussionPost"] = df["DiscussionPost"].fillna("").astype(str).str.strip()
            st.success("Grading…")
            outp = st.empty()
            prog = st.progress(0)
            for i, post in enumerate(df["DiscussionPost"], 1):
                g, r = grade_post(post)
                st.session_state.graded_data.append({"Post": post, "Grade": g, "Reason": r})
                outp.dataframe(pd.DataFrame(st.session_state.graded_data), use_container_width=True)
                prog.progress(i / len(df))
            st.success("✅ Done!")

# Final table + download
st.header("📋 Graded Results")
if st.session_state.graded_data:
    final_df = pd.DataFrame(st.session_state.graded_data)
    st.dataframe(final_df, use_container_width=True)
    csv = final_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, "graded_posts.csv", "text/csv")
    if st.button("🗑️ Reset All"):
        st.session_state.clear()
        st.experimental_rerun()
else:
    st.info("No posts graded yet.")
