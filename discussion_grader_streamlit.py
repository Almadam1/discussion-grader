import os
import re
import json
import pandas as pd
import streamlit as st
import openai
from openai import OpenAIError
import nltk
from nltk import pos_tag, word_tokenize

# Configure page layout
st.set_page_config(page_title="Discussion Post Grader", layout="wide")

# Constants
MODEL_NAME       = "gpt-3.5-turbo"
SEM_TEMPERATURE  = 0.0
SUM_TEMPERATURE  = 0.7
MAX_POST_LENGTH  = 2000

# Ensure NLTK data
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("taggers/averaged_perceptron_tagger")
except LookupError:
    nltk.download("punkt")
    nltk.download("averaged_perceptron_tagger")

# Load OpenAI key
OPENAI_KEY = st.secrets["openai"]["api_key"]
if not OPENAI_KEY:
    st.error("\u274c No OpenAI key in secrets.")
    st.stop()
openai.api_key = OPENAI_KEY

# Prompt input
st.title("\ud83d\udcdd Discussion Post Grader")

if "prompt_text" not in st.session_state:
    st.session_state.prompt_text = None

if st.session_state.prompt_text is None:
    prompt_input = st.text_area(
        "1) Enter the instructor's grading prompt",
        placeholder="e.g. Must be at least 20 characters long, contain 5 vowels, and 2 verbs"
    )
    if st.button("\u2714\ufe0f Confirm Prompt", key="confirm_prompt_button"):
        st.session_state.prompt_text = prompt_input.strip().replace("\u2026", "...")
else:
    st.markdown(f"**Grading Prompt:** {st.session_state.prompt_text}")

if not st.session_state.prompt_text:
    st.info("Please enter and confirm your grading prompt above.")
    st.stop()

PROMPT = st.session_state.prompt_text

# Count-rule parsing
COUNT_PATTERNS = [
    r"(?:at\\s+least)\\s+(\\d+)\\s+([A-Za-z ]+?)(?:[.,]|$)",
    r"(?:must|should)\\s+contain\\s+(\\d+)\\s+([A-Za-z ]+?)(?:[.,]|$)",
    r"(?:contains)\\s+(\\d+)\\s+([A-Za-z ]+?)(?:[.,]|$)",
]

@st.cache_data
def parse_count_rules(prompt: str):
    rules = []
    for pat in COUNT_PATTERNS:
        for num, feat in re.findall(pat, prompt, re.IGNORECASE):
            key = feat.strip().lower().rstrip("s")
            rules.append((key, int(num)))
    return rules

rules = parse_count_rules(PROMPT)

if rules:
    unknown = [feat for feat, _ in rules if feat not in {"capital letter","vowel","noun","verb"}]
    if unknown:
        st.warning(f"\u26a0\ufe0f Cannot count feature(s): {', '.join(unknown)}; those will be checked semantically.")

# Feature counting
FEATURE_FNS = {
    "capital letter": lambda text: len(re.findall(r"[A-Z]", text)),
    "vowel": lambda text: len(re.findall(r"[aeiouAEIOU]", text)),
    "noun": lambda text: sum(1 for _, t in pos_tag(word_tokenize(text)) if t.startswith("NN")),
    "verb": lambda text: sum(1 for _, t in pos_tag(word_tokenize(text)) if t.startswith("VB"))
}

# Safe LLM call
def safe_chat(messages, **kwargs):
    try:
        return openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=messages,
            **kwargs
        )
    except OpenAIError as e:
        st.error(f"LLM error: {e}")
        return None

# Semantic fallback
def semantic_grade(text: str):
    snippet = text if len(text) <= MAX_POST_LENGTH else text[:MAX_POST_LENGTH] + "\u2026"
    user_content = (
        f"Grading criteria (semantic): {PROMPT}\n\n"
        f"Student post:\n{snippet}\n\n"
        "Respond only with valid JSON:\n"
        "{\"meets\": true|false, \"reason\": \"one-sentence explanation\"}"
    )
    resp = safe_chat([
        {"role": "system", "content": "You are an expert grader."},
        {"role": "user", "content": user_content}
    ], temperature=SEM_TEMPERATURE, max_tokens=150)
    if not resp:
        return 0, "LLM failure"
    try:
        out = resp.choices[0].message.content.strip()
        obj = json.loads(re.search(r"(\{.*\})", out, re.DOTALL).group(1))
        meets = bool(obj.get("meets"))
        return (100 if meets else 0), obj.get("reason", "").strip()
    except:
        return 0, "LLM returned invalid JSON"

# Unified grading
def grade_post(text: str):
    text = text.replace("\u2026", "...")
    if rules:
        failures = []
        for feat, need in rules:
            fn = FEATURE_FNS.get(feat)
            if fn:
                have = fn(text)
                if have < need:
                    failures.append(f"{have} {feat}(s) (needs {need})")
        meets = not failures
        reason = "Meets all criteria." if meets else "Missing: " + "; ".join(failures)
        return (100 if meets else 0), reason
    return semantic_grade(text)

# Upload and grade
uploaded = st.file_uploader("Upload CSV with a `DiscussionPost` column", type="csv", disabled=not PROMPT, key="file_uploader")
if uploaded:
    df = pd.read_csv(uploaded)
    if "DiscussionPost" not in df.columns:
        st.error("CSV must contain a column named `DiscussionPost`.")
    else:
        st.subheader("Grading in progress…")
        placeholder = st.empty()
        progress = st.progress(0)
        results = []
        passed = failed = 0

        for i, post in enumerate(df["DiscussionPost"].astype(str), start=1):
            grade, reason = grade_post(post)
            results.append({"DiscussionPost": post, "Grade": grade, "Reason": reason})
            passed += grade == 100
            failed += grade != 100
            placeholder.dataframe(pd.DataFrame(results), use_container_width=True)
            progress.progress(i / len(df))

        st.download_button(
            "⬇️ Download Graded CSV",
            data=pd.DataFrame(results).to_csv(index=False).encode("utf-8"),
            file_name="graded_results.csv",
            mime="text/csv",
            key="download_button"
        )
        st.markdown(f"**Summary:** {passed} passed • {failed} failed out of {len(results)}")

        # Summary
        st.subheader("💡 Summary of Discussion Posts")
        all_posts = "\n\n".join(df["DiscussionPost"].astype(str).tolist())

        @st.cache_data(show_spinner=False)
        def summarize(posts: str) -> str:
            resp = safe_chat([
                {"role": "system", "content": "You are an expert summarizer."},
                {"role": "user", "content": "Please provide a concise paragraph summary of these discussion posts:\n\n" + posts}
            ], temperature=SUM_TEMPERATURE, max_tokens=200)
            return resp.choices[0].message.content.strip() if resp else "Could not generate summary."

        st.write(summarize(all_posts))

        if st.button("\ud83d\udd04 Reset App", key="reset_button"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.cache_data.clear()
            st.experimental_rerun()
