"""Ask the Cosmos - a space tutor that answers questions using an LLM
grounded in NASA's public Image & Video Library and the Astronomy Picture of the Day."""

import os
import re

import requests
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Ask the Cosmos", page_icon="🚀")


def secret(name, default=None):
    """Read a setting from environment variables or Streamlit secrets."""
    if name in os.environ:
        return os.environ[name]
    try:
        return st.secrets[name]
    except Exception:
        return default


API_KEY = secret("LLM_API_KEY")
BASE_URL = secret("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL = secret("LLM_MODEL")
NASA_KEY = secret("NASA_API_KEY", "DEMO_KEY")  # DEMO_KEY works but is rate limited

LEVELS = {
    "Curious kid (about 10)": "Use simple words, fun comparisons, and short sentences. Avoid jargon.",
    "Curious teen": "Be clear and engaging. Explain technical terms the first time you use them.",
    "Adult beginner": "Be accurate and conversational. Light technical detail is fine.",
    "Space nerd": "Be precise and technical. Include numbers, units, and key concepts.",
}

EXAMPLES = [
    "How do black holes form?",
    "Why is Mars red?",
    "What did the James Webb Space Telescope discover?",
    "Tell me about the Apollo 25 moon landing",  # false-premise test
]

SYSTEM_PROMPT = (
    "You are Cosmos, a friendly and honest space tutor. "
    "You are given reference material from NASA's public library; prefer it when it is relevant "
    "and cite it as [1], [2], etc. If the reference material does not cover the question, "
    "you may use well-established general knowledge, but say so. "
    "Never invent missions, dates, numbers, or quotes. If you are not sure, say you are not sure. "
    "If the question contains a false premise (for example a mission or object that does not exist), "
    "politely point that out instead of playing along. "
    "If the question is not about space or astronomy, gently steer back to space. "
    "Keep answers under about 200 words."
)

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "do", "does", "did", "how", "what", "why", "when",
    "where", "who", "which", "can", "could", "would", "should", "of", "in", "on", "at", "to", "for",
    "and", "or", "it", "its", "i", "me", "my", "you", "your", "about", "tell", "explain", "please",
    "there", "that", "this", "with", "from", "be", "if", "so", "than", "then", "really", "much",
}


# ---------- NASA data ----------
def keywords(question: str) -> str:
    words = re.findall(r"[a-zA-Z0-9\-]+", question.lower())
    kept = [w for w in words if w not in STOPWORDS]
    return " ".join(kept[:6]) or question


@st.cache_data(ttl=3600, show_spinner=False)
def search_nasa(question: str, n: int = 3):
    """Search the NASA Image and Video Library (no API key needed)."""
    try:
        r = requests.get(
            "https://images-api.nasa.gov/search",
            params={"q": keywords(question), "media_type": "image"},
            timeout=10,
        )
        r.raise_for_status()
        items = r.json()["collection"]["items"]
    except Exception:
        return []

    results = []
    for item in items:
        if len(results) >= n:
            break
        info = item["data"][0]
        desc = re.sub(r"<[^>]+>", "", info.get("description", "") or "").strip()
        if len(desc) < 40:  # skip entries with no useful text
            continue
        results.append(
            {
                "title": info.get("title", "Untitled"),
                "description": desc[:700],
                "url": f"https://images.nasa.gov/details/{info['nasa_id']}",
            }
        )
    return results


@st.cache_data(ttl=3600, show_spinner=False)
def get_apod():
    """Fetch NASA's Astronomy Picture of the Day."""
    try:
        r = requests.get(
            "https://api.nasa.gov/planetary/apod", params={"api_key": NASA_KEY}, timeout=10
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# ---------- Chat logic ----------
def answer(question, history, level):
    if not API_KEY or not MODEL:
        return "The tutor isn't configured yet: the LLM_API_KEY and LLM_MODEL secrets are required."

    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    sources = search_nasa(question)
    context = "\n\n".join(
        f"[{i}] {s['title']}: {s['description']}" for i, s in enumerate(sources, 1)
    ) or "No NASA reference material was found for this question."

    messages = [{"role": "system", "content": SYSTEM_PROMPT + " " + LEVELS[level]}]
    for turn in history[-6:]:  # keep only recent turns
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append(
        {"role": "user", "content": f"NASA reference material:\n{context}\n\nQuestion: {question}"}
    )

    try:
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, max_tokens=600, temperature=0.4
        )
        text = resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Sorry, I couldn't reach the AI model right now ({type(e).__name__}). Please try again."

    if sources:
        links = "\n".join(f"- [{i}] [{s['title']}]({s['url']})" for i, s in enumerate(sources, 1))
        text += f"\n\n**NASA sources:**\n{links}"
    return text


# ---------- Interface ----------
st.title("🚀 Ask the Cosmos")
st.caption("A space tutor powered by an LLM and grounded in NASA's public library.")

apod = get_apod()
if apod:
    with st.expander("Today's NASA picture", expanded=False):
        st.subheader(apod.get("title", ""))
        if apod.get("media_type") == "image":
            st.image(apod["url"])
        else:
            st.write(f"Today's picture is a video: {apod.get('url')}")
        st.write(apod.get("explanation", "")[:400] + "...")

with st.sidebar:
    level = st.selectbox("Explain it like I'm a...", list(LEVELS), index=1)
    st.markdown("**Try one of these:**")
    clicked = None
    for ex in EXAMPLES:
        if st.button(ex):
            clicked = ex

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask anything about space...") or clicked

if prompt:
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching NASA and thinking..."):
            reply = answer(prompt, history, level)
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
