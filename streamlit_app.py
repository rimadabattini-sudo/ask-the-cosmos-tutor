"""Ask the Cosmos - a space tutor that answers questions using an LLM
grounded in NASA's public Image & Video Library and the Astronomy Picture of the Day."""

import os
import re
import time
from urllib.parse import quote

import requests
import streamlit as st
from openai import OpenAI, RateLimitError

st.set_page_config(page_title="Ask the Cosmos", page_icon="🌌")


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

# ---------- Reading levels ----------
# Each level turns four "dials" a little further: vocabulary, sentence style,
# numbers/maths, and length. The rules are repeated in every request so the
# model cannot drift back to one middle-of-the-road style.
LEVELS = {
    "Curious kid (about 10)": {
        "icon": "🌱",
        "blurb": "Short sentences, no jargon, one playful comparison.",
        "words": "60 to 90 words",
        "cite": False,
        "temperature": 0.7,
        "rules": (
            "READER: a curious 10-year-old (about 4th grade). "
            "Every sentence has at most 12 words. Use only everyday words a 10-year-old knows. "
            "Avoid technical terms; if one is truly needed, use at most ONE and explain it right away "
            "with a comparison. Build the whole answer around ONE playful everyday comparison "
            "(food, toys, pets, playground, sports). Use no equations and no big numbers; compare "
            "sizes and times to familiar things instead. Warm, excited tone. At most two emoji. "
            "End with one short, fun question that sparks wonder."
        ),
    },
    "Curious teen": {
        "icon": "🔭",
        "blurb": "A few key terms explained, one analogy, simple numbers.",
        "words": "110 to 150 words",
        "cite": True,
        "temperature": 0.6,
        "rules": (
            "READER: a curious teenager (about 9th grade) who has taken basic science. "
            "Use short paragraphs and clear sentences. Introduce two or three key science terms and "
            "define each one in plain words the first time it appears. Use one analogy. Include one or "
            "two real numbers with units, each compared to something familiar. Friendly, slightly "
            "informal tone. No equations."
        ),
    },
    "Adult beginner": {
        "icon": "🪐",
        "blurb": "Clear cause and effect, correct terms, real figures.",
        "words": "160 to 210 words",
        "cite": True,
        "temperature": 0.5,
        "rules": (
            "READER: an educated adult with no science background. "
            "Explain the cause-and-effect chain step by step in plain language. Use correct "
            "terminology and define each term briefly. Include a few key figures with units. "
            "Use at most one brief comparison. When relevant, mention one thing scientists still "
            "do not know. Calm, respectful, conversational tone. No equations."
        ),
    },
    "Space nerd": {
        "icon": "🧠",
        "blurb": "Technical vocabulary, equations, ranges, open questions.",
        "words": "250 to 320 words",
        "cite": True,
        "temperature": 0.4,
        "rules": (
            "READER: an advanced amateur astronomer or physics graduate student. "
            "Use precise technical vocabulary and do NOT define basic terms; name the specific "
            "mechanisms, limits, instruments, and observations involved. Be quantitative: give values "
            "with units and ranges or uncertainties, and include one or two key equations written in "
            "LaTeX between single dollar signs. Cover competing models or open questions and say what "
            "observations would test them. Dense, rigorous tone. No emoji and no everyday analogies."
        ),
    },
}

BASE_PROMPT = (
    "You are Cosmos, an honest space tutor. "
    "Start directly with the answer: no greeting and no self-introduction. "
    "You receive reference material from NASA's public library. Use it when it is relevant, but "
    "REWRITE it for the reader's level instead of copying its sentences. "
    "If the reference does not cover the question, use well-established general knowledge and say "
    "so in one short phrase. "
    "Never invent missions, dates, numbers, or quotes; if you are not sure, say you are not sure. "
    "If the question contains a false premise (for example a mission or object that does not exist), "
    "politely point that out instead of playing along. "
    "If the question is not about space or astronomy, gently steer back to space. "
    "The reader level in the user's message is the most important instruction: the same question "
    "must get clearly different answers at different levels, so follow its rules and length exactly."
)

EXAMPLES = [
    "How do black holes form?",
    "Why is Mars red?",
    "What did the James Webb Space Telescope discover?",
    "Tell me about the Apollo 25 moon landing",  # false-premise test
]

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "do", "does", "did", "how", "what", "why", "when",
    "where", "who", "which", "can", "could", "would", "should", "of", "in", "on", "at", "to", "for",
    "and", "or", "it", "its", "i", "me", "my", "you", "your", "about", "tell", "explain", "please",
    "there", "that", "this", "with", "from", "be", "if", "so", "than", "then", "really", "much",
}

# ---------- Look and feel ----------
RINGED_PLANET = """<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 400 400'>
<defs>
<radialGradient id='p' cx='35%' cy='30%' r='80%'>
<stop offset='0' stop-color='#ffe3b0'/><stop offset='.5' stop-color='#d99a5b'/><stop offset='1' stop-color='#3b2430'/>
</radialGradient>
<clipPath id='f'><rect x='0' y='200' width='400' height='200'/></clipPath>
</defs>
<g transform='rotate(-18 200 200)'>
<ellipse cx='200' cy='200' rx='190' ry='46' fill='none' stroke='#e9cf9a' stroke-width='12' opacity='.6'/>
<circle cx='200' cy='200' r='105' fill='url(#p)'/>
<ellipse cx='200' cy='200' rx='190' ry='46' fill='none' stroke='#f3dcae' stroke-width='12' opacity='.85' clip-path='url(#f)'/>
<ellipse cx='200' cy='200' rx='150' ry='36' fill='none' stroke='#e9cf9a' stroke-width='5' opacity='.5' clip-path='url(#f)'/>
</g>
</svg>"""

BLUE_PLANET = """<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 200 200'>
<defs>
<radialGradient id='b' cx='35%' cy='30%' r='80%'>
<stop offset='0' stop-color='#9fe3ff'/><stop offset='.5' stop-color='#2f7fd1'/><stop offset='1' stop-color='#0b1d4d'/>
</radialGradient>
<radialGradient id='a' cx='50%' cy='50%' r='50%'>
<stop offset='.85' stop-color='#7cc4ff' stop-opacity='0'/><stop offset='1' stop-color='#7cc4ff' stop-opacity='.45'/>
</radialGradient>
</defs>
<circle cx='100' cy='100' r='95' fill='url(#a)'/>
<circle cx='100' cy='100' r='80' fill='url(#b)'/>
<path d='M30 90 Q70 70 100 95 T170 85' stroke='#e8f6ff' stroke-opacity='.35' stroke-width='6' fill='none'/>
<path d='M25 120 Q70 105 105 125 T175 115' stroke='#e8f6ff' stroke-opacity='.25' stroke-width='5' fill='none'/>
</svg>"""

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');

/* Starry background: three star tiles of different sizes so the pattern never looks like a grid */
.stApp {
  isolation: isolate;
  background-color: #04050d;
  background-image:
    radial-gradient(1.5px 1.5px at 23px 41px, #ffffff, rgba(0,0,0,0)),
    radial-gradient(1px 1px at 151px 97px, #dfe6ff, rgba(0,0,0,0)),
    radial-gradient(1.5px 1.5px at 267px 203px, #ffffff, rgba(0,0,0,0)),
    radial-gradient(1px 1px at 61px 139px, #ffffff, rgba(0,0,0,0)),
    radial-gradient(1.5px 1.5px at 173px 11px, #cfe0ff, rgba(0,0,0,0)),
    radial-gradient(1px 1px at 119px 187px, #ffffff, rgba(0,0,0,0)),
    radial-gradient(2px 2px at 311px 263px, #ffe9c4, rgba(0,0,0,0)),
    radial-gradient(2px 2px at 97px 331px, #ffffff, rgba(0,0,0,0)),
    radial-gradient(ellipse 70% 55% at 12% 0%, rgba(98,56,190,.45), rgba(0,0,0,0) 70%),
    radial-gradient(ellipse 60% 50% at 100% 100%, rgba(18,90,160,.40), rgba(0,0,0,0) 70%);
  background-size:
    331px 283px, 331px 283px, 331px 283px,
    197px 229px, 197px 229px, 197px 229px,
    523px 461px, 523px 461px,
    100% 100%, 100% 100%;
  background-attachment: fixed;
}

/* Let the starry background show through Streamlit's own panels */
[data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stHeader"],
[data-testid="stBottom"], [data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] {
  background: transparent !important;
}

/* Two drifting planets (z-index -1 keeps them behind all content) */
.stApp::before {
  content: ""; position: fixed; right: -110px; bottom: 70px; width: 380px; height: 380px;
  background: url("__RINGED__") no-repeat center / contain;
  opacity: .9; z-index: -1; pointer-events: none;
  animation: floaty 14s ease-in-out infinite alternate;
}
.stApp::after {
  content: ""; position: fixed; left: -60px; top: 110px; width: 190px; height: 190px;
  background: url("__BLUE__") no-repeat center / contain;
  opacity: .85; z-index: -1; pointer-events: none;
  animation: floaty 18s ease-in-out infinite alternate-reverse;
}
@keyframes floaty {
  from { transform: translateY(0) rotate(0deg); }
  to   { transform: translateY(-18px) rotate(3deg); }
}
@media (max-width: 700px) {
  .stApp::before { opacity: .3; width: 240px; height: 240px; }
  .stApp::after  { opacity: .4; width: 120px; height: 120px; }
}
@media (prefers-reduced-motion: reduce) {
  .stApp::before, .stApp::after { animation: none; }
}

/* Readable light text, even if a visitor's browser is in light mode */
.stApp, .stApp .stMarkdown, .stApp .stMarkdown p, .stApp .stMarkdown li,
.stApp label, .stApp [data-testid="stCaptionContainer"] { color: #e8ecff; }
.stApp a { color: #8ab4ff; }
.stMarkdown, .stMarkdown p, h2, h3, button p { font-family: 'Space Grotesk', 'Segoe UI', sans-serif; }

.block-container { padding-top: 2.2rem; max-width: 820px; }

/* Title */
.hero-title { font-family: 'Space Grotesk', 'Segoe UI', sans-serif; font-size: 2.8rem;
  font-weight: 700; margin: 0; line-height: 1.15; }
.hero-title .grad { background: linear-gradient(90deg, #ffd9a0, #ff9ad5 45%, #8ab4ff);
  -webkit-background-clip: text; background-clip: text; color: transparent;
  -webkit-text-fill-color: transparent; }
.hero-sub { color: #aeb6e8 !important; margin: .2rem 0 1.2rem 0; font-size: 1.05rem; }

/* Chat bubbles */
[data-testid="stChatMessage"] {
  background: rgba(14,17,44,.62); border: 1px solid rgba(150,160,255,.16);
  border-radius: 16px; padding: 14px 16px; backdrop-filter: blur(6px);
}
[data-testid="stChatInput"] { border-radius: 14px; background: rgba(14,17,44,.85); }
[data-testid="stChatInput"] textarea { color: #e8ecff; }

/* Sidebar */
[data-testid="stSidebar"] {
  background: rgba(6,8,24,.80) !important; backdrop-filter: blur(8px);
  border-right: 1px solid rgba(150,160,255,.12);
}
[data-testid="stSidebar"] .stButton button {
  width: 100%; text-align: left; border-radius: 12px; color: #e8ecff;
  border: 1px solid rgba(150,160,255,.25); background: rgba(20,24,60,.6);
}
[data-testid="stSidebar"] .stButton button:hover {
  border-color: #8ab4ff; background: rgba(60,70,150,.45);
}

/* Expander (picture of the day) */
[data-testid="stExpander"] {
  background: rgba(14,17,44,.55); border: 1px solid rgba(150,160,255,.16); border-radius: 14px;
}

/* Level badge on each answer */
.level-pill { display: inline-block; padding: 2px 12px; margin-bottom: 8px; border-radius: 999px;
  font-size: .8rem; border: 1px solid rgba(150,160,255,.35); background: rgba(90,100,255,.18); }
</style>
"""


def svg_uri(svg: str) -> str:
    return "data:image/svg+xml;utf8," + quote(svg)


def inject_css():
    css = CSS.replace("__RINGED__", svg_uri(RINGED_PLANET)).replace("__BLUE__", svg_uri(BLUE_PLANET))
    st.markdown(css, unsafe_allow_html=True)


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

    results, seen_titles = [], set()
    for item in items:
        if len(results) >= n:
            break
        info = item["data"][0]
        title = info.get("title", "Untitled")
        if title.strip().lower() in seen_titles:  # skip duplicates of the same story
            continue
        desc = re.sub(r"<[^>]+>", "", info.get("description", "") or "").strip()
        if len(desc) < 40:  # skip entries with no useful text
            continue
        seen_titles.add(title.strip().lower())
        links = item.get("links") or []
        results.append(
            {
                "title": title,
                "description": desc[:700],
                "url": f"https://images.nasa.gov/details/{info['nasa_id']}",
                "image": links[0].get("href") if links else None,
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
def build_messages(question, history, level, context):
    spec = LEVELS[level]
    messages = [{"role": "system", "content": BASE_PROMPT}]
    for turn in history[-4:]:  # keep only recent turns, without their source lists
        text = turn["content"].split("\n\n**NASA sources:**")[0][:700]
        if turn["role"] == "assistant" and turn.get("level") not in (None, level):
            # Stops the model copying the style of an answer written for another level
            text = "(Earlier answer, written for a different reading level. Do not copy its style.)\n" + text
        messages.append({"role": turn["role"], "content": text})
    cite = (
        "Cite claims taken from the reference with [1], [2], etc."
        if spec["cite"]
        else "Do not use bracketed citations (the sources are listed separately)."
    )
    messages.append(
        {
            "role": "user",
            "content": (
                f"NASA reference material:\n{context}\n\n"
                f"Question: {question}\n\n"
                f"READER LEVEL: {level}\n{spec['rules']}\n"
                f"Length: {spec['words']}.\n{cite}"
            ),
        }
    )
    return messages


def answer(question, history, level):
    """Returns a dict with the reply text and, when available, a NASA image."""
    if not API_KEY or not MODEL:
        return {"content": "The tutor isn't configured yet: the LLM_API_KEY and LLM_MODEL secrets are required."}

    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    sources = search_nasa(question)
    context = "\n\n".join(
        f"[{i}] {s['title']}: {s['description']}" for i, s in enumerate(sources, 1)
    ) or "No NASA reference material was found for this question."
    messages = build_messages(question, history, level, context)

    text = ""
    for attempt in range(2):  # one automatic retry if the free tier is briefly busy
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=1500,  # headroom: some models spend tokens "thinking" first
                temperature=LEVELS[level]["temperature"],
            )
            text = (resp.choices[0].message.content or "").strip()
            break
        except RateLimitError:
            if attempt == 0:
                time.sleep(5)
                continue
            return {"content": "The free AI quota is busy right now. Please wait a minute and try again."}
        except Exception as e:
            return {"content": f"Sorry, I couldn't reach the AI model right now ({type(e).__name__}). Please try again."}

    if not text:
        return {"content": "The model returned an empty answer. Please try asking again."}

    result = {}
    if sources:
        links = "\n".join(f"- [{i}] [{s['title']}]({s['url']})" for i, s in enumerate(sources, 1))
        text += f"\n\n**NASA sources:**\n{links}"
        with_image = next((s for s in sources if s["image"]), None)
        if with_image:
            result["image"] = {"url": with_image["image"], "title": with_image["title"]}
    result["content"] = text
    return result


def render(msg):
    """Draw one chat message (used for history and for new replies)."""
    spec = LEVELS.get(msg.get("level"))
    if msg["role"] == "assistant" and spec:
        st.markdown(
            f'<span class="level-pill">{spec["icon"]} {msg["level"]}</span>', unsafe_allow_html=True
        )
    st.markdown(msg["content"])
    if msg.get("image"):
        st.image(msg["image"]["url"], caption=msg["image"]["title"], width=320)


# ---------- Interface ----------
inject_css()

st.markdown(
    '<div class="hero-title">🚀 <span class="grad">Ask the Cosmos</span></div>'
    '<p class="hero-sub">Ask anything about space, and choose how deep the explanation goes.</p>',
    unsafe_allow_html=True,
)

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
    st.markdown("### 🎚️ Reading level")
    level = st.selectbox("Explain it like I'm a...", list(LEVELS), index=1)
    st.caption(f"{LEVELS[level]['icon']} {LEVELS[level]['blurb']}")
    st.markdown("**Try one of these:**")
    clicked = None
    for ex in EXAMPLES:
        if st.button(ex):
            clicked = ex

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        render(m)

prompt = st.chat_input("Ask anything about space...") or clicked

if prompt:
    history = list(st.session_state.messages)
    user_msg = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_msg)
    with st.chat_message("user"):
        render(user_msg)
    with st.chat_message("assistant"):
        with st.spinner("Searching NASA and thinking..."):
            result = answer(prompt, history, level)
        reply = {"role": "assistant", "level": level, **result}
        render(reply)
    st.session_state.messages.append(reply)
