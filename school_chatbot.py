import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import json
from sarvamai import SarvamAI

with open('school_website_rag.json', 'r') as f:
  texts = [sentence['text'] for sentence in json.load(f)['texts']]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
index = faiss.read_index('school_vector_db.index')
client = SarvamAI(api_subscription_key = st.secrets['SARVAM_API_KEY'])

if "messages" not in st.session_state:
    st.session_state.messages = []

def reply(prompt):
 scores, indices = index.search(model.encode([prompt]), k = 5)

 context = ''

 for idx in indices.squeeze():
  context += texts[idx] + '\n\n\n'

 system_prompt = f"given the context -  {context}\n\n\nThe chat history - {st.session_state.messages}\n\n\nreply to the prompt {prompt}. If the prompt is completely unrelated to the context, just reply school stuff only. Don't mention anything related to the given context in the reply"

 ai_response = client.chat.completions(
    model = "sarvam-105b",
    messages = [
        {"role": "user", "content": system_prompt}
    ],
    reasoning_effort = None,
 ).choices[0].message.content

 st.session_state.messages.append({
        "role": "user",
        "avatar": "👤",
        "content": prompt
    })
 
 st.session_state.messages.append({
        "role": "school assistant",
        "avatar": "🤖",
        "content": ai_response
    })

st.write("")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Caveat:wght@600&display=swap');

:root {
    --indigo: #4f46e5;
    --violet: #7c3aed;
    --sky: #0ea5e9;
    --amber: #f59e0b;
    --mint: #10b981;
    --ink: #1e1b4b;
    --paper: #f8faff;
    --card: rgba(255,255,255,0.78);
}

html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Notebook-paper style background with soft animated glow */
.stApp {
    background:
        radial-gradient(circle at 15% 10%, rgba(124,58,237,0.12), transparent 40%),
        radial-gradient(circle at 85% 20%, rgba(14,165,233,0.12), transparent 40%),
        repeating-linear-gradient(0deg, transparent 0 31px, rgba(79,70,229,0.06) 31px 32px),
        var(--paper);
    background-attachment: fixed;
}

.block-container { padding-top: 1.5rem; max-width: 980px; }

/* ---------------- HERO ---------------- */
.hero {
    position: relative;
    overflow: hidden;
    border-radius: 28px;
    padding: 2.2rem 2.4rem;
    color: #fff;
    background: linear-gradient(120deg, #4f46e5, #7c3aed, #0ea5e9, #4f46e5);
    background-size: 300% 300%;
    animation: gradientShift 12s ease infinite, fadeDown .8s ease both;
    box-shadow: 0 20px 50px -15px rgba(79,70,229,.55);
    margin-bottom: 1.2rem;
}
.hero h1 {
    margin: 0; font-size: 2.2rem; font-weight: 700; letter-spacing: -.5px;
}
.hero p { margin: .4rem 0 0; opacity: .92; font-size: 1.02rem; }
.hero .tagline {
    font-family: 'Caveat', cursive; font-size: 1.5rem; opacity: .95; margin-top: .6rem;
}
.hero .logo {
    width: 78px; height: 78px; border-radius: 22px; object-fit: cover;
    border: 3px solid rgba(255,255,255,.6);
    box-shadow: 0 8px 24px rgba(0,0,0,.25);
    animation: bob 4s ease-in-out infinite;
    float: right; margin-left: 1rem;
}
.float {
    position: absolute; font-size: 2rem; opacity: .35; user-select: none;
    animation: floaty 7s ease-in-out infinite;
}
.f1 { top: 12%; right: 24%; animation-delay: 0s; }
.f2 { bottom: 10%; right: 8%;  animation-delay: 1.2s; font-size: 2.6rem; }
.f3 { top: 55%; right: 36%; animation-delay: 2.4s; }
.f4 { top: 8%;  right: 5%;  animation-delay: 3.4s; }

/* ---------------- STAT CARDS ---------------- */
.stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: .8rem; margin-bottom: 1.2rem; }
.stat {
    background: var(--card);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,.9);
    border-radius: 20px; padding: .9rem 1rem;
    box-shadow: 0 8px 24px -12px rgba(30,27,75,.25);
    transition: transform .25s ease, box-shadow .25s ease;
    animation: fadeUp .7s ease both;
}
.stat:nth-child(2){animation-delay:.08s} .stat:nth-child(3){animation-delay:.16s} .stat:nth-child(4){animation-delay:.24s}
.stat:hover { transform: translateY(-6px) scale(1.02); box-shadow: 0 18px 30px -12px rgba(79,70,229,.4); }
.stat .ico { font-size: 1.5rem; }
.stat .lbl { font-size: .75rem; color: #64748b; margin-top: .2rem; }
.stat .val { font-weight: 600; color: var(--ink); }

/* ---------------- SUBJECT CHIP BUTTONS ---------------- */
div[data-testid="stButton"] > button {
    width: 100%;
    border-radius: 16px;
    border: 1px solid rgba(79,70,229,.18);
    background: #fff;
    color: var(--ink);
    font-weight: 500;
    padding: .65rem .4rem;
    transition: all .25s cubic-bezier(.2,.8,.2,1);
    box-shadow: 0 4px 14px -8px rgba(30,27,75,.3);
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-4px);
    background: linear-gradient(120deg, var(--indigo), var(--violet));
    color: #fff; border-color: transparent;
    box-shadow: 0 14px 24px -10px rgba(79,70,229,.6);
}
div[data-testid="stButton"] > button:active { transform: scale(.96); }

/* ---------------- CHAT ---------------- */
.stChatMessage {
    border-radius: 22px;
    padding: 1rem 1.2rem;
    margin-bottom: .8rem;
    border: 1px solid rgba(255,255,255,.9);
    background: var(--card);
    backdrop-filter: blur(10px);
    box-shadow: 0 10px 28px -16px rgba(30,27,75,.35);
    animation: popIn .45s cubic-bezier(.2,.9,.3,1.2) both;
}
/* user bubble */
.stChatMessage:has([data-testid="stChatMessageAvatarUser"]) {
    background: linear-gradient(120deg, #4f46e5, #7c3aed);
    color: #fff;
    margin-left: 12%;
}
.stChatMessage:has([data-testid="stChatMessageAvatarUser"]) * { color: #fff !important; }
/* assistant bubble */
.stChatMessage:has([data-testid="stChatMessageAvatarAssistant"]) {
    margin-right: 8%;
    border-left: 5px solid var(--amber);
}

/* ---------------- EMPTY STATE ---------------- */
.welcome {
    text-align: center; padding: 1.6rem 1rem; color: #475569;
    animation: fadeUp .8s ease both;
}
.welcome .big { font-size: 3rem; display: inline-block; animation: bob 3s ease-in-out infinite; }
.welcome h3 { color: var(--ink); margin: .4rem 0 .2rem; }
.dots span {
    display: inline-block; width: 8px; height: 8px; margin: 0 2px; border-radius: 50%;
    background: var(--indigo); animation: bounce 1.2s infinite ease-in-out;
}
.dots span:nth-child(2){animation-delay:.15s} .dots span:nth-child(3){animation-delay:.3s}

/* ---------------- CHAT INPUT ---------------- */
div[data-testid="stChatInput"] {
    border-radius: 24px;
    box-shadow: 0 12px 40px -12px rgba(79,70,229,.45);
    border: 2px solid transparent;
    background:
        linear-gradient(#fff,#fff) padding-box,
        linear-gradient(120deg, var(--indigo), var(--sky), var(--violet)) border-box;
    transition: transform .25s ease, box-shadow .25s ease;
}
div[data-testid="stChatInput"]:focus-within {
    transform: translateY(-3px);
    box-shadow: 0 18px 50px -12px rgba(124,58,237,.6);
}

/* ---------------- SIDEBAR ---------------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e1b4b 0%, #312e81 60%, #4338ca 100%);
}
section[data-testid="stSidebar"] * { color: #e0e7ff !important; }
.side-card {
    background: rgba(255,255,255,.08);
    border: 1px solid rgba(255,255,255,.15);
    border-radius: 18px; padding: .9rem 1rem; margin-bottom: .8rem;
    animation: fadeUp .6s ease both;
    transition: transform .2s ease, background .2s ease;
}
.side-card:hover { transform: translateX(6px); background: rgba(255,255,255,.14); }
.side-card b { color: #fff !important; }
.side-title { font-family: 'Caveat', cursive; font-size: 1.7rem; margin-bottom: .6rem; }

/* ---------------- KEYFRAMES ---------------- */
@keyframes gradientShift { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
@keyframes fadeDown { from{opacity:0; transform:translateY(-24px)} to{opacity:1; transform:none} }
@keyframes fadeUp   { from{opacity:0; transform:translateY(24px)}  to{opacity:1; transform:none} }
@keyframes popIn    { from{opacity:0; transform:translateY(18px) scale(.94)} to{opacity:1; transform:none} }
@keyframes floaty   { 0%,100%{transform:translateY(0) rotate(0)} 50%{transform:translateY(-18px) rotate(12deg)} }
@keyframes bob      { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-8px)} }
@keyframes bounce   { 0%,80%,100%{transform:scale(.6); opacity:.4} 40%{transform:scale(1.1); opacity:1} }

@media (max-width: 700px) {
    .stats { grid-template-columns: repeat(2, 1fr); }
    .hero h1 { font-size: 1.6rem; }
    .stChatMessage:has([data-testid="stChatMessageAvatarUser"]) { margin-left: 0; }
}
</style>
"""

st.markdown(CSS, unsafe_allow_html = True)

# ------------------------------------------------------------------
# CHAT HISTORY (rendered first so order stays correct)
# ------------------------------------------------------------------
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome">
        <div class="big">🎒</div>
        <h3>Hi there, parent! 👋</h3>
    </div>
    """, unsafe_allow_html = True)

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar = message.get("avatar")):
        st.markdown(message["content"])

# ------------------------------------------------------------------
# INPUT
# ------------------------------------------------------------------
prompt = st.chat_input("Ask anything about presidium school...") 
if prompt:
    with st.spinner("PresidiumSchoolBuddy is thinking... "):
        reply(prompt)
    st.rerun()
st.write("")
