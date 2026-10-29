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
 scores, indices = index.search(model.encode([prompt]), k = 10)

 context = ''

 for score, idx in zip(scores, indices.squeeze()):
  print(score)
  if float(score) < 0.5:
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




st.set_page_config(
    page_title="Presidium School Buddy",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

def ui(html):
    # Flatten HTML so Markdown never treats indented lines / blank lines as a code block
    return "".join(line.strip() for line in html.splitlines())
 
LOGO_URL = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQtI1K_OG63uKi0XhRXNTEcrni4IGHHq4llRhVgKmEcWJe6ncaaeZwkOA8&s=10"
 
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Caveat:wght@600;700&display=swap');
 
:root {
    --indigo:#4f46e5; --violet:#7c3aed; --sky:#0ea5e9; --amber:#f59e0b;
    --mint:#10b981; --rose:#f43f5e; --ink:#1e1b4b; --paper:#f6f8ff;
    --card:rgba(255,255,255,0.80);
}
html, body, [class*="css"] { font-family:'Poppins',sans-serif; }
 
/* Streamlit chrome */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display:none !important; }
header[data-testid="stHeader"] { background:transparent; }
 
/* Background: ruled notebook + drifting colour blobs */
.stApp {
    background:
        radial-gradient(circle at 12% 8%,  rgba(124,58,237,.14), transparent 42%),
        radial-gradient(circle at 88% 18%, rgba(14,165,233,.14), transparent 42%),
        radial-gradient(circle at 50% 100%, rgba(245,158,11,.10), transparent 45%),
        repeating-linear-gradient(0deg, transparent 0 31px, rgba(79,70,229,.055) 31px 32px),
        var(--paper);
    background-attachment: fixed;
}
.block-container { padding-top:1.2rem; max-width:1040px; }
 
/* ================= HERO ================= */
.hero {
    position:relative; overflow:hidden; border-radius:32px;
    padding:2.4rem 2.6rem; color:#fff;
    display:grid; grid-template-columns:1.25fr 1fr; align-items:center; gap:1rem;
    background:linear-gradient(120deg,#4f46e5,#7c3aed,#0ea5e9,#4f46e5);
    background-size:300% 300%;
    animation:gradientShift 14s ease infinite, fadeDown .8s ease both;
    box-shadow:0 26px 60px -20px rgba(79,70,229,.6);
    margin-bottom:1.3rem;
}
.hero::before, .hero::after {
    content:""; position:absolute; border-radius:50%; background:rgba(255,255,255,.10);
    animation:drift 12s ease-in-out infinite;
}
.hero::before { width:260px; height:260px; top:-90px; left:-60px; }
.hero::after  { width:200px; height:200px; bottom:-80px; right:30%; animation-delay:3s; }
.hero-left { position:relative; z-index:2; }
.badge {
    display:inline-flex; align-items:center; gap:.5rem;
    background:rgba(255,255,255,.18); border:1px solid rgba(255,255,255,.35);
    padding:.3rem .8rem; border-radius:999px; font-size:.78rem; font-weight:500;
    backdrop-filter:blur(8px);
}
.badge i { width:8px; height:8px; border-radius:50%; background:#34d399; animation:pulse 1.6s infinite; }
.hero h1 { margin:.8rem 0 0; font-size:2.5rem; line-height:1.1; font-weight:800; letter-spacing:-.8px; }
.hero h1 span { background:linear-gradient(90deg,#fde68a,#fff); -webkit-background-clip:text; background-clip:text; color:transparent; }
.hero p { margin:.7rem 0 0; opacity:.93; font-size:1rem; max-width:470px; }
.hero .hand { font-family:'Caveat',cursive; font-size:1.6rem; margin-top:.7rem; opacity:.97; }
.logo-pill {
    display:inline-flex; align-items:center; gap:.7rem; margin-top:1rem;
    background:rgba(255,255,255,.16); padding:.4rem .9rem .4rem .4rem; border-radius:18px;
    border:1px solid rgba(255,255,255,.3); font-weight:600; font-size:.9rem;
}
.logo-pill img { width:42px; height:42px; border-radius:12px; object-fit:cover; }
.float { position:absolute; font-size:2rem; opacity:.32; user-select:none; animation:floaty 7s ease-in-out infinite; z-index:1; }
.f1{top:10%; right:46%; animation-delay:0s}
.f2{bottom:8%; right:4%; font-size:2.6rem; animation-delay:1.2s}
.f3{top:62%; right:48%; animation-delay:2.4s}
.f4{top:7%; right:4%; animation-delay:3.4s}
.f5{bottom:12%; left:48%; animation-delay:4.2s}
 
/* ================= ROBOT ================= */
.robot-wrap { position:relative; z-index:2; height:290px; display:flex; justify-content:center; align-items:flex-end; }
.say {
    position:absolute; top:0; right:6%; min-width:190px; height:46px;
    background:#fff; color:var(--ink); border-radius:18px 18px 18px 4px;
    box-shadow:0 12px 26px -10px rgba(0,0,0,.35);
    font-size:.82rem; font-weight:600; animation:bob 4s ease-in-out infinite;
}
.say span { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; padding:0 .9rem; opacity:0; animation:sayCycle 12s infinite; }
.say span:nth-child(2){animation-delay:3s} .say span:nth-child(3){animation-delay:6s} .say span:nth-child(4){animation-delay:9s}
.robot { position:relative; width:150px; height:230px; animation:hover 3.6s ease-in-out infinite; }
.antenna { position:absolute; left:50%; top:0; width:4px; height:26px; background:#c7d2fe; transform:translateX(-50%); border-radius:4px; }
.antenna i { position:absolute; top:-12px; left:50%; width:16px; height:16px; margin-left:-8px; border-radius:50%; background:var(--amber); box-shadow:0 0 0 0 rgba(245,158,11,.7); animation:ping 1.8s infinite; }
.head {
    position:absolute; top:24px; left:50%; width:118px; height:92px; margin-left:-59px;
    background:linear-gradient(160deg,#ffffff,#e0e7ff); border-radius:34px;
    box-shadow:inset 0 -8px 14px rgba(79,70,229,.18), 0 14px 24px -10px rgba(0,0,0,.4);
}
.head::before, .head::after { content:""; position:absolute; top:34px; width:12px; height:26px; border-radius:8px; background:#a5b4fc; }
.head::before { left:-8px; } .head::after { right:-8px; }
.visor { position:absolute; left:12px; right:12px; top:16px; height:52px; border-radius:24px; background:linear-gradient(160deg,#1e1b4b,#312e81); display:flex; justify-content:center; align-items:center; gap:20px; }
.eye { width:16px; height:20px; border-radius:50%; background:#67e8f9; box-shadow:0 0 12px #22d3ee; animation:blink 4.5s infinite; }
.mouth { position:absolute; bottom:8px; left:50%; width:22px; height:6px; margin-left:-11px; border-radius:0 0 12px 12px; border-bottom:3px solid #6366f1; }
.torso {
    position:absolute; top:112px; left:50%; width:92px; height:84px; margin-left:-46px;
    background:linear-gradient(160deg,#ffffff,#c7d2fe); border-radius:26px 26px 32px 32px;
    box-shadow:inset 0 -8px 14px rgba(79,70,229,.2), 0 14px 24px -10px rgba(0,0,0,.4);
}
.chest { position:absolute; left:50%; top:18px; width:34px; height:34px; margin-left:-17px; border-radius:50%; background:linear-gradient(135deg,var(--violet),var(--sky)); display:flex; align-items:center; justify-content:center; color:#fff; font-size:.95rem; animation:heart 1.4s infinite; }
.arm { position:absolute; top:122px; width:18px; height:62px; border-radius:12px; background:linear-gradient(160deg,#fff,#a5b4fc); transform-origin:50% 8px; box-shadow:0 8px 14px -8px rgba(0,0,0,.4); }
.arm.l { left:6px; animation:swingL 3.6s ease-in-out infinite; }
.arm.r { right:6px; animation:wave 1.6s ease-in-out infinite; }
.arm.r::after { content:"✋"; position:absolute; bottom:-18px; left:-5px; font-size:1.4rem; }
.shadow { position:absolute; bottom:-12px; left:50%; width:110px; height:16px; margin-left:-55px; border-radius:50%; background:rgba(0,0,0,.28); filter:blur(6px); animation:shadow 3.6s ease-in-out infinite; }
 
/* ================= SECTION TITLES ================= */
.sec { display:flex; align-items:center; gap:.6rem; margin:1.4rem 0 .8rem; animation:fadeUp .7s ease both; }
.sec h2 { margin:0; font-size:1.25rem; font-weight:700; color:var(--ink); }
.sec .line { flex:1; height:2px; background:linear-gradient(90deg,rgba(79,70,229,.35),transparent); border-radius:2px; }
.sec .sub { font-family:'Caveat',cursive; font-size:1.2rem; color:var(--violet); }
 
/* ================= LEARNING JOURNEY ================= */
.journey { display:grid; grid-template-columns:repeat(5,1fr); gap:.7rem; position:relative; }
.stage {
    background:var(--card); backdrop-filter:blur(10px); border:1px solid #fff;
    border-radius:22px; padding:1rem .8rem; text-align:center; position:relative; overflow:hidden;
    box-shadow:0 10px 26px -14px rgba(30,27,75,.3);
    transition:transform .3s cubic-bezier(.2,.9,.3,1.3), box-shadow .3s ease; animation:fadeUp .7s ease both;
}
.stage:nth-child(2){animation-delay:.08s} .stage:nth-child(3){animation-delay:.16s} .stage:nth-child(4){animation-delay:.24s} .stage:nth-child(5){animation-delay:.32s}
.stage::before { content:""; position:absolute; top:0; left:0; right:0; height:5px; background:var(--c); }
.stage:hover { transform:translateY(-8px) rotate(-1deg); box-shadow:0 22px 34px -14px var(--c); }
.stage .ico { font-size:2rem; display:inline-block; transition:transform .4s; }
.stage:hover .ico { transform:scale(1.25) rotate(10deg); }
.stage b { display:block; color:var(--ink); margin-top:.3rem; font-size:.92rem; }
.stage small { color:#64748b; font-size:.72rem; display:block; line-height:1.35; margin-top:.15rem; }
.stage .age { display:inline-block; margin-top:.5rem; font-size:.68rem; font-weight:600; color:#fff; background:var(--c); padding:.1rem .6rem; border-radius:999px; }
 
/* ================= PILLARS ================= */
.pillars { display:grid; grid-template-columns:repeat(4,1fr); gap:.8rem; }
.pillar {
    background:var(--card); backdrop-filter:blur(10px); border:1px solid #fff; border-radius:22px;
    padding:1rem; box-shadow:0 10px 26px -14px rgba(30,27,75,.3);
    transition:transform .25s ease, box-shadow .25s ease; animation:fadeUp .8s ease both;
}
.pillar:hover { transform:translateY(-6px) scale(1.02); box-shadow:0 20px 30px -14px rgba(79,70,229,.45); }
.pillar .ico { width:44px; height:44px; border-radius:14px; display:flex; align-items:center; justify-content:center; font-size:1.4rem; background:var(--bg); margin-bottom:.5rem; animation:bob 4s ease-in-out infinite; }
.pillar b { color:var(--ink); font-size:.95rem; }
.pillar p { margin:.2rem 0 0; font-size:.76rem; color:#64748b; line-height:1.4; }
 
/* ================= QUICK BUTTONS ================= */
div[data-testid="stButton"] > button {
    width:100%; border-radius:16px; border:1px solid rgba(79,70,229,.2); background:#fff;
    color:var(--ink); font-weight:500; font-size:.85rem; padding:.65rem .4rem;
    transition:all .25s cubic-bezier(.2,.8,.2,1); box-shadow:0 4px 14px -8px rgba(30,27,75,.35);
}
div[data-testid="stButton"] > button:hover {
    transform:translateY(-4px); background:linear-gradient(120deg,var(--indigo),var(--violet));
    color:#fff; border-color:transparent; box-shadow:0 14px 24px -10px rgba(79,70,229,.6);
}
div[data-testid="stButton"] > button:active { transform:scale(.95); }
 
/* ================= CHAT ================= */
.chat-head {
    display:flex; align-items:center; gap:.9rem; padding:.8rem 1.1rem; margin:1.2rem 0 1rem;
    background:linear-gradient(120deg,#1e1b4b,#4338ca); color:#fff; border-radius:22px;
    box-shadow:0 16px 34px -16px rgba(30,27,75,.7); animation:fadeUp .7s ease both;
}
.mini-bot { width:46px; height:46px; border-radius:16px; background:linear-gradient(160deg,#fff,#c7d2fe); position:relative; animation:bob 3s ease-in-out infinite; flex-shrink:0; }
.mini-bot::before { content:""; position:absolute; inset:9px 6px 9px 6px; border-radius:10px; background:#1e1b4b; }
.mini-bot::after { content:""; position:absolute; top:19px; left:13px; width:6px; height:8px; border-radius:50%; background:#67e8f9; box-shadow:14px 0 0 #67e8f9, 0 0 8px #22d3ee; animation:blink 4s infinite; }
.chat-head b { font-size:1rem; display:block; }
.chat-head small { opacity:.8; font-size:.75rem; display:flex; align-items:center; gap:.4rem; }
.chat-head small i { width:7px; height:7px; border-radius:50%; background:#34d399; animation:pulse 1.6s infinite; }
.chat-head .typing { margin-left:auto; }
 
/* hidden role markers */
div[data-testid="stElementContainer"]:has(.mk-user), div[data-testid="stElementContainer"]:has(.mk-bot),
.element-container:has(.mk-user), .element-container:has(.mk-bot) { display:none !important; }
 
.stChatMessage {
    border-radius:22px; padding:1rem 1.2rem; margin-bottom:.8rem;
    border:1px solid rgba(255,255,255,.9); background:var(--card); backdrop-filter:blur(10px);
    box-shadow:0 10px 28px -16px rgba(30,27,75,.35);
    animation:popIn .45s cubic-bezier(.2,.9,.3,1.2) both;
}
.stChatMessage:has(.mk-user) { background:linear-gradient(120deg,#4f46e5,#7c3aed); margin-left:12%; }
.stChatMessage:has(.mk-user) * { color:#fff !important; }
.stChatMessage:has(.mk-bot) { margin-right:8%; border-left:5px solid var(--amber); }
.stChatMessage [data-testid^="stChatMessageAvatar"] { background:rgba(255,255,255,.92); border-radius:50%; box-shadow:0 4px 10px -4px rgba(0,0,0,.3); }
 
/* empty state */
.welcome { text-align:center; padding:1.4rem 1rem 1rem; color:#475569; animation:fadeUp .8s ease both; }
.welcome h3 { color:var(--ink); margin:.2rem 0; font-weight:700; }
.dots span { display:inline-block; width:8px; height:8px; margin:0 2px; border-radius:50%; background:var(--indigo); animation:bounce 1.2s infinite ease-in-out; }
.dots span:nth-child(2){animation-delay:.15s} .dots span:nth-child(3){animation-delay:.3s}
 
/* input */
div[data-testid="stChatInput"] {
    border-radius:26px; border:2px solid transparent;
    background:linear-gradient(#fff,#fff) padding-box, linear-gradient(120deg,var(--indigo),var(--sky),var(--violet)) border-box;
    box-shadow:0 14px 40px -14px rgba(79,70,229,.5); transition:transform .25s ease, box-shadow .25s ease;
}
div[data-testid="stChatInput"]:focus-within { transform:translateY(-3px); box-shadow:0 20px 50px -14px rgba(124,58,237,.65); }
div[data-testid="stSpinner"] { color:var(--indigo); font-weight:500; }
 
/* ================= SIDEBAR ================= */
section[data-testid="stSidebar"] { background:linear-gradient(180deg,#1e1b4b 0%,#312e81 60%,#4338ca 100%); }
section[data-testid="stSidebar"] * { color:#e0e7ff !important; }
.side-title { font-family:'Caveat',cursive; font-size:1.8rem; margin-bottom:.6rem; }
.side-card {
    background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.16); border-radius:18px;
    padding:.8rem 1rem; margin-bottom:.7rem; animation:fadeUp .6s ease both;
    transition:transform .2s ease, background .2s ease;
}
.side-card:hover { transform:translateX(6px); background:rgba(255,255,255,.15); }
.side-card b { color:#fff !important; }
.side-card small { opacity:.85; }
 
/* ================= KEYFRAMES ================= */
@keyframes gradientShift { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
@keyframes fadeDown { from{opacity:0; transform:translateY(-26px)} to{opacity:1; transform:none} }
@keyframes fadeUp   { from{opacity:0; transform:translateY(26px)}  to{opacity:1; transform:none} }
@keyframes popIn    { from{opacity:0; transform:translateY(18px) scale(.94)} to{opacity:1; transform:none} }
@keyframes floaty   { 0%,100%{transform:translateY(0) rotate(0)} 50%{transform:translateY(-18px) rotate(12deg)} }
@keyframes drift    { 0%,100%{transform:translate(0,0)} 50%{transform:translate(30px,20px)} }
@keyframes bob      { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-8px)} }
@keyframes hover    { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-14px)} }
@keyframes shadow   { 0%,100%{transform:scale(1); opacity:.28} 50%{transform:scale(.78); opacity:.16} }
@keyframes bounce   { 0%,80%,100%{transform:scale(.6); opacity:.4} 40%{transform:scale(1.1); opacity:1} }
@keyframes pulse    { 0%,100%{box-shadow:0 0 0 0 rgba(52,211,153,.7)} 70%{box-shadow:0 0 0 8px rgba(52,211,153,0)} }
@keyframes ping     { 0%{box-shadow:0 0 0 0 rgba(245,158,11,.75)} 70%{box-shadow:0 0 0 14px rgba(245,158,11,0)} 100%{box-shadow:0 0 0 0 rgba(245,158,11,0)} }
@keyframes blink    { 0%,92%,100%{transform:scaleY(1)} 95%{transform:scaleY(.08)} }
@keyframes heart    { 0%,100%{transform:scale(1)} 50%{transform:scale(1.18)} }
@keyframes wave     { 0%,100%{transform:rotate(-8deg)} 25%{transform:rotate(-50deg)} 50%{transform:rotate(-8deg)} 75%{transform:rotate(-50deg)} }
@keyframes swingL   { 0%,100%{transform:rotate(6deg)} 50%{transform:rotate(-4deg)} }
@keyframes sayCycle { 0%{opacity:0; transform:translateY(8px)} 4%,22%{opacity:1; transform:none} 26%,100%{opacity:0; transform:translateY(-6px)} }
 
@media (max-width:820px) {
    .hero { grid-template-columns:1fr; padding:1.6rem; }
    .hero h1 { font-size:1.8rem; }
    .robot-wrap { height:260px; }
    .journey { grid-template-columns:repeat(2,1fr); }
    .pillars { grid-template-columns:repeat(2,1fr); }
    .stChatMessage:has(.mk-user) { margin-left:0; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
 
# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="side-title">🎒 Parent Corner</div>', unsafe_allow_html=True)
    st.markdown(ui("""
    <div class="side-card"><b>🤖 Presidium Buddy</b><br><small>Ask me anything about admissions, academics, activities and campus life.</small></div>
    <div class="side-card"><b>💡 Tip</b><br><small>Be specific. Mention a grade or topic for sharper answers.</small></div>
    <div class="side-card"><b>🕒 Available</b><br><small>24 x 7, instant replies</small></div>
    """), unsafe_allow_html=True)
    if st.button("🧹 Start new chat"):
        st.session_state.messages = []
        st.rerun()
 
# ------------------------------------------------------------------
# HERO + ROBOT
# ------------------------------------------------------------------
hero = ui("""
<div class="hero">
    <span class="float f1">📚</span><span class="float f2">🔬</span><span class="float f3">✏️</span>
    <span class="float f4">📐</span><span class="float f5">🎨</span>
 
    <div class="hero-left">
        <span class="badge"><i></i> AI School Assistant · Online</span>
        <h1>Meet your <span>Presidium School Buddy</span></h1>
        <p>Instant answers about admissions, academics, activities and everything in between, without the phone calls.</p>
        <div class="hand">Learn today, lead tomorrow ✨</div>
        <div class="logo-pill"><img src="__LOGO__" alt="logo"> Presidium School</div>
    </div>
 
    <div class="robot-wrap">
        <div class="say">
            <span>Hi! I'm your school buddy 👋</span>
            <span>Ask me about admissions 🎓</span>
            <span>Curious about activities? ⚽</span>
            <span>Let's chat! 💬</span>
        </div>
        <div class="robot">
            <div class="antenna"><i></i></div>
            <div class="head">
                <div class="visor"><b class="eye"></b><b class="eye"></b></div>
                <div class="mouth"></div>
            </div>
            <div class="arm l"></div>
            <div class="torso"><div class="chest">🎓</div></div>
            <div class="arm r"></div>
            <div class="shadow"></div>
        </div>
    </div>
</div>
""".replace("__LOGO__", LOGO_URL))
st.markdown(hero, unsafe_allow_html=True)
 
# ------------------------------------------------------------------
# HIGH LEVEL SCHOOL INFO
# ------------------------------------------------------------------
st.markdown(ui("""
<div class="sec"><h2>🗺️ The School Journey</h2><div class="line"></div><span class="sub">a quick guide</span></div>
<div class="journey">
    <div class="stage" style="--c:#f43f5e"><div class="ico">🧸</div><b>Pre-Primary</b><small>Play-based learning, early literacy &amp; social skills</small><span class="age">~ Ages 3-5</span></div>
    <div class="stage" style="--c:#f59e0b"><div class="ico">🖍️</div><b>Primary</b><small>Reading, writing, numeracy &amp; curiosity</small><span class="age">~ Grades 1-5</span></div>
    <div class="stage" style="--c:#10b981"><div class="ico">🔭</div><b>Middle</b><small>Core subjects &amp; hands-on exploration</small><span class="age">~ Grades 6-8</span></div>
    <div class="stage" style="--c:#0ea5e9"><div class="ico">🧪</div><b>Secondary</b><small>Board-level foundation &amp; study skills</small><span class="age">~ Grades 9-10</span></div>
    <div class="stage" style="--c:#7c3aed"><div class="ico">🎯</div><b>Senior Secondary</b><small>Stream choice &amp; career readiness</small><span class="age">~ Grades 11-12</span></div>
</div>
 
<div class="sec"><h2>🏫 What makes a great school</h2><div class="line"></div><span class="sub">ask me about each</span></div>
<div class="pillars">
    <div class="pillar" style="--bg:#e0e7ff"><div class="ico">📘</div><b>Academics</b><p>Curriculum, teaching approach, assessments and learning support.</p></div>
    <div class="pillar" style="--bg:#fef3c7"><div class="ico">⚽</div><b>Activities</b><p>Sports, arts, clubs and events that build confidence beyond books.</p></div>
    <div class="pillar" style="--bg:#d1fae5"><div class="ico">🛡️</div><b>Safety &amp; Care</b><p>Campus security, wellbeing, transport and a caring environment.</p></div>
    <div class="pillar" style="--bg:#fce7f3"><div class="ico">🤝</div><b>Community</b><p>Parent-teacher engagement, events and open communication.</p></div>
</div>
 
<div class="sec"><h2>⚡ Quick questions</h2><div class="line"></div><span class="sub">tap to ask</span></div>
"""), unsafe_allow_html=True)
 
quick = [
    ("🎓 Admissions", "How does the admission process work?"),
    ("💰 Fees", "What is the fee structure?"),
    ("🏫 Facilities", "What facilities does the school offer?"),
    ("⚽ Activities", "What sports and extracurricular activities are available?"),
    ("🚌 Transport", "Tell me about school transport"),
]
cols = st.columns(len(quick))
clicked = None
for col, (label, text) in zip(cols, quick):
    if col.button(label, key=f"q_{label}"):
        clicked = text
 
# ------------------------------------------------------------------
# CHAT
# ------------------------------------------------------------------
st.markdown(ui("""
<div class="chat-head">
    <div class="mini-bot"></div>
    <div><b>Presidium School Buddy</b><small><i></i> Online · replies instantly</small></div>
    <div class="typing dots"><span></span><span></span><span></span></div>
</div>
"""), unsafe_allow_html=True)
 
if not st.session_state.messages:
    st.markdown(ui("""
    <div class="welcome">
        <h3>Hi there, parent! 👋</h3>
        <div>Pick a quick question above or type your own below.</div>
    </div>
    """), unsafe_allow_html=True)
 
for message in st.session_state.messages:
    marker = "mk-user" if message["role"] == "user" else "mk-bot"
    with st.chat_message(message["role"], avatar = message.get("avatar")):
        st.markdown(f'<span class="{marker}"></span>', unsafe_allow_html=True)
        st.markdown(message["content"])
 
# ------------------------------------------------------------------
# INPUT
# ------------------------------------------------------------------
prompt = st.chat_input("Ask anything about presidium school...") or clicked
if prompt:
    with st.spinner("PresidiumSchoolBuddy is thinking... "):
        reply(prompt)
    st.rerun()
st.write("")