import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import requests
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

st.set_page_config(
    page_title="Stock Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

if "page" not in st.session_state:
    st.session_state.page = "landing"

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — LANDING
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "landing":

    st.markdown("""
    <style>
    html, body, .stApp {
        background: #00000f !important;
        margin: 0 !important; padding: 0 !important;
    }
    .block-container { padding: 0 !important; max-width: 100% !important; }
    #MainMenu, footer, header { visibility: hidden; }
    div[data-testid="stSidebar"] { display: none !important; }
    section[data-testid="stMain"] > div { padding: 0 !important; }
    .stButton { display: flex; justify-content: center; }
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: #fff !important; border: none !important;
        border-radius: 100px !important; padding: 16px 48px !important;
        font-size: 1rem !important; font-weight: 600 !important;
        box-shadow: 0 0 30px rgba(99,102,241,0.5) !important;
        letter-spacing: 0.04em !important;
        margin-top: -60px !important;
        position: relative !important;
        z-index: 9999 !important;
    }
    .stButton > button:hover {
        box-shadow: 0 0 50px rgba(99,102,241,0.7) !important;
        transform: translateY(-2px) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    components.html("""
    <!DOCTYPE html>
    <html>
    <head>
    <link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
    <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
        font-family: 'DM Sans', sans-serif;
        background: #00000f;
        color: #fff;
        overflow: hidden;
        height: 820px;
    }

    /* ── NAV ── */
    .nav-bar {
        position: fixed; top: 0; left: 0; right: 0; z-index: 999;
        display: flex; align-items: center; justify-content: space-between;
        padding: 18px 52px;
        background: rgba(0,0,15,0.65);
        backdrop-filter: blur(20px);
        border-bottom: 1px solid rgba(99,102,241,0.12);
    }
    .nav-logo {
        font-family: 'Syne', sans-serif;
        font-size: 1rem; font-weight: 800; color: #fff; letter-spacing: 0.06em;
    }
    .nav-logo span { color: #818cf8; }
    .nav-links { display: flex; align-items: center; gap: 34px; }
    .nav-links a { font-size: 0.8rem; color: rgba(255,255,255,0.45); text-decoration: none; transition: color 0.2s; }
    .nav-links a:hover { color: #fff; }
    .nav-right { display: flex; align-items: center; gap: 8px; }
    .nav-login {
        font-size: 0.78rem; color: rgba(255,255,255,0.6);
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 100px; padding: 7px 18px;
        cursor: pointer; font-family: 'DM Sans', sans-serif; text-decoration: none;
        transition: all 0.2s;
    }
    .nav-login:hover { background: rgba(255,255,255,0.1); }
    .nav-signup {
        font-size: 0.78rem; color: #fff;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        border: none; border-radius: 100px; padding: 8px 20px;
        cursor: pointer; font-family: 'DM Sans', sans-serif; font-weight: 500;
        text-decoration: none; box-shadow: 0 0 16px rgba(99,102,241,0.4);
        transition: all 0.2s;
    }
    .nav-signup:hover { box-shadow: 0 0 28px rgba(99,102,241,0.6); }

    /* ── HERO ── */
    .hero {
        height: 820px; position: relative; overflow: hidden; background: #00000f;
    }

    /* Animated blobs */
    .blob {
        position: absolute; border-radius: 50%;
        filter: blur(90px); pointer-events: none;
        animation: blobMove 10s ease-in-out infinite;
    }
    .blob1 { width:580px; height:580px; background:rgba(99,102,241,0.16); top:-150px; left:-120px; animation-delay:0s; }
    .blob2 { width:420px; height:420px; background:rgba(139,92,246,0.13); bottom:-80px; right:-60px; animation-delay:3s; }
    .blob3 { width:300px; height:300px; background:rgba(16,185,129,0.08); top:50%; left:50%; animation-delay:6s; }
    @keyframes blobMove {
        0%,100% { transform:translate(0,0) scale(1); }
        33%      { transform:translate(25px,-18px) scale(1.06); }
        66%      { transform:translate(-18px,12px) scale(0.95); }
    }

    /* Grid */
    .grid-bg {
        position:absolute; inset:0; z-index:1;
        background-image:
            linear-gradient(rgba(99,102,241,0.035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(99,102,241,0.035) 1px, transparent 1px);
        background-size:65px 65px;
    }

    /* Wave canvas — stock graph style */
    #waveCanvas {
        position:absolute; bottom:0; left:0; width:100%; z-index:2; opacity:0.45;
    }

    /* Center bottom glow */
    .center-glow {
        position:absolute; bottom:-120px; left:50%; transform:translateX(-50%);
        width:700px; height:450px; z-index:2;
        background:radial-gradient(ellipse 55% 45% at 50% 55%,
            rgba(99,102,241,0.32) 0%, rgba(139,92,246,0.16) 42%, transparent 100%);
        filter:blur(24px);
        animation:glowPulse 5s ease-in-out infinite;
    }
    @keyframes glowPulse { 0%,100%{opacity:0.65} 50%{opacity:1} }

    /* Neon rings */
    .ring {
        position:absolute; border-radius:50%; z-index:3;
        border:1px solid rgba(99,102,241,0.12);
        left:50%; transform:translateX(-50%);
        animation:ringPulse 5s ease-in-out infinite;
        pointer-events:none;
    }
    .ring1 { width:280px; height:280px; bottom:2%; animation-delay:0s; }
    .ring2 { width:480px; height:480px; bottom:-8%; animation-delay:1s; }
    .ring3 { width:680px; height:680px; bottom:-18%; animation-delay:2s; }
    @keyframes ringPulse {
        0%,100% { opacity:0.35; transform:translateX(-50%) scale(1); }
        50%      { opacity:0.7;  transform:translateX(-50%) scale(1.025); }
    }

    /* Floating glass cards */
    .fcard {
        position:absolute; z-index:6;
        background:rgba(8,9,30,0.78);
        border:1px solid rgba(99,102,241,0.22);
        border-radius:14px; padding:14px 18px;
        backdrop-filter:blur(18px);
        box-shadow:0 8px 32px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.04);
        animation:floatBob 4s ease-in-out infinite;
    }
    .fcard.a { left:4%;  top:36%; animation-delay:0s; min-width:160px; }
    .fcard.b { right:4%; top:42%; animation-delay:1.8s; min-width:150px; }
    .fcard.c { left:8%;  bottom:16%; animation-delay:0.9s; min-width:155px; }
    .fcard.d { right:7%; bottom:20%; animation-delay:2.5s; min-width:148px; }
    @keyframes floatBob {
        0%,100% { transform:translateY(0); }
        50%      { transform:translateY(-9px); }
    }
    .fc-label { font-size:0.58rem; color:rgba(255,255,255,0.28); text-transform:uppercase; letter-spacing:0.1em; margin-bottom:5px; }
    .fc-value { font-family:'Syne',sans-serif; font-size:1rem; font-weight:700; color:#fff; display:flex; align-items:center; gap:6px; }
    .fc-sub   { font-size:0.6rem; color:rgba(255,255,255,0.28); margin-top:4px; }
    .live-dot { width:7px; height:7px; border-radius:50%; background:#10b981; box-shadow:0 0 7px #10b981; flex-shrink:0; animation:blink 2s infinite; }
    .blue-dot { background:#818cf8; box-shadow:0 0 7px #818cf8; }
    .pink-dot { background:#ec4899; box-shadow:0 0 7px #ec4899; }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.35} }
    .prog-bar { width:80px; height:3px; background:rgba(255,255,255,0.1); border-radius:2px; margin-top:6px; overflow:hidden; }
    .prog-fill { height:100%; border-radius:2px; background:linear-gradient(90deg,#6366f1,#a78bfa); box-shadow:0 0 8px rgba(99,102,241,0.7); animation:progGrow 2s ease-out forwards; }
    @keyframes progGrow { from{width:0} to{width:99.9%} }

    /* Ticker tape */
    .ticker-wrap {
        position:absolute; bottom:0; left:0; right:0; z-index:8;
        background:rgba(0,0,15,0.85); backdrop-filter:blur(10px);
        border-top:1px solid rgba(99,102,241,0.1);
        padding:8px 0; overflow:hidden;
    }
    .ticker-inner {
        display:flex; gap:48px; white-space:nowrap;
        animation:tickerScroll 25s linear infinite;
    }
    .ticker-item { font-size:0.72rem; font-family:'DM Sans',sans-serif; display:flex; align-items:center; gap:8px; }
    .t-name { color:rgba(255,255,255,0.45); }
    .t-price { color:#fff; font-weight:500; }
    .t-up   { color:#10b981; font-size:0.65rem; }
    .t-down { color:#ef4444; font-size:0.65rem; }
    @keyframes tickerScroll { 0%{transform:translateX(0)} 100%{transform:translateX(-50%)} }

    /* Hero text */
    .hero-inner {
        position:absolute; inset:0; z-index:7;
        display:flex; flex-direction:column; align-items:center; justify-content:center;
        padding-top:65px; padding-bottom:60px; text-align:center;
    }
    .hero-badge {
        display:inline-flex; align-items:center; gap:8px;
        background:rgba(99,102,241,0.1); border:1px solid rgba(99,102,241,0.28);
        border-radius:100px; padding:5px 16px;
        font-size:0.68rem; color:rgba(165,180,252,0.9);
        letter-spacing:0.1em; text-transform:uppercase;
        margin-bottom:22px; backdrop-filter:blur(10px);
        animation:fadeUp 0.8s ease both;
    }
    .hero-title {
        font-family:'Syne',sans-serif;
        font-size:3.6rem; font-weight:800; line-height:1.1;
        letter-spacing:-0.02em; color:#fff; margin-bottom:16px;
        animation:fadeUp 0.8s 0.1s ease both;
    }
    .hero-title .grad {
        background:linear-gradient(135deg,#818cf8 0%,#c084fc 100%);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    }
    .hero-sub {
        font-size:0.9rem; color:rgba(255,255,255,0.38);
        line-height:1.75; max-width:380px; margin:0 auto;
        animation:fadeUp 0.8s 0.2s ease both;
    }
    @keyframes fadeUp { from{opacity:0;transform:translateY(18px)} to{opacity:1;transform:translateY(0)} }

    /* Particles */
    .particle {
        position:absolute; border-radius:50%; pointer-events:none; z-index:4;
        animation:particleFloat linear infinite;
    }
    @keyframes particleFloat {
        0%   { opacity:0; transform:translateY(0) scale(0); }
        10%  { opacity:0.9; }
        85%  { opacity:0.3; }
        100% { opacity:0; transform:translateY(-460px) scale(1.4); }
    }
    </style>
    </head>
    <body>

    <!-- NAV -->
    <div class="nav-bar">
        <div class="nav-logo">STOCK<span>IQ</span></div>
        <div class="nav-links">
            <a href="#">About</a>
            <a href="#">Trading</a>
            <a href="#">Contact</a>
            <a href="#">FAQ</a>
        </div>
        <div class="nav-right">
            <a class="nav-login" href="#">Login</a>
            <a class="nav-signup" href="#">Sign Up</a>
        </div>
    </div>

    <!-- HERO -->
    <div class="hero">
        <div class="blob blob1"></div>
        <div class="blob blob2"></div>
        <div class="blob blob3"></div>
        <div class="grid-bg"></div>
        <div class="center-glow"></div>
        <div class="ring ring1"></div>
        <div class="ring ring2"></div>
        <div class="ring ring3"></div>

        <canvas id="waveCanvas" height="320"></canvas>

        <!-- Glass cards — user friendly text -->
        <div class="fcard a">
            <div class="fc-label">Market</div>
            <div class="fc-value"><span class="live-dot"></span>Live Now</div>
            <div class="fc-sub">NYSE · NASDAQ · BSE · LSE</div>
        </div>
        <div class="fcard b">
            <div class="fc-label">Accuracy</div>
            <div class="fc-value" style="font-size:1.5rem;color:#a78bfa;">99.9%</div>
            <div class="prog-bar"><div class="prog-fill"></div></div>
        </div>
        <div class="fcard c">
            <div class="fc-label">Companies Tracked</div>
            <div class="fc-value"><span class="live-dot blue-dot"></span>100+ MNCs</div>
            <div class="fc-sub">US · India · Europe · Asia</div>
        </div>
        <div class="fcard d">
            <div class="fc-label">Data Updates</div>
            <div class="fc-value"><span class="live-dot pink-dot"></span>Real-time</div>
            <div class="fc-sub">Every 5 minutes</div>
        </div>

        <!-- Particles -->
        <div id="particles"></div>

        <!-- Hero text -->
        <div class="hero-inner">
            <div class="hero-badge">
                <span class="live-dot"></span>
                Live Stock Intelligence
            </div>
            <h1 class="hero-title">
                Elevate Your<br>
                <span class="grad">Trading Experience</span>
            </h1>
            <p class="hero-sub">
                Unlock your trading potential in a fully regulated<br>
                environment, powered by AI
            </p>
        </div>

        <!-- Ticker tape -->
        <div class="ticker-wrap">
            <div class="ticker-inner">
                <div class="ticker-item"><span class="t-name">AAPL</span><span class="t-price">$213.49</span><span class="t-up">▲ 1.24%</span></div>
                <div class="ticker-item"><span class="t-name">MSFT</span><span class="t-price">$421.30</span><span class="t-up">▲ 0.87%</span></div>
                <div class="ticker-item"><span class="t-name">GOOGL</span><span class="t-price">$178.92</span><span class="t-down">▼ 0.34%</span></div>
                <div class="ticker-item"><span class="t-name">NVDA</span><span class="t-price">$946.20</span><span class="t-up">▲ 2.15%</span></div>
                <div class="ticker-item"><span class="t-name">TSLA</span><span class="t-price">$182.45</span><span class="t-down">▼ 1.02%</span></div>
                <div class="ticker-item"><span class="t-name">AMZN</span><span class="t-price">$198.73</span><span class="t-up">▲ 0.65%</span></div>
                <div class="ticker-item"><span class="t-name">META</span><span class="t-price">$524.10</span><span class="t-up">▲ 1.44%</span></div>
                <div class="ticker-item"><span class="t-name">TCS</span><span class="t-price">₹3,842</span><span class="t-up">▲ 0.92%</span></div>
                <div class="ticker-item"><span class="t-name">INFY</span><span class="t-price">₹1,624</span><span class="t-down">▼ 0.28%</span></div>
                <div class="ticker-item"><span class="t-name">BMW</span><span class="t-price">€88.42</span><span class="t-up">▲ 0.55%</span></div>
                <!-- duplicate for seamless loop -->
                <div class="ticker-item"><span class="t-name">AAPL</span><span class="t-price">$213.49</span><span class="t-up">▲ 1.24%</span></div>
                <div class="ticker-item"><span class="t-name">MSFT</span><span class="t-price">$421.30</span><span class="t-up">▲ 0.87%</span></div>
                <div class="ticker-item"><span class="t-name">GOOGL</span><span class="t-price">$178.92</span><span class="t-down">▼ 0.34%</span></div>
                <div class="ticker-item"><span class="t-name">NVDA</span><span class="t-price">$946.20</span><span class="t-up">▲ 2.15%</span></div>
                <div class="ticker-item"><span class="t-name">TSLA</span><span class="t-price">$182.45</span><span class="t-down">▼ 1.02%</span></div>
                <div class="ticker-item"><span class="t-name">AMZN</span><span class="t-price">$198.73</span><span class="t-up">▲ 0.65%</span></div>
                <div class="ticker-item"><span class="t-name">META</span><span class="t-price">$524.10</span><span class="t-up">▲ 1.44%</span></div>
                <div class="ticker-item"><span class="t-name">TCS</span><span class="t-price">₹3,842</span><span class="t-up">▲ 0.92%</span></div>
                <div class="ticker-item"><span class="t-name">INFY</span><span class="t-price">₹1,624</span><span class="t-down">▼ 0.28%</span></div>
                <div class="ticker-item"><span class="t-name">BMW</span><span class="t-price">€88.42</span><span class="t-up">▲ 0.55%</span></div>
            </div>
        </div>
    </div>

    <script>
    // ── Stock-style animated graph ───────────────────────────────────────────
    const canvas = document.getElementById('waveCanvas');
    const ctx = canvas.getContext('2d');
    canvas.width = window.innerWidth;

    // Generate realistic stock-like data
    function genStockData(n, base, volatility) {
        let pts = [base];
        for (let i = 1; i < n; i++) {
            const change = (Math.random() - 0.48) * volatility;
            pts.push(Math.max(pts[i-1] + change, base * 0.5));
        }
        return pts;
    }

    const N = 120;
    let data1 = genStockData(N, 180, 8);
    let data2 = genStockData(N, 200, 6);
    let data3 = genStockData(N, 160, 5);
    let frame = 0;

    function drawStock() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        frame++;

        // Slowly evolve data
        if (frame % 4 === 0) {
            data1.shift();
            const last1 = data1[data1.length-1];
            data1.push(Math.max(last1 + (Math.random()-0.48)*8, 100));
            data2.shift();
            const last2 = data2[data2.length-1];
            data2.push(Math.max(last2 + (Math.random()-0.48)*6, 100));
            data3.shift();
            const last3 = data3[data3.length-1];
            data3.push(Math.max(last3 + (Math.random()-0.46)*5, 100));
        }

        const W = canvas.width;
        const H = canvas.height;

        function drawLine(data, color, glowColor, lineW, fillOpacity) {
            const minV = Math.min(...data);
            const maxV = Math.max(...data);
            const range = maxV - minV || 1;

            ctx.beginPath();
            for (let i = 0; i < data.length; i++) {
                const x = (i / (data.length-1)) * W;
                const y = H - ((data[i] - minV) / range) * (H * 0.75) - H * 0.05;
                if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
            }

            // Fill
            const fillPath = new Path2D(ctx.currentPath);
            ctx.lineTo(W, H); ctx.lineTo(0, H); ctx.closePath();
            const grad = ctx.createLinearGradient(0, 0, 0, H);
            grad.addColorStop(0, color.replace(')', `,${fillOpacity})`).replace('rgb','rgba'));
            grad.addColorStop(1, color.replace(')', ',0)').replace('rgb','rgba'));
            ctx.fillStyle = grad; ctx.fill();

            // Line
            ctx.beginPath();
            for (let i = 0; i < data.length; i++) {
                const x = (i / (data.length-1)) * W;
                const y = H - ((data[i] - minV) / range) * (H * 0.75) - H * 0.05;
                if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
            }
            ctx.strokeStyle = color;
            ctx.lineWidth = lineW;
            ctx.shadowColor = glowColor;
            ctx.shadowBlur = 14;
            ctx.stroke();
            ctx.shadowBlur = 0;
        }

        drawLine(data3, 'rgba(16,185,129,0.7)',  '#10b981', 1.5, 0.12);
        drawLine(data2, 'rgba(192,132,252,0.65)', '#c084fc', 1.5, 0.1);
        drawLine(data1, 'rgba(129,140,248,0.85)', '#818cf8', 2,   0.18);

        requestAnimationFrame(drawStock);
    }
    drawStock();

    // ── Particles ────────────────────────────────────────────────────────────
    const pc = document.getElementById('particles');
    const colors = ['rgba(99,102,241,0.85)','rgba(139,92,246,0.75)','rgba(16,185,129,0.75)','rgba(192,132,252,0.65)'];
    for (let i = 0; i < 24; i++) {
        const p = document.createElement('div');
        p.className = 'particle';
        const sz = Math.random()*2.8+1.2;
        p.style.cssText = `
            width:${sz}px; height:${sz}px;
            left:${10+Math.random()*80}%;
            bottom:${5+Math.random()*55}%;
            background:${colors[Math.floor(Math.random()*colors.length)]};
            animation-duration:${7+Math.random()*9}s;
            animation-delay:${Math.random()*7}s;
        `;
        pc.appendChild(p);
    }
    </script>
    </body>
    </html>
    """, height=820, scrolling=False)

    # GET STARTED BUTTON — clearly visible
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("Get Started →", use_container_width=True):
            st.session_state.page = "dashboard"
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "dashboard":

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');
    html, body, .stApp {
        background: linear-gradient(135deg, #020818 0%, #050f2e 50%, #020818 100%) !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    .block-container { padding-top: 1.5rem !important; padding-bottom: 1rem !important; }
    #MainMenu, footer, header { visibility: hidden; }
    div[data-testid="stSidebar"] {
        background: rgba(5,15,46,0.95) !important;
        border-right: 1px solid rgba(99,179,255,0.12) !important;
    }
    div[data-testid="stSidebar"] label { color: #7a90c0 !important; font-size: 0.82rem !important; }
    div[data-testid="stSidebar"] .stSelectbox > div > div {
        background: rgba(13,32,96,0.5) !important;
        border: 1px solid rgba(99,179,255,0.15) !important;
        color: #e2e8f8 !important; border-radius: 8px !important;
    }
    div[data-testid="stSidebar"] input {
        background: rgba(13,32,96,0.5) !important;
        border: 1px solid rgba(99,179,255,0.15) !important;
        color: #e2e8f8 !important; border-radius: 8px !important;
    }
    .sidebar-section {
        font-size: 0.62rem; color: #3a5080;
        text-transform: uppercase; letter-spacing: 0.12em;
        margin: 18px 0 8px 0; padding-bottom: 5px;
        border-bottom: 1px solid rgba(99,179,255,0.08);
    }
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
        color: #fff !important; border: none !important;
        border-radius: 8px !important; font-weight: 500 !important;
        font-size: 0.88rem !important; width: 100% !important;
    }
    .page-header {
        display: flex; align-items: center; justify-content: space-between;
        padding: 14px 0 16px 0;
        border-bottom: 1px solid rgba(99,179,255,0.1); margin-bottom: 20px;
    }
    .company-block { display: flex; align-items: center; gap: 14px; }
    .company-avatar {
        width: 46px; height: 46px; border-radius: 10px;
        background: linear-gradient(135deg, #3b82f6, #6366f1);
        display: flex; align-items: center; justify-content: center;
        font-family: 'Syne', sans-serif; font-weight: 800; font-size: 0.75rem; color: #fff;
    }
    .company-name { font-family: 'Syne', sans-serif; font-size: 1.25rem; font-weight: 800; color: #fff; }
    .company-meta { font-size: 0.72rem; color: #7a90c0; margin-top: 2px; }
    .big-price { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; color: #fff; text-align: right; }
    .price-up   { font-size: 0.85rem; color: #10b981; text-align: right; }
    .price-down { font-size: 0.85rem; color: #ef4444; text-align: right; }
    .metrics-row { display: grid; grid-template-columns: repeat(5,1fr); gap: 10px; margin-bottom: 20px; }
    .metric-card {
        background: rgba(13,32,96,0.5); border: 1px solid rgba(99,179,255,0.1);
        border-radius: 10px; padding: 14px 16px; position: relative; overflow: hidden;
    }
    .metric-card::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; }
    .mc-blue::before   { background: #3b82f6; }
    .mc-green::before  { background: #10b981; }
    .mc-red::before    { background: #ef4444; }
    .mc-purple::before { background: #a78bfa; }
    .mc-amber::before  { background: #f59e0b; }
    .mc-sky::before    { background: #38bdf8; }
    .metric-lbl { font-size: 0.62rem; color: #4a6090; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px; }
    .metric-sub { font-size: 0.58rem; color: #3a5080; font-style: italic; margin-bottom: 5px; }
    .metric-val { font-family: 'Syne', sans-serif; font-size: 1.15rem; font-weight: 700; color: #e2e8f8; }
    .metric-val.up   { color: #10b981; }
    .metric-val.down { color: #ef4444; }
    .section-lbl {
        font-size: 0.62rem; color: #3a5080; text-transform: uppercase; letter-spacing: 0.12em;
        margin: 16px 0 10px 0; padding-bottom: 6px; border-bottom: 1px solid rgba(99,179,255,0.07);
    }
    .chart-guide {
        background: rgba(13,32,96,0.4); border: 1px solid rgba(99,179,255,0.1);
        border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;
        font-size: 0.75rem; color: #7a90c0; line-height: 1.9;
    }
    .news-card {
        background: rgba(13,32,96,0.4); border: 1px solid rgba(99,179,255,0.1);
        border-left: 3px solid rgba(99,179,255,0.1);
        border-radius: 0 8px 8px 0; padding: 10px 12px; margin-bottom: 8px;
    }
    .news-card.pos { border-left-color: #10b981; }
    .news-card.neg { border-left-color: #ef4444; }
    .news-card.neu { border-left-color: #f59e0b; }
    .news-meta { font-size: 0.62rem; margin-bottom: 4px; display:flex; align-items:center; gap:6px; }
    .pill { display:inline-block; font-size:0.58rem; border-radius:10px; padding:1px 7px; text-transform:uppercase; letter-spacing:0.05em; }
    .pill-pos { background:rgba(16,185,129,0.12); color:#10b981; }
    .pill-neg { background:rgba(239,68,68,0.12);  color:#ef4444; }
    .pill-neu { background:rgba(245,158,11,0.12); color:#f59e0b; }
    .news-ttl { font-size:0.78rem; color:#a0b8e0; line-height:1.45; text-decoration:none; }
    .news-ttl:hover { color:#60a5fa; }
    .rsi-hint { font-size:0.7rem; margin-top:6px; padding:6px 10px; border-radius:6px; }
    .rsi-neutral { background:rgba(59,130,246,0.08); color:#60a5fa; }
    .rsi-over    { background:rgba(239,68,68,0.08);  color:#ef4444; }
    .rsi-under   { background:rgba(16,185,129,0.08); color:#10b981; }
    .ai-box { background:rgba(13,32,96,0.45); border:1px solid rgba(99,179,255,0.12); border-radius:12px; padding:22px 26px; }
    .ai-signal { display:inline-block; font-family:'Syne',sans-serif; font-size:1rem; font-weight:700; padding:7px 18px; border-radius:8px; margin-bottom:14px; }
    .sig-buy  { background:rgba(16,185,129,0.12); color:#10b981; border:1px solid rgba(16,185,129,0.25); }
    .sig-sell { background:rgba(239,68,68,0.12);  color:#ef4444; border:1px solid rgba(239,68,68,0.25); }
    .sig-hold { background:rgba(245,158,11,0.12); color:#f59e0b; border:1px solid rgba(245,158,11,0.25); }
    .ai-body  { font-size:0.85rem; line-height:1.75; color:#8aabcc; white-space:pre-wrap; }
    .ai-disc  { font-size:0.65rem; color:#3a5080; font-style:italic; margin-top:14px; padding-top:10px; border-top:1px solid rgba(99,179,255,0.07); }
    </style>
    """, unsafe_allow_html=True)

    COMPANY_MAP = {
        "Nvidia": "NVDA", "Apple": "AAPL", "Alphabet (Google)": "GOOGL",
        "Microsoft": "MSFT", "Amazon": "AMZN", "TSMC": "TSM",
        "Meta Platforms": "META", "Broadcom": "AVGO", "Tesla": "TSLA",
        "Samsung": "005930.KS", "Intel": "INTC", "IBM": "IBM",
        "Oracle": "ORCL", "Cisco": "CSCO", "SAP": "SAP",
        "Accenture": "ACN", "Adobe": "ADBE", "Salesforce": "CRM",
        "Qualcomm": "QCOM", "Cognizant": "CTSH", "Palantir": "PLTR",
        "Netflix": "NFLX", "Uber": "UBER", "Airbnb": "ABNB",
        "Shopify": "SHOP", "AMD": "AMD", "Toyota": "TM",
        "Volkswagen": "VWAGY", "BMW": "BMWYY", "Mercedes-Benz": "MBGYY",
        "Honda": "HMC", "Hyundai Motor": "HYMTF", "Ford": "F",
        "General Motors": "GM", "Nestle": "NSRGY", "Coca-Cola": "KO",
        "PepsiCo": "PEP", "Unilever": "UL", "McDonald's": "MCD",
        "Starbucks": "SBUX", "Shell": "SHEL", "ExxonMobil": "XOM",
        "BP": "BP", "Chevron": "CVX", "TotalEnergies": "TTE",
        "JPMorgan Chase": "JPM", "HSBC": "HSBC", "Citigroup": "C",
        "Goldman Sachs": "GS", "Bank of America": "BAC", "Visa": "V",
        "Mastercard": "MA", "Berkshire Hathaway": "BRK-B",
        "Johnson & Johnson": "JNJ", "Pfizer": "PFE", "Eli Lilly": "LLY",
        "Moderna": "MRNA", "UnitedHealth": "UNH", "Procter & Gamble": "PG",
        "Nike": "NKE", "Adidas": "ADDYY", "Walmart": "WMT",
        "Disney": "DIS", "LVMH": "LVMUY", "Boeing": "BA",
        "Siemens": "SIEGY", "ASML": "ASML", "Sony": "SONY",
        "Alibaba": "BABA", "Tencent": "TCEHY",
        "Tata Consultancy Services": "TCS.NS", "Infosys": "INFY",
        "Wipro": "WIT", "HCL Technologies": "HCLTECH.NS",
        "Tech Mahindra": "TECHM.NS", "Reliance Industries": "RELIANCE.NS",
        "Tata Motors": "TTM", "Mahindra & Mahindra": "M&M.NS",
        "Larsen & Toubro": "LT.NS", "Sun Pharmaceutical": "SUNPHARMA.NS",
        "Dr. Reddy's Laboratories": "RDY", "Bharti Airtel": "BHARTIARTL.NS",
        "Adani Enterprises": "ADANIENT.NS",
        "Hindustan Unilever": "HINDUNILVR.NS",
        "Asian Paints": "ASIANPAINT.NS",
    }

    PERIOD_MAP = {
        "1 Week":   (7,    "Past 7 days"),
        "1 Month":  (30,   "Past 30 days"),
        "3 Months": (90,   "Past 3 months"),
        "6 Months": (180,  "Past 6 months"),
        "1 Year":   (365,  "Past 12 months"),
        "2 Years":  (730,  "Past 2 years"),
        "5 Years":  (1825, "Past 5 years"),
    }

    @st.cache_data(ttl=300)
    def fetch_stock(ticker, start, end):
        return yf.Ticker(ticker).history(start=start, end=end)

    @st.cache_data(ttl=600)
    def fetch_news(company, ticker, api_key):
        q = f'"{company}" stock OR "{ticker}" earnings'
        url = (f"https://newsapi.org/v2/everything?q={requests.utils.quote(q)}"
               f"&language=en&sortBy=publishedAt&pageSize=20&apiKey={api_key}")
        try:
            arts = requests.get(url, timeout=8).json().get("articles", [])
            return [a for a in arts if company.lower() in
                    (a.get("title","") + a.get("description","")).lower()][:7]
        except:
            return []

    def calc_rsi(prices, period=14):
        d = prices.diff()
        gain = d.where(d > 0, 0).rolling(period).mean()
        loss = (-d.where(d < 0, 0)).rolling(period).mean()
        return 100 - (100 / (1 + gain / loss))

    def sentiment(text):
        pos = ['surge','gain','profit','growth','record','beat','strong','rise','rally','upgrade','revenue','exceed']
        neg = ['fall','drop','loss','decline','miss','weak','down','cut','layoff','lawsuit','crash','downgrade','warning']
        t = text.lower()
        p = sum(1 for w in pos if w in t)
        n = sum(1 for w in neg if w in t)
        return "Positive" if p > n else "Negative" if n > p else "Neutral"

    def ai_recommend(company, ticker, df, news, period_lbl, llm):
        if df.empty: return "No data available."
        latest, first = df.iloc[-1], df.iloc[0]
        chg = ((latest["Close"] - first["Close"]) / first["Close"]) * 100
        rsi_val = calc_rsi(df["Close"]).iloc[-1]
        news_txt = "\n".join([f"- {a['title']}" for a in news[:5]]) or "No recent news."
        msgs = [
            SystemMessage(content="You are a senior stock market analyst. Give a BUY, HOLD or SELL recommendation in 3-4 plain sentences. No markdown, no bullet points."),
            HumanMessage(content=f"Company: {company} ({ticker})\nPrice: ${latest['Close']:.2f} | Change ({period_lbl}): {chg:+.2f}%\nRSI: {rsi_val:.1f}\nNews:\n{news_txt}\nGive BUY/HOLD/SELL with reason.")
        ]
        try:
            return llm.invoke(msgs).content
        except Exception as e:
            return f"Analysis unavailable: {e}"

    with st.sidebar:
        if st.button("← Back to Home"):
            st.session_state.page = "landing"
            st.rerun()

        st.markdown('<div class="sidebar-section">Search Company</div>', unsafe_allow_html=True)
        query = st.text_input("", placeholder="Apple, BMW, TCS...", label_visibility="collapsed")

        if query.strip():
            matches = {k: v for k, v in COMPANY_MAP.items() if query.lower() in k.lower()}
        else:
            matches = {}

        if matches:
            company_display = st.selectbox("", list(matches.keys()), label_visibility="collapsed")
            ticker  = matches[company_display]
            company = company_display
        elif query.strip():
            st.warning("No company found.")
            st.stop()
        else:
            company = "Apple"
            ticker  = "AAPL"

        st.markdown('<div class="sidebar-section">Time Period</div>', unsafe_allow_html=True)
        period_opt = st.selectbox("", list(PERIOD_MAP.keys()), label_visibility="collapsed")
        days, period_lbl = PERIOD_MAP[period_opt]
        end_dt   = datetime.now()
        start_dt = end_dt - timedelta(days=days)

        st.markdown('<div class="sidebar-section">Chart Overlays</div>', unsafe_allow_html=True)
        show_ema = st.checkbox("EMA 20 & 50 — Trend lines", value=True)
        show_rsi = st.checkbox("RSI — Momentum", value=True)
        show_vol = st.checkbox("Volume — Shares traded", value=True)
        show_bb  = st.checkbox("Bollinger Bands — Volatility", value=False)

        st.markdown("<br>", unsafe_allow_html=True)
        run_ai = st.button("▶  Run AI Analysis")

    with st.spinner(f"Loading {company}..."):
        df   = fetch_stock(ticker, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
        news = fetch_news(company, ticker, os.getenv("NEWS_API_KEY", ""))

    if df.empty:
        st.error(f"No data found for {ticker}.")
        st.stop()

    latest = df.iloc[-1]
    prev   = df.iloc[-2] if len(df) > 1 else latest
    first  = df.iloc[0]
    day_chg      = latest["Close"] - prev["Close"]
    day_chg_pct  = (day_chg / prev["Close"]) * 100
    period_chg   = ((latest["Close"] - first["Close"]) / first["Close"]) * 100
    day_cls      = "up" if day_chg >= 0 else "down"
    period_cls   = "up" if period_chg >= 0 else "down"
    day_arrow    = "▲" if day_chg >= 0 else "▼"
    period_arrow = "▲" if period_chg >= 0 else "▼"
    abbr         = "".join([w[0] for w in company.split()[:3]]).upper()

    st.markdown(f"""
    <div class="page-header">
      <div class="company-block">
        <div class="company-avatar">{abbr}</div>
        <div>
          <div class="company-name">{company}</div>
          <div class="company-meta">{ticker} · {period_lbl}</div>
        </div>
      </div>
      <div>
        <div class="big-price">${latest['Close']:.2f}</div>
        <div class="price-{'up' if day_chg>=0 else 'down'}">{day_arrow} {abs(day_chg_pct):.2f}% today</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metrics-row">
      <div class="metric-card mc-blue">
        <div class="metric-lbl">Current Price</div>
        <div class="metric-sub">Last closing price (USD)</div>
        <div class="metric-val">${latest['Close']:.2f}</div>
      </div>
      <div class="metric-card mc-{'green' if day_chg>=0 else 'red'}">
        <div class="metric-lbl">Day Change</div>
        <div class="metric-sub">vs. previous trading day</div>
        <div class="metric-val {day_cls}">{day_arrow} {abs(day_chg_pct):.2f}%</div>
      </div>
      <div class="metric-card mc-purple">
        <div class="metric-lbl">Period Return</div>
        <div class="metric-sub">{period_lbl}</div>
        <div class="metric-val {period_cls}">{period_arrow} {abs(period_chg):.2f}%</div>
      </div>
      <div class="metric-card mc-amber">
        <div class="metric-lbl">Today's Range</div>
        <div class="metric-sub">Lowest → Highest today</div>
        <div class="metric-val" style="font-size:0.95rem;">${latest['Low']:.2f} – ${latest['High']:.2f}</div>
      </div>
      <div class="metric-card mc-sky">
        <div class="metric-lbl">Volume</div>
        <div class="metric-sub">Shares traded today</div>
        <div class="metric-val">{latest['Volume']/1e6:.1f}M</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    chart_col, news_col = st.columns([3, 1], gap="medium")

    with chart_col:
        st.markdown("""
        <div class="chart-guide">
            <strong style="color:#4a6090;font-size:0.65rem;text-transform:uppercase;letter-spacing:0.1em;">How to read this chart</strong><br>
            🟢 Green candle = price went UP that day &nbsp;|&nbsp;
            🔴 Red candle = price went DOWN &nbsp;|&nbsp;
            <span style="color:#60a5fa;">─</span> Blue = EMA 20 &nbsp;|&nbsp;
            <span style="color:#f59e0b;">─</span> Orange = EMA 50
        </div>
        """, unsafe_allow_html=True)

        rows = 1
        row_heights = [0.65]
        if show_rsi: rows += 1; row_heights.append(0.18)
        if show_vol: rows += 1; row_heights.append(0.17)

        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                            vertical_spacing=0.03, row_heights=row_heights)

        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name="Daily Price",
            increasing_line_color="#10b981", decreasing_line_color="#ef4444",
            increasing_fillcolor="#064e3b", decreasing_fillcolor="#450a0a",
        ), row=1, col=1)

        if show_ema:
            ema20 = df["Close"].ewm(span=20).mean()
            ema50 = df["Close"].ewm(span=50).mean()
            fig.add_trace(go.Scatter(x=df.index, y=ema20, name="EMA 20",
                line=dict(color="#60a5fa", width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=ema50, name="EMA 50",
                line=dict(color="#f59e0b", width=1.5)), row=1, col=1)

        if show_bb:
            sma = df["Close"].rolling(20).mean()
            std = df["Close"].rolling(20).std()
            fig.add_trace(go.Scatter(x=df.index, y=sma+std*2, name="BB Upper",
                line=dict(color="#a78bfa", width=1, dash="dash")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=sma-std*2, name="BB Lower",
                line=dict(color="#a78bfa", width=1, dash="dash"),
                fill="tonexty", fillcolor="rgba(167,139,250,0.04)"), row=1, col=1)

        fig.update_yaxes(title_text="Price (USD $)", tickprefix="$",
            title_font=dict(size=10, color="#3a5080"),
            tickfont=dict(size=9, color="#3a5080"),
            gridcolor="rgba(99,179,255,0.05)", row=1, col=1)

        cur = 2
        if show_rsi:
            rsi = calc_rsi(df["Close"])
            fig.add_trace(go.Scatter(x=df.index, y=rsi, name="RSI",
                line=dict(color="#f472b6", width=1.5)), row=cur, col=1)
            fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,68,68,0.04)", line_width=0, row=cur, col=1)
            fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(16,185,129,0.04)", line_width=0, row=cur, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", opacity=0.35,
                annotation_text="Overbought 70", annotation_position="right",
                annotation_font=dict(size=8, color="#ef4444"), row=cur, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="#10b981", opacity=0.35,
                annotation_text="Oversold 30", annotation_position="right",
                annotation_font=dict(size=8, color="#10b981"), row=cur, col=1)
            fig.update_yaxes(title_text="RSI", range=[0,100],
                title_font=dict(size=9, color="#3a5080"),
                tickfont=dict(size=9, color="#3a5080"),
                gridcolor="rgba(99,179,255,0.05)", row=cur, col=1)
            cur += 1

        if show_vol:
            vcols = ["#10b981" if df["Close"].iloc[i] >= df["Open"].iloc[i]
                     else "#ef4444" for i in range(len(df))]
            fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                marker_color=vcols, opacity=0.65), row=cur, col=1)
            fig.update_yaxes(title_text="Volume",
                title_font=dict(size=9, color="#3a5080"),
                tickfont=dict(size=9, color="#3a5080"),
                gridcolor="rgba(99,179,255,0.05)", row=cur, col=1)

        fig.update_xaxes(title_text="Date",
            title_font=dict(size=9, color="#3a5080"),
            tickfont=dict(size=9, color="#3a5080"),
            gridcolor="rgba(99,179,255,0.05)",
            row=cur if (show_rsi or show_vol) else 1, col=1)

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(5,15,46,0.6)",
            font=dict(family="DM Sans"), xaxis_rangeslider_visible=False,
            height=520, margin=dict(l=8,r=8,t=8,b=8),
            legend=dict(orientation="h", y=1.02, x=0,
                font=dict(size=9, color="#7a90c0"), bgcolor="rgba(0,0,0,0)"),
            hovermode="x unified",
            hoverlabel=dict(bgcolor="#0d2060", font_color="#e2e8f8", font_size=11)
        )
        st.plotly_chart(fig, use_container_width=True)

        if show_rsi:
            rsi_now = calc_rsi(df["Close"]).iloc[-1]
            if rsi_now > 70:
                st.markdown(f'<div class="rsi-hint rsi-over">RSI {rsi_now:.0f} — above 70, stock may be overbought</div>', unsafe_allow_html=True)
            elif rsi_now < 30:
                st.markdown(f'<div class="rsi-hint rsi-under">RSI {rsi_now:.0f} — below 30, stock may be oversold</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="rsi-hint rsi-neutral">RSI {rsi_now:.0f} — neutral zone, no strong momentum signal</div>', unsafe_allow_html=True)

    with news_col:
        st.markdown(f'<div class="section-lbl">{company} News</div>', unsafe_allow_html=True)
        if not news:
            st.markdown('<div style="color:#3a5080;font-size:0.8rem;font-style:italic;">No recent news found.</div>', unsafe_allow_html=True)
        else:
            for a in news[:6]:
                title = a.get("title","").split(" - ")[0]
                url   = a.get("url","#")
                date  = a.get("publishedAt","")[:10]
                s     = sentiment(title + " " + a.get("description",""))
                sc    = "pos" if s=="Positive" else "neg" if s=="Negative" else "neu"
                pc    = "pill-pos" if s=="Positive" else "pill-neg" if s=="Negative" else "pill-neu"
                mc    = "#10b981" if s=="Positive" else "#ef4444" if s=="Negative" else "#f59e0b"
                st.markdown(f"""
                <div class="news-card {sc}">
                  <div class="news-meta" style="color:{mc};">
                    {date} &nbsp;<span class="pill {pc}">{s}</span>
                  </div>
                  <a class="news-ttl" href="{url}" target="_blank">{title[:105]}{'...' if len(title)>105 else ''}</a>
                </div>""", unsafe_allow_html=True)

    if run_ai:
        st.markdown('<div class="section-lbl">AI Analysis</div>', unsafe_allow_html=True)
        groq_key = os.getenv("GROQ_API_KEY","")
        if not groq_key:
            st.warning("Add GROQ_API_KEY to your .env file.")
        else:
            with st.spinner(f"Analysing {company}..."):
                llm = ChatGroq(groq_api_key=groq_key, model_name="llama-3.1-8b-instant", temperature=0)
                result = ai_recommend(company, ticker, df, news, period_lbl, llm)

            if "BUY"  in result.upper(): sig, sc = "▲  BUY",  "sig-buy"
            elif "SELL" in result.upper(): sig, sc = "▼  SELL", "sig-sell"
            else:                          sig, sc = "◆  HOLD", "sig-hold"

            st.markdown(f"""
            <div class="ai-box">
              <span class="ai-signal {sc}">{sig}</span>
              <div class="ai-body">{result}</div>
              <div class="ai-disc">Automated AI summary. Not financial advice.</div>
            </div>""", unsafe_allow_html=True)