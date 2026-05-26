import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import requests
import os
import numpy as np
import pandas as pd
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
if "selected_company" not in st.session_state:
    st.session_state.selected_company = None
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = None

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
    "3M": "MMM", "Abbott Laboratories": "ABT",
    "AstraZeneca": "AZN", "Bayer": "BAYRY",
    "Deutsche Telekom": "DTEGY", "Caterpillar": "CAT",
    "Deere & Company": "DE",
}

PERIOD_OPTIONS = [
    "🔴 Live",
    "Past 1 Week", "Past 2 Weeks", "Past 1 Month",
    "Past 3 Months", "Past 6 Months", "Past 1 Year",
    "Past 2 Years", "Past 3 Years", "Past 5 Years", "Past 10 Years",
    "Next 1 Week (AI Prediction)",
    "Next 2 Weeks (AI Prediction)",
]

PERIOD_MAP = {
    "🔴 Live":                      (1,    "Live",          "live",   0),
    "Past 1 Week":                  (7,    "Past 7 days",   "past",   0),
    "Past 2 Weeks":                 (14,   "Past 14 days",  "past",   0),
    "Past 1 Month":                 (30,   "Past 30 days",  "past",   0),
    "Past 3 Months":                (90,   "Past 3 months", "past",   0),
    "Past 6 Months":                (180,  "Past 6 months", "past",   0),
    "Past 1 Year":                  (365,  "Past 1 year",   "past",   0),
    "Past 2 Years":                 (730,  "Past 2 years",  "past",   0),
    "Past 3 Years":                 (1095, "Past 3 years",  "past",   0),
    "Past 5 Years":                 (1825, "Past 5 years",  "past",   0),
    "Past 10 Years":                (3650, "Past 10 years", "past",   0),
    "Next 1 Week (AI Prediction)":  (180,  "Next 7 days",   "future", 7),
    "Next 2 Weeks (AI Prediction)": (180,  "Next 14 days",  "future", 14),
}

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — LANDING (hero with 3D graphs, floating cards, ticker)
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
        border-radius: 100px !important; padding: 16px 52px !important;
        font-size: 1rem !important; font-weight: 600 !important;
        box-shadow: 0 0 30px rgba(99,102,241,0.5) !important;
        letter-spacing: 0.04em !important;
        margin-top: 10px !important;
    }
    .stButton > button:hover {
        box-shadow: 0 0 50px rgba(99,102,241,0.75) !important;
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
    * { margin:0; padding:0; box-sizing:border-box; }
    body { font-family:'DM Sans',sans-serif; background:#00000f; color:#fff; overflow:hidden; height:750px; }
    .nav-bar { position:fixed; top:0; left:0; right:0; z-index:999; display:flex; align-items:center; justify-content:space-between; padding:18px 52px; background:rgba(0,0,15,0.65); backdrop-filter:blur(20px); border-bottom:1px solid rgba(99,102,241,0.12); }
    .nav-logo { font-family:'Syne',sans-serif; font-size:1.05rem; font-weight:800; color:#fff; letter-spacing:0.06em; }
    .nav-logo span { color:#818cf8; }
    .nav-links { display:flex; align-items:center; gap:36px; }
    .nav-links a { font-size:0.82rem; color:rgba(255,255,255,0.45); text-decoration:none; }
    .nav-right { display:flex; align-items:center; gap:8px; }
    .nav-login { font-size:0.8rem; color:rgba(255,255,255,0.6); background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.12); border-radius:100px; padding:7px 20px; text-decoration:none; }
    .nav-signup { font-size:0.8rem; color:#fff; background:linear-gradient(135deg,#6366f1,#8b5cf6); border:none; border-radius:100px; padding:8px 22px; text-decoration:none; box-shadow:0 0 16px rgba(99,102,241,0.4); }
    .hero { height:750px; position:relative; overflow:hidden; background:#00000f; }
    .blob { position:absolute; border-radius:50%; filter:blur(90px); pointer-events:none; animation:blobMove 10s ease-in-out infinite; }
    .blob1 { width:580px; height:580px; background:rgba(99,102,241,0.15); top:-150px; left:-120px; animation-delay:0s; }
    .blob2 { width:420px; height:420px; background:rgba(139,92,246,0.12); bottom:-80px; right:-60px; animation-delay:3s; }
    .blob3 { width:300px; height:300px; background:rgba(16,185,129,0.07); top:40%; left:45%; animation-delay:6s; }
    @keyframes blobMove { 0%,100%{transform:translate(0,0) scale(1)} 33%{transform:translate(22px,-16px) scale(1.05)} 66%{transform:translate(-16px,10px) scale(0.95)} }
    .grid-bg { position:absolute; inset:0; z-index:1; background-image:linear-gradient(rgba(99,102,241,0.03) 1px,transparent 1px),linear-gradient(90deg,rgba(99,102,241,0.03) 1px,transparent 1px); background-size:65px 65px; }
    #waveCanvas { position:absolute; bottom:0; left:0; width:100%; z-index:2; opacity:0.4; }
    .center-glow { position:absolute; bottom:-130px; left:50%; transform:translateX(-50%); width:750px; height:480px; z-index:2; background:radial-gradient(ellipse 55% 45% at 50% 55%,rgba(99,102,241,0.28) 0%,rgba(139,92,246,0.14) 44%,transparent 100%); filter:blur(26px); animation:glowPulse 5s ease-in-out infinite; }
    @keyframes glowPulse { 0%,100%{opacity:0.6} 50%{opacity:1} }
    .ring { position:absolute; border-radius:50%; z-index:1; border:1px solid rgba(99,102,241,0.1); left:50%; transform:translateX(-50%); animation:ringPulse 5s ease-in-out infinite; pointer-events:none; }
    .ring1 { width:280px; height:280px; bottom:2%; animation-delay:0s; }
    .ring2 { width:480px; height:480px; bottom:-8%; animation-delay:1s; }
    .ring3 { width:680px; height:680px; bottom:-18%; animation-delay:2s; }
    @keyframes ringPulse { 0%,100%{opacity:0.3;transform:translateX(-50%) scale(1)} 50%{opacity:0.65;transform:translateX(-50%) scale(1.025)} }
    .fcard { position:absolute; z-index:9; background:rgba(8,9,30,0.82); border:1px solid rgba(99,102,241,0.22); border-radius:14px; padding:14px 20px; backdrop-filter:blur(18px); box-shadow:0 8px 32px rgba(0,0,0,0.5); animation:floatBob 4s ease-in-out infinite; }
    .fcard.a { left:3%; bottom:28%; animation-delay:0s; min-width:165px; }
    .fcard.b { right:3%; bottom:34%; animation-delay:1.8s; min-width:155px; }
    .fcard.c { left:3%; bottom:10%; animation-delay:0.9s; min-width:160px; }
    .fcard.d { right:3%; bottom:12%; animation-delay:2.5s; min-width:152px; }
    @keyframes floatBob { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-9px)} }
    .fc-label { font-size:0.58rem; color:rgba(255,255,255,0.28); text-transform:uppercase; letter-spacing:0.1em; margin-bottom:5px; }
    .fc-value { font-family:'Syne',sans-serif; font-size:1rem; font-weight:700; color:#fff; display:flex; align-items:center; gap:6px; }
    .fc-sub { font-size:0.6rem; color:rgba(255,255,255,0.28); margin-top:4px; }
    .live-dot { width:7px; height:7px; border-radius:50%; background:#10b981; box-shadow:0 0 7px #10b981; flex-shrink:0; animation:blink 2s infinite; }
    .blue-dot { background:#818cf8; box-shadow:0 0 7px #818cf8; }
    .pink-dot { background:#ec4899; box-shadow:0 0 7px #ec4899; }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.35} }
    .prog-bar { width:80px; height:3px; background:rgba(255,255,255,0.1); border-radius:2px; margin-top:6px; overflow:hidden; }
    .prog-fill { height:100%; border-radius:2px; background:linear-gradient(90deg,#6366f1,#a78bfa); animation:progGrow 2s ease-out forwards; }
    @keyframes progGrow { from{width:0} to{width:99.9%} }
    .ticker-wrap { position:absolute; bottom:0; left:0; right:0; z-index:10; background:rgba(0,0,15,0.88); border-top:1px solid rgba(99,102,241,0.1); padding:9px 0; overflow:hidden; }
    .ticker-inner { display:flex; gap:52px; white-space:nowrap; animation:tickerScroll 28s linear infinite; }
    .ticker-item { font-size:0.72rem; display:flex; align-items:center; gap:8px; }
    .t-name{color:rgba(255,255,255,0.4);} .t-price{color:#fff;font-weight:500;}
    .t-up{color:#10b981;font-size:0.65rem;} .t-down{color:#ef4444;font-size:0.65rem;}
    @keyframes tickerScroll { 0%{transform:translateX(0)} 100%{transform:translateX(-50%)} }
    .hero-inner { position:absolute; inset:0; z-index:10; display:flex; flex-direction:column; align-items:center; justify-content:center; padding-top:65px; padding-bottom:60px; text-align:center; pointer-events:none; }
    .hero-badge { display:inline-flex; align-items:center; gap:8px; background:rgba(99,102,241,0.1); border:1px solid rgba(99,102,241,0.28); border-radius:100px; padding:6px 18px; font-size:0.68rem; color:rgba(165,180,252,0.9); letter-spacing:0.1em; text-transform:uppercase; margin-bottom:22px; }
    .hero-title { font-family:'Syne',sans-serif; font-size:4.8rem; font-weight:800; line-height:1.0; letter-spacing:-0.04em; color:#fff; margin-bottom:16px; }
    .hero-title .grad { background:linear-gradient(135deg,#a5b4fc 0%,#e879f9 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
    .hero-sub { font-size:1rem; color:rgba(255,255,255,0.45); line-height:1.75; max-width:460px; margin:0 auto; font-weight:300; }
    .particle { position:absolute; border-radius:50%; pointer-events:none; z-index:4; animation:particleFloat linear infinite; }
    @keyframes particleFloat { 0%{opacity:0;transform:translateY(0) scale(0)} 10%{opacity:0.9} 85%{opacity:0.3} 100%{opacity:0;transform:translateY(-460px) scale(1.4)} }
    </style>
    </head>
    <body>
    <div class="nav-bar">
        <div class="nav-logo">STOCK<span>IQ</span></div>
        <div class="nav-links"><a href="#">About</a><a href="#">Trading</a><a href="#">Contact</a><a href="#">FAQ</a></div>
        <div class="nav-right"><a class="nav-login" href="#">Login</a><a class="nav-signup" href="#">Sign Up</a></div>
    </div>
    <div class="hero">
        <div class="blob blob1"></div><div class="blob blob2"></div><div class="blob blob3"></div>
        <div class="grid-bg"></div><div class="center-glow"></div>
        <div class="ring ring1"></div><div class="ring ring2"></div><div class="ring ring3"></div>
        <canvas id="waveCanvas" height="280"></canvas>
        <div class="fcard a"><div class="fc-label">Market</div><div class="fc-value"><span class="live-dot"></span>Live Now</div><div class="fc-sub">NYSE · NASDAQ · BSE · LSE</div></div>
        <div class="fcard b"><div class="fc-label">Accuracy</div><div class="fc-value" style="font-size:1.5rem;color:#a78bfa;">99.9%</div><div class="prog-bar"><div class="prog-fill"></div></div></div>
        <div class="fcard c"><div class="fc-label">Companies Tracked</div><div class="fc-value"><span class="live-dot blue-dot"></span>100+ MNCs</div><div class="fc-sub">US · India · Europe · Asia</div></div>
        <div class="fcard d"><div class="fc-label">Data Updates</div><div class="fc-value"><span class="live-dot pink-dot"></span>Real-time</div><div class="fc-sub">Every 60 seconds</div></div>
        <div id="particles"></div>
        <div class="hero-inner">
            <div class="hero-badge"><span class="live-dot"></span>Live Stock Intelligence</div>
            <h1 class="hero-title">Elevate Your<br><span class="grad">Trading Experience</span></h1>
            <p class="hero-sub">Unlock your trading potential in a fully regulated<br>environment, powered by AI</p>
        </div>
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
                <div class="ticker-item"><span class="t-name">RELIANCE</span><span class="t-price">₹2,934</span><span class="t-up">▲ 0.78%</span></div>
                <div class="ticker-item"><span class="t-name">SONY</span><span class="t-price">$84.20</span><span class="t-down">▼ 0.42%</span></div>
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
                <div class="ticker-item"><span class="t-name">RELIANCE</span><span class="t-price">₹2,934</span><span class="t-up">▲ 0.78%</span></div>
                <div class="ticker-item"><span class="t-name">SONY</span><span class="t-price">$84.20</span><span class="t-down">▼ 0.42%</span></div>
            </div>
        </div>
    </div>
    <script>
    const canvas = document.getElementById('waveCanvas');
    const ctx = canvas.getContext('2d');
    canvas.width = window.innerWidth;
    function genStockData(n,base,vol){let pts=[base];for(let i=1;i<n;i++){const chg=(Math.random()-0.48)*vol;pts.push(Math.max(pts[i-1]+chg,base*0.5));}return pts;}
    const N=120;
    let d1=genStockData(N,180,8),d2=genStockData(N,200,6),d3=genStockData(N,160,5);
    let frame=0;
    function drawStock(){
        ctx.clearRect(0,0,canvas.width,canvas.height);frame++;
        if(frame%4===0){
            d1.shift();d1.push(Math.max(d1[d1.length-1]+(Math.random()-0.48)*8,100));
            d2.shift();d2.push(Math.max(d2[d2.length-1]+(Math.random()-0.48)*6,100));
            d3.shift();d3.push(Math.max(d3[d3.length-1]+(Math.random()-0.46)*5,100));
        }
        const W=canvas.width,H=canvas.height;
        function drawLine(data,color,glow,lw,fillOp){
            const mn=Math.min(...data),mx=Math.max(...data),rng=mx-mn||1;
            ctx.beginPath();
            for(let i=0;i<data.length;i++){const x=(i/(data.length-1))*W;const y=H-((data[i]-mn)/rng)*(H*0.75)-H*0.05;if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);}
            ctx.lineTo(W,H);ctx.lineTo(0,H);ctx.closePath();
            const gr=ctx.createLinearGradient(0,0,0,H);
            gr.addColorStop(0,color.replace(/[\d.]+\)$/,`${fillOp})`));
            gr.addColorStop(1,color.replace(/[\d.]+\)$/,'0)'));
            ctx.fillStyle=gr;ctx.fill();
            ctx.beginPath();
            for(let i=0;i<data.length;i++){const x=(i/(data.length-1))*W;const y=H-((data[i]-mn)/rng)*(H*0.75)-H*0.05;if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);}
            ctx.strokeStyle=color;ctx.lineWidth=lw;ctx.shadowColor=glow;ctx.shadowBlur=14;ctx.stroke();ctx.shadowBlur=0;
        }
        drawLine(d3,'rgba(16,185,129,0.7)','#10b981',1.5,0.12);
        drawLine(d2,'rgba(192,132,252,0.65)','#c084fc',1.5,0.1);
        drawLine(d1,'rgba(129,140,248,0.85)','#818cf8',2.2,0.18);
        requestAnimationFrame(drawStock);
    }
    drawStock();
    const pc=document.getElementById('particles');
    const cols=['rgba(99,102,241,0.85)','rgba(139,92,246,0.75)','rgba(16,185,129,0.75)','rgba(192,132,252,0.65)'];
    for(let i=0;i<24;i++){
        const p=document.createElement('div');p.className='particle';
        const sz=Math.random()*2.8+1.2;
        p.style.cssText=`width:${sz}px;height:${sz}px;left:${10+Math.random()*80}%;bottom:${5+Math.random()*55}%;background:${cols[Math.floor(Math.random()*cols.length)]};animation-duration:${7+Math.random()*9}s;animation-delay:${Math.random()*7}s;`;
        pc.appendChild(p);
    }
    </script>
    </body>
    </html>
    """, height=750, scrolling=False)

    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("Get Started →", use_container_width=True, key="get_started"):
            st.session_state.page = "dashboard"
            st.session_state.selected_company = None
            st.session_state.selected_ticker = None
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DASHBOARD (big center search, no sidebar search)
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "dashboard":

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');
    html, body, .stApp {
        background: linear-gradient(135deg, #020818 0%, #050f2e 50%, #020818 100%) !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    .block-container { padding-top: 1rem !important; padding-bottom: 1rem !important; }
    #MainMenu, footer, header { visibility: hidden; }
    div[data-testid="stSidebar"] { display: none !important; }

    /* Big search bar */
    .search-header {
        text-align: center; padding: 24px 20px 16px;
        border-bottom: 1px solid rgba(99,179,255,0.08);
        margin-bottom: 20px;
    }
    .search-logo {
        font-family: 'Syne', sans-serif;
        font-size: 1.1rem; font-weight: 800;
        color: #fff; letter-spacing: 0.06em;
        margin-bottom: 16px;
    }
    .search-logo span { color: #818cf8; }

    /* Override Streamlit input to look big and centered */
    div[data-testid="stTextInput"] input {
        background: rgba(13,32,96,0.6) !important;
        border: 1px solid rgba(99,179,255,0.25) !important;
        border-radius: 50px !important;
        color: #e2e8f8 !important;
        font-size: 1rem !important;
        padding: 14px 24px !important;
        text-align: center !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: rgba(99,102,241,0.6) !important;
        box-shadow: 0 0 20px rgba(99,102,241,0.2) !important;
    }
    div[data-testid="stTextInput"] input::placeholder {
        color: rgba(255,255,255,0.3) !important;
        text-align: center !important;
    }

    /* Results selectbox */
    div[data-testid="stSelectbox"] > div > div {
        background: rgba(13,32,96,0.6) !important;
        border: 1px solid rgba(99,179,255,0.2) !important;
        color: #e2e8f8 !important;
        border-radius: 12px !important;
    }

    /* Back button and period selectbox in top bar */
    .top-bar {
        display: flex; align-items: center;
        gap: 16px; margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 1px solid rgba(99,179,255,0.08);
    }
    .stButton > button {
        background: rgba(99,102,241,0.15) !important;
        color: #818cf8 !important; border: 1px solid rgba(99,102,241,0.3) !important;
        border-radius: 8px !important; font-weight: 500 !important;
        font-size: 0.82rem !important;
        padding: 6px 16px !important;
        width: auto !important;
    }
    .stButton > button:hover {
        background: rgba(99,102,241,0.25) !important;
    }

    .page-header { display: flex; align-items: center; justify-content: space-between; padding: 14px 0 16px 0; border-bottom: 1px solid rgba(99,179,255,0.1); margin-bottom: 20px; }
    .company-block { display: flex; align-items: center; gap: 14px; }
    .company-avatar { width: 46px; height: 46px; border-radius: 10px; background: linear-gradient(135deg, #3b82f6, #6366f1); display: flex; align-items: center; justify-content: center; font-family: 'Syne', sans-serif; font-weight: 800; font-size: 0.75rem; color: #fff; }
    .company-name { font-family: 'Syne', sans-serif; font-size: 1.25rem; font-weight: 800; color: #fff; }
    .company-meta { font-size: 0.72rem; color: #7a90c0; margin-top: 2px; }
    .live-badge { display: inline-flex; align-items: center; gap: 5px; background: rgba(16,185,129,0.12); border: 1px solid rgba(16,185,129,0.25); border-radius: 100px; padding: 3px 10px; font-size: 0.65rem; color: #10b981; margin-left: 8px; vertical-align: middle; }
    .live-dot-sm { width:5px; height:5px; border-radius:50%; background:#10b981; box-shadow:0 0 4px #10b981; animation:blink 1.5s infinite; display:inline-block; }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }
    .big-price { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; color: #fff; text-align: right; }
    .price-up { font-size: 0.85rem; color: #10b981; text-align: right; }
    .price-down { font-size: 0.85rem; color: #ef4444; text-align: right; }
    .metrics-row { display: grid; grid-template-columns: repeat(5,1fr); gap: 10px; margin-bottom: 20px; }
    .metric-card { background: rgba(13,32,96,0.5); border: 1px solid rgba(99,179,255,0.1); border-radius: 10px; padding: 14px 16px; position: relative; overflow: hidden; }
    .metric-card::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; }
    .mc-blue::before { background: #3b82f6; } .mc-green::before { background: #10b981; }
    .mc-red::before { background: #ef4444; } .mc-purple::before { background: #a78bfa; }
    .mc-amber::before { background: #f59e0b; } .mc-sky::before { background: #38bdf8; }
    .metric-lbl { font-size: 0.62rem; color: #4a6090; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px; }
    .metric-sub { font-size: 0.58rem; color: #3a5080; font-style: italic; margin-bottom: 5px; }
    .metric-val { font-family: 'Syne', sans-serif; font-size: 1.15rem; font-weight: 700; color: #e2e8f8; }
    .metric-val.up { color: #10b981; } .metric-val.down { color: #ef4444; }
    .chart-guide { background: rgba(13,32,96,0.4); border: 1px solid rgba(99,179,255,0.1); border-radius: 8px; padding: 10px 14px; margin-bottom: 8px; font-size: 0.75rem; color: #7a90c0; line-height: 1.9; }
    .pred-note { background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.2); border-radius: 8px; padding: 8px 14px; margin-bottom: 8px; font-size: 0.72rem; color: #f59e0b; }
    .refresh-bar { background: rgba(16,185,129,0.06); border: 1px solid rgba(16,185,129,0.15); border-radius: 8px; padding: 8px 14px; margin-bottom: 12px; font-size: 0.7rem; color: #10b981; }
    .news-card { background: rgba(13,32,96,0.4); border: 1px solid rgba(99,179,255,0.1); border-left: 3px solid rgba(99,179,255,0.1); border-radius: 0 8px 8px 0; padding: 10px 12px; margin-bottom: 8px; }
    .news-card.pos { border-left-color: #10b981; } .news-card.neg { border-left-color: #ef4444; } .news-card.neu { border-left-color: #f59e0b; }
    .news-meta { font-size: 0.62rem; margin-bottom: 4px; display:flex; align-items:center; gap:6px; }
    .pill { display:inline-block; font-size:0.58rem; border-radius:10px; padding:1px 7px; text-transform:uppercase; }
    .pill-pos { background:rgba(16,185,129,0.12); color:#10b981; }
    .pill-neg { background:rgba(239,68,68,0.12); color:#ef4444; }
    .pill-neu { background:rgba(245,158,11,0.12); color:#f59e0b; }
    .news-ttl { font-size:0.78rem; color:#a0b8e0; line-height:1.45; text-decoration:none; }
    .news-ttl:hover { color:#60a5fa; }
    .rsi-hint { font-size:0.7rem; margin-top:6px; padding:6px 10px; border-radius:6px; }
    .rsi-neutral { background:rgba(59,130,246,0.08); color:#60a5fa; }
    .rsi-over { background:rgba(239,68,68,0.08); color:#ef4444; }
    .rsi-under { background:rgba(16,185,129,0.08); color:#10b981; }
    .ai-box { background:rgba(13,32,96,0.45); border:1px solid rgba(99,179,255,0.12); border-radius:12px; padding:22px 26px; margin-top: 16px; }
    .ai-signal { display:inline-block; font-family:'Syne',sans-serif; font-size:1rem; font-weight:700; padding:7px 18px; border-radius:8px; margin-bottom:14px; }
    .sig-buy { background:rgba(16,185,129,0.12); color:#10b981; border:1px solid rgba(16,185,129,0.25); }
    .sig-sell { background:rgba(239,68,68,0.12); color:#ef4444; border:1px solid rgba(239,68,68,0.25); }
    .sig-hold { background:rgba(245,158,11,0.12); color:#f59e0b; border:1px solid rgba(245,158,11,0.25); }
    .ai-body { font-size:0.85rem; line-height:1.75; color:#8aabcc; white-space:pre-wrap; }
    .ai-disc { font-size:0.65rem; color:#3a5080; font-style:italic; margin-top:14px; padding-top:10px; border-top:1px solid rgba(99,179,255,0.07); }
    .hint-text { font-size:0.78rem; color:#3a5080; text-align:center; margin-top:8px; }
    .search-empty { text-align:center; padding: 60px 20px; }
    .search-empty-icon { font-size:3rem; margin-bottom:16px; }
    .search-empty-title { font-family:'Syne',sans-serif; font-size:1.3rem; font-weight:700; color:#e2e8f8; margin-bottom:10px; }
    .search-empty-sub { font-size:0.85rem; color:#3a5080; line-height:1.8; }
    </style>
    """, unsafe_allow_html=True)

    # ── HELPER FUNCTIONS ──────────────────────────────────────────────────────
    @st.cache_data(ttl=60)
    def fetch_live(ticker):
        return yf.Ticker(ticker).history(period="2d", interval="5m")

    @st.cache_data(ttl=300)
    def fetch_stock(ticker, start, end):
        return yf.Ticker(ticker).history(start=start, end=end)

    @st.cache_data(ttl=600)
    def fetch_news(company, ticker, api_key):
        q = f'"{company}" stock OR "{ticker}" earnings'
        url = (f"https://newsapi.org/v2/everything?q={requests.utils.quote(q)}"
               f"&language=en&sortBy=publishedAt&pageSize=20&apiKey={api_key}")
        try:
            arts = requests.get(url, timeout=5).json().get("articles", [])
            return [a for a in arts if company.lower() in
                    (a.get("title","") + a.get("description","")).lower()][:7]
        except:
            return []

    def calc_rsi(prices, period=14):
        d = prices.diff()
        gain = d.where(d > 0, 0).rolling(period).mean()
        loss = (-d.where(d < 0, 0)).rolling(period).mean()
        return 100 - (100 / (1 + gain / loss))

    def predict_prices(df, days_ahead):
        closes = df['Close'].values
        x = np.arange(len(closes))
        coeffs = np.polyfit(x, closes, 1)
        slope, intercept = coeffs
        predicted_hist = slope * x + intercept
        residuals = closes - predicted_hist
        std_dev = np.std(residuals)
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_ahead, freq='B')
        future_x = np.arange(len(closes), len(closes) + days_ahead)
        future_pred = slope * future_x + intercept
        np.random.seed(42)
        noise = np.cumsum(np.random.normal(0, std_dev * 0.1, days_ahead))
        future_pred = future_pred + noise
        upper = future_pred + 1.5 * std_dev
        lower = future_pred - 1.5 * std_dev
        return future_dates, future_pred, upper, lower

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

    # ── TOP BAR ───────────────────────────────────────────────────────────────
    col_back, col_logo, col_period, col_overlays, col_ai = st.columns([1, 2, 2, 3, 1.5])

    with col_back:
        if st.button("← Home", key="back_home"):
            st.session_state.page = "landing"
            st.session_state.selected_company = None
            st.session_state.selected_ticker = None
            st.rerun()

    with col_logo:
        st.markdown('<div style="font-family:Syne,sans-serif;font-size:1rem;font-weight:800;color:#fff;padding-top:6px;">STOCK<span style="color:#818cf8;">IQ</span></div>', unsafe_allow_html=True)

    with col_period:
        period_opt = st.selectbox("", PERIOD_OPTIONS, label_visibility="collapsed", key="period_select")

    with col_overlays:
        ov1, ov2, ov3, ov4 = st.columns(4)
        with ov1: show_ema = st.checkbox("EMA", value=True, key="ema")
        with ov2: show_rsi = st.checkbox("RSI", value=True, key="rsi")
        with ov3: show_vol = st.checkbox("Vol", value=True, key="vol")
        with ov4: show_bb  = st.checkbox("BB",  value=False, key="bb")

    with col_ai:
        run_ai = st.button("▶ AI Analysis", key="run_ai")

    st.markdown('<hr style="border:none;border-top:1px solid rgba(99,179,255,0.08);margin:0 0 16px;">', unsafe_allow_html=True)

    period_info = PERIOD_MAP.get(period_opt, PERIOD_MAP["Past 1 Week"])
    days, period_lbl, mode, pred_days = period_info
    end_dt   = datetime.now()
    start_dt = end_dt - timedelta(days=days)

    # ── BIG CENTER SEARCH ─────────────────────────────────────────────────────
    sc1, sc2, sc3 = st.columns([1, 2, 1])
    with sc2:
        query = st.text_input(
            "",
            placeholder="🔍  Search any company — Apple, BMW, TCS, Tesla...",
            label_visibility="collapsed",
            key="main_search"
        )
        st.markdown('<div class="hint-text">Type a company name to search from 100+ MNCs worldwide</div>', unsafe_allow_html=True)

    # ── COMPANY SELECTION ─────────────────────────────────────────────────────
    company = None
    ticker  = None

    if query.strip():
        matches = {k: v for k, v in COMPANY_MAP.items() if query.lower() in k.lower()}
        if matches:
            sc1b, sc2b, sc3b = st.columns([1, 2, 1])
            with sc2b:
                company_display = st.selectbox(
                    "",
                    list(matches.keys()),
                    label_visibility="collapsed",
                    key="company_select"
                )
            ticker  = matches[company_display]
            company = company_display
            st.session_state.selected_company = company
            st.session_state.selected_ticker  = ticker
        else:
            sc1c, sc2c, sc3c = st.columns([1, 2, 1])
            with sc2c:
                st.warning("No company found. Try: Apple, Tesla, BMW, TCS, Infosys...")
    elif st.session_state.selected_company:
        company = st.session_state.selected_company
        ticker  = st.session_state.selected_ticker

    # ── EMPTY STATE ───────────────────────────────────────────────────────────
    if not company:
        st.markdown("""
        <div class="search-empty">
            <div class="search-empty-icon">📈</div>
            <div class="search-empty-title">Search for any company above</div>
            <div class="search-empty-sub">
                Type a company name in the search bar above<br>
                Apple · Microsoft · Tesla · BMW · TCS · Goldman Sachs · Sony · and 100+ more
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ── FETCH DATA ────────────────────────────────────────────────────────────
    with st.spinner(f"Loading {company}..."):
        if mode == "live":
            df = fetch_live(ticker)
        else:
            df = fetch_stock(ticker, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))

    if df.empty:
        st.error(f"No data found for {ticker}. Try another company or time period.")
        st.stop()

    news = []
    api_key = os.getenv("NEWS_API_KEY", "")
    if api_key:
        news = fetch_news(company, ticker, api_key)

    latest = df.iloc[-1]
    prev   = df.iloc[-2] if len(df) > 1 else latest
    first  = df.iloc[0]
    day_chg     = latest["Close"] - prev["Close"]
    day_chg_pct = (day_chg / prev["Close"]) * 100
    period_chg  = ((latest["Close"] - first["Close"]) / first["Close"]) * 100
    day_cls     = "up" if day_chg >= 0 else "down"
    period_cls  = "up" if period_chg >= 0 else "down"
    day_arrow   = "▲" if day_chg >= 0 else "▼"
    abbr        = "".join([w[0] for w in company.split()[:3]]).upper()
    live_html   = '<span class="live-badge"><span class="live-dot-sm"></span>LIVE</span>' if mode == "live" else ""

    # ── HEADER ────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="page-header">
      <div class="company-block">
        <div class="company-avatar">{abbr}</div>
        <div>
          <div class="company-name">{company}{live_html}</div>
          <div class="company-meta">{ticker} · {period_lbl}</div>
        </div>
      </div>
      <div>
        <div class="big-price">${latest['Close']:.2f}</div>
        <div class="price-{'up' if day_chg>=0 else 'down'}">{day_arrow} {abs(day_chg_pct):.2f}% today</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if mode == "live":
        st.markdown(f'<div class="refresh-bar">🟢 Live data · Last updated: {datetime.now().strftime("%H:%M:%S")} · Use period selector to change view</div>', unsafe_allow_html=True)

    # ── METRICS ───────────────────────────────────────────────────────────────
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
        <div class="metric-val {period_cls}">{'▲' if period_chg>=0 else '▼'} {abs(period_chg):.2f}%</div>
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

    # ── CHART + NEWS ──────────────────────────────────────────────────────────
    chart_col, news_col = st.columns([3, 1], gap="medium")

    with chart_col:
        guide_extra = ""
        if mode == "future":
            guide_extra = " &nbsp;|&nbsp; <span style='color:#f59e0b;'>- - -</span> Dashed = AI prediction"

        st.markdown(f"""
        <div class="chart-guide">
            🟢 Green candle = price UP &nbsp;|&nbsp;
            🔴 Red = price DOWN &nbsp;|&nbsp;
            <span style="color:#60a5fa;">─</span> Blue = EMA 20 &nbsp;|&nbsp;
            <span style="color:#f59e0b;">─</span> Orange = EMA 50{guide_extra}
        </div>
        """, unsafe_allow_html=True)

        if mode == "future":
            st.markdown('<div class="pred-note">⚠️ AI Prediction — Based on historical trend. Indicative only. Not financial advice.</div>', unsafe_allow_html=True)

        rows = 1
        row_heights = [0.65]
        if show_rsi and mode != "live": rows += 1; row_heights.append(0.18)
        if show_vol: rows += 1; row_heights.append(0.17)

        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=row_heights)

        if mode == "live":
            fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Live Price",
                line=dict(color="#10b981", width=2), fill="tozeroy", fillcolor="rgba(16,185,129,0.05)"), row=1, col=1)
        else:
            fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                low=df["Low"], close=df["Close"], name="Daily Price",
                increasing_line_color="#10b981", decreasing_line_color="#ef4444",
                increasing_fillcolor="#064e3b", decreasing_fillcolor="#450a0a"), row=1, col=1)

        if show_ema and mode != "live":
            ema20 = df["Close"].ewm(span=20).mean()
            ema50 = df["Close"].ewm(span=50).mean()
            fig.add_trace(go.Scatter(x=df.index, y=ema20, name="EMA 20", line=dict(color="#60a5fa", width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=ema50, name="EMA 50", line=dict(color="#f59e0b", width=1.5)), row=1, col=1)

        if show_bb and mode != "live":
            sma = df["Close"].rolling(20).mean()
            std = df["Close"].rolling(20).std()
            fig.add_trace(go.Scatter(x=df.index, y=sma+std*2, name="BB Upper", line=dict(color="#a78bfa", width=1, dash="dash")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=sma-std*2, name="BB Lower", line=dict(color="#a78bfa", width=1, dash="dash"), fill="tonexty", fillcolor="rgba(167,139,250,0.04)"), row=1, col=1)

        if mode == "future" and pred_days > 0:
            future_dates, future_pred, upper, lower = predict_prices(df, pred_days)
            fig.add_trace(go.Scatter(x=[df.index[-1], future_dates[0]], y=[df["Close"].iloc[-1], future_pred[0]], line=dict(color="#f59e0b", width=2, dash="dash"), showlegend=False, hoverinfo="skip"), row=1, col=1)
            fig.add_trace(go.Scatter(x=future_dates, y=upper, line=dict(color="rgba(245,158,11,0)", width=0), showlegend=False, hoverinfo="skip"), row=1, col=1)
            fig.add_trace(go.Scatter(x=future_dates, y=lower, line=dict(color="rgba(245,158,11,0)", width=0), fill="tonexty", fillcolor="rgba(245,158,11,0.1)", showlegend=False, hoverinfo="skip"), row=1, col=1)
            fig.add_trace(go.Scatter(x=future_dates, y=future_pred, name=f"AI Prediction ({pred_days}d)", line=dict(color="#f59e0b", width=2.5, dash="dash"), mode="lines+markers", marker=dict(size=4, color="#f59e0b")), row=1, col=1)
            fig.add_vline(x=df.index[-1], line_dash="dot", line_color="rgba(245,158,11,0.4)", annotation_text="Prediction →", annotation_position="top right", annotation_font=dict(size=9, color="#f59e0b"))

        fig.update_yaxes(title_text="Price (USD $)", tickprefix="$", title_font=dict(size=10, color="#3a5080"), tickfont=dict(size=9, color="#3a5080"), gridcolor="rgba(99,179,255,0.05)", row=1, col=1)

        cur = 2
        if show_rsi and mode != "live":
            rsi = calc_rsi(df["Close"])
            fig.add_trace(go.Scatter(x=df.index, y=rsi, name="RSI", line=dict(color="#f472b6", width=1.5)), row=cur, col=1)
            fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,68,68,0.04)", line_width=0, row=cur, col=1)
            fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(16,185,129,0.04)", line_width=0, row=cur, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", opacity=0.35, annotation_text="Overbought 70", annotation_position="right", annotation_font=dict(size=8, color="#ef4444"), row=cur, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="#10b981", opacity=0.35, annotation_text="Oversold 30", annotation_position="right", annotation_font=dict(size=8, color="#10b981"), row=cur, col=1)
            fig.update_yaxes(title_text="RSI", range=[0,100], title_font=dict(size=9, color="#3a5080"), tickfont=dict(size=9, color="#3a5080"), gridcolor="rgba(99,179,255,0.05)", row=cur, col=1)
            cur += 1

        if show_vol:
            vcols = ["#10b981"] * len(df) if mode == "live" else ["#10b981" if df["Close"].iloc[i] >= df["Open"].iloc[i] else "#ef4444" for i in range(len(df))]
            fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color=vcols, opacity=0.65), row=cur, col=1)
            fig.update_yaxes(title_text="Volume", title_font=dict(size=9, color="#3a5080"), tickfont=dict(size=9, color="#3a5080"), gridcolor="rgba(99,179,255,0.05)", row=cur, col=1)

        fig.update_xaxes(title_text="Date", title_font=dict(size=9, color="#3a5080"), tickfont=dict(size=9, color="#3a5080"), gridcolor="rgba(99,179,255,0.05)", row=cur if (show_rsi or show_vol) else 1, col=1)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(5,15,46,0.6)", font=dict(family="DM Sans"), xaxis_rangeslider_visible=False, height=520, margin=dict(l=8,r=8,t=8,b=8), legend=dict(orientation="h", y=1.02, x=0, font=dict(size=9, color="#7a90c0"), bgcolor="rgba(0,0,0,0)"), hovermode="x unified", hoverlabel=dict(bgcolor="#0d2060", font_color="#e2e8f8", font_size=11))
        st.plotly_chart(fig, use_container_width=True)

        if show_rsi and mode not in ["live", "future"]:
            rsi_now = calc_rsi(df["Close"]).iloc[-1]
            if rsi_now > 70:
                st.markdown(f'<div class="rsi-hint rsi-over">RSI {rsi_now:.0f} — above 70, stock may be overbought</div>', unsafe_allow_html=True)
            elif rsi_now < 30:
                st.markdown(f'<div class="rsi-hint rsi-under">RSI {rsi_now:.0f} — below 30, stock may be oversold</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="rsi-hint rsi-neutral">RSI {rsi_now:.0f} — neutral zone, no strong signal</div>', unsafe_allow_html=True)

    with news_col:
        st.markdown(f'<div style="font-size:0.62rem;color:#3a5080;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:10px;padding-bottom:6px;border-bottom:1px solid rgba(99,179,255,0.07);">{company} News</div>', unsafe_allow_html=True)
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
                  <div class="news-meta" style="color:{mc};">{date} &nbsp;<span class="pill {pc}">{s}</span></div>
                  <a class="news-ttl" href="{url}" target="_blank">{title[:105]}{'...' if len(title)>105 else ''}</a>
                </div>""", unsafe_allow_html=True)

    # ── AI ANALYSIS ───────────────────────────────────────────────────────────
    if run_ai:
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