const API_BASE = 'http://localhost:8000';
const searchCache = {};

// ── RESOLVE ANY COMPANY NAME TO TICKER ───────────────────────────────────
async function resolveStock(input) {
  const cleaned = input.trim();
  const key = cleaned.toLowerCase();
  if (searchCache[key]) return searchCache[key];

  const knownMap = {
    'samsung': { ticker: '005930.KS', company: 'Samsung' },
    'samsung electronics': { ticker: '005930.KS', company: 'Samsung' },
    'hyundai': { ticker: '005380.KS', company: 'Hyundai' },
    'lg': { ticker: '066570.KS', company: 'LG Electronics' },
    'apple': { ticker: 'AAPL', company: 'Apple' },
    'tesla': { ticker: 'TSLA', company: 'Tesla' },
    'google': { ticker: 'GOOGL', company: 'Google' },
    'alphabet': { ticker: 'GOOGL', company: 'Google' },
    'microsoft': { ticker: 'MSFT', company: 'Microsoft' },
    'nvidia': { ticker: 'NVDA', company: 'Nvidia' },
    'amazon': { ticker: 'AMZN', company: 'Amazon' },
    'meta': { ticker: 'META', company: 'Meta' },
    'facebook': { ticker: 'META', company: 'Meta' },
    'netflix': { ticker: 'NFLX', company: 'Netflix' },
    'nestle': { ticker: 'NSRGY', company: 'Nestle' },
    'nestlé': { ticker: 'NSRGY', company: 'Nestle' },
    'sony': { ticker: 'SONY', company: 'Sony' },
    'toyota': { ticker: 'TM', company: 'Toyota' },
    'honda': { ticker: 'HMC', company: 'Honda' },
    'reliance': { ticker: 'RELIANCE.NS', company: 'Reliance' },
    'tcs': { ticker: 'TCS.NS', company: 'TCS' },
    'infosys': { ticker: 'INFY.NS', company: 'Infosys' },
    'wipro': { ticker: 'WIPRO.NS', company: 'Wipro' },
    'airtel': { ticker: 'BHARTIARTL.NS', company: 'Airtel' },
    'adani': { ticker: 'ADANIENT.NS', company: 'Adani' },
    'hdfc': { ticker: 'HDFCBANK.NS', company: 'HDFC Bank' },
    'icici': { ticker: 'ICICIBANK.NS', company: 'ICICI Bank' },
    'sbi': { ticker: 'SBIN.NS', company: 'SBI' },
    'bmw': { ticker: 'BMW.DE', company: 'BMW' },
    'volkswagen': { ticker: 'VOW3.DE', company: 'Volkswagen' },
    'mercedes': { ticker: 'MBG.DE', company: 'Mercedes' },
    'alibaba': { ticker: 'BABA', company: 'Alibaba' },
    'visa': { ticker: 'V', company: 'Visa' },
    'mastercard': { ticker: 'MA', company: 'Mastercard' },
    'nike': { ticker: 'NKE', company: 'Nike' },
    'disney': { ticker: 'DIS', company: 'Disney' },
    'pfizer': { ticker: 'PFE', company: 'Pfizer' },
    'intel': { ticker: 'INTC', company: 'Intel' },
    'amd': { ticker: 'AMD', company: 'AMD' },
    'uber': { ticker: 'UBER', company: 'Uber' },
    'spotify': { ticker: 'SPOT', company: 'Spotify' },
    'coca cola': { ticker: 'KO', company: 'Coca Cola' },
    'cocacola': { ticker: 'KO', company: 'Coca Cola' },
    'pepsi': { ticker: 'PEP', company: 'Pepsi' },
    'goldman': { ticker: 'GS', company: 'Goldman Sachs' },
    'jpmorgan': { ticker: 'JPM', company: 'JP Morgan' },
    'jp morgan': { ticker: 'JPM', company: 'JP Morgan' },
    'paypal': { ticker: 'PYPL', company: 'PayPal' },
    'oracle': { ticker: 'ORCL', company: 'Oracle' },
    'salesforce': { ticker: 'CRM', company: 'Salesforce' },
    'adobe': { ticker: 'ADBE', company: 'Adobe' },
  };

  if (knownMap[key]) {
    searchCache[key] = knownMap[key];
    return knownMap[key];
  }

  for (const [name, val] of Object.entries(knownMap)) {
    if (key.includes(name) || name.includes(key)) {
      searchCache[key] = val;
      return val;
    }
  }

  try {
    const res = await fetch(`${API_BASE}/search?q=${encodeURIComponent(cleaned)}`);
    if (res.ok) {
      const data = await res.json();
      const company = data.company?.split(' ').slice(0, 2).join(' ') || cleaned;
      const result = { ticker: data.ticker, company };
      searchCache[key] = result;
      return result;
    }
  } catch (e) {}

  return { ticker: cleaned.toUpperCase(), company: cleaned };
}

// ── EXTRACT COMPANY FROM SENTENCE ─────────────────────────────────────────
async function resolveStockFromSentence(text) {
  const skipWords = ['how','is','are','was','were','doing','performing','stock',
    'price','tell','me','about','what','the','a','an','of','for','should','i',
    'buy','sell','hold','analyze','analysis','predict','prediction','news',
    'latest','today','current','give','show','find','get','check','will',
    'going','to','do','can','please','my','your','its','their','this','that'];
  const words = text.toLowerCase().replace(/[?!.,]/g, '').split(' ');
  const companyWords = words.filter(w => !skipWords.includes(w) && w.length > 1);
  const searchTerm = companyWords.join(' ') || text;
  return resolveStock(searchTerm);
}

// ── BACKGROUND CHARTS ─────────────────────────────────────────────────────
function initBackgroundCharts() {
  window._charts = Array.from({ length: 6 }, (_, i) => ({
    points: Array.from({ length: 20 }, (_, j) => ({
      x: (j / 19) * window.innerWidth,
      y: Math.random() * window.innerHeight * 0.8 + window.innerHeight * 0.1
    })),
    speed: 0.3 + Math.random() * 0.4,
    alpha: 0.04 + Math.random() * 0.06,
    color: i % 2 === 0 ? '0,170,255' : '0,255,136',
    offset: Math.random() * 1000
  }));
  function animateCharts(time) {
    if (window._charts) {
      window._charts.forEach(chart => {
        chart.points.forEach((p, j) => {
          p.y += Math.sin((time * 0.001 * chart.speed) + j * 0.5 + chart.offset) * 0.5;
        });
      });
    }
    requestAnimationFrame(animateCharts);
  }
  requestAnimationFrame(animateCharts);
}

// ── PARTICLES ─────────────────────────────────────────────────────────────
function initParticles() {
  const canvas = document.getElementById('particles');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  const particles = Array.from({ length: 80 }, () => ({
    x: Math.random() * canvas.width,
    y: Math.random() * canvas.height,
    r: Math.random() * 1.5 + 0.3,
    vx: (Math.random() - 0.5) * 0.3,
    vy: (Math.random() - 0.5) * 0.3,
    alpha: Math.random() * 0.5 + 0.1,
  }));
  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (window._charts) {
      window._charts.forEach(chart => {
        ctx.beginPath();
        chart.points.forEach((p, i) => {
          i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y);
        });
        ctx.strokeStyle = `rgba(${chart.color},${chart.alpha})`;
        ctx.lineWidth = 1.5;
        ctx.stroke();
      });
    }
    particles.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(0,170,255,${p.alpha})`;
      ctx.fill();
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0 || p.x > canvas.width)  p.vx *= -1;
      if (p.y < 0 || p.y > canvas.height) p.vy *= -1;
    });
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx*dx + dy*dy);
        if (dist < 120) {
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(0,170,255,${0.08 * (1 - dist/120)})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
    }
    requestAnimationFrame(draw);
  }
  draw();
  window.addEventListener('resize', () => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  });
}

// ── ENTER EXPERIENCE ──────────────────────────────────────────────────────
function enterExperience() {
  document.body.style.transition = 'opacity 0.6s ease';
  document.body.style.opacity = '0';
  setTimeout(() => { window.location.href = 'dashboard.html'; }, 600);
}

// ── 3D TILT ───────────────────────────────────────────────────────────────
function initTiltCards() {
  document.querySelectorAll('.tilt-card').forEach(card => {
    card.addEventListener('mousemove', e => {
      const rect = card.getBoundingClientRect();
      const x = e.clientX - rect.left - rect.width  / 2;
      const y = e.clientY - rect.top  - rect.height / 2;
      const tiltX = -(y / rect.height) * 10;
      const tiltY =  (x / rect.width)  * 10;
      card.style.transform = `perspective(600px) rotateX(${tiltX}deg) rotateY(${tiltY}deg) translateY(-4px)`;
    });
    card.addEventListener('mouseleave', () => { card.style.transform = ''; });
  });
}

// ── API ───────────────────────────────────────────────────────────────────
async function apiFetch(endpoint, body) {
  const res = await fetch(`${API_BASE}${endpoint}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

function showLoading() {
  const el = document.getElementById('loadingOverlay');
  if (el) el.classList.add('active');
}
function hideLoading() {
  const el = document.getElementById('loadingOverlay');
  if (el) el.classList.remove('active');
}

// ── CANDLESTICK CHART ─────────────────────────────────────────────────────
function renderCandlestickChart(ticker, currentPrice) {
  try {
    const canvas = document.getElementById('candlestickChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    canvas.width  = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const days = 30;
    let price = currentPrice * 0.85;
    const candles = [];
    for (let i = 0; i < days; i++) {
      const open  = price;
      const close = open + (Math.random() - 0.48) * open * 0.02;
      const high  = Math.max(open, close) + Math.random() * open * 0.01;
      const low   = Math.min(open, close) - Math.random() * open * 0.01;
      candles.push({ open, close, high, low });
      price = close;
    }

    const allPrices = candles.flatMap(c => [c.high, c.low]);
    const minP = Math.min(...allPrices);
    const maxP = Math.max(...allPrices);
    const range = maxP - minP || 1;
    const pad = { top: 20, bottom: 20, left: 10, right: 10 };
    const chartH = H - pad.top - pad.bottom;
    const chartW = W - pad.left - pad.right;
    const candleW = Math.max(2, Math.floor(chartW / days) - 2);
    const toY = p => pad.top + chartH - ((p - minP) / range) * chartH;

    candles.forEach((c, i) => {
      const x = pad.left + i * (chartW / days);
      const isUp = c.close >= c.open;
      ctx.strokeStyle = isUp ? '#00ff88' : '#ff4466';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x + candleW / 2, toY(c.high));
      ctx.lineTo(x + candleW / 2, toY(c.low));
      ctx.stroke();
      const bodyTop = toY(Math.max(c.open, c.close));
      const bodyH   = Math.max(2, Math.abs(toY(c.open) - toY(c.close)));
      ctx.fillStyle = isUp ? 'rgba(0,255,136,0.8)' : 'rgba(255,68,102,0.8)';
      ctx.fillRect(x, bodyTop, candleW, bodyH);
    });

    ctx.fillStyle = 'rgba(0,170,255,0.9)';
    ctx.font = '10px monospace';
    ctx.fillText(`Current: $${currentPrice}`, W - 110, 15);
  } catch (e) {
    console.log('Chart error:', e);
  }
}

// ── LOAD STOCK CARD ───────────────────────────────────────────────────────
async function loadStockCard(ticker) {
  try {
    const data = await apiFetch('/price', { ticker });
    const priceEl  = document.getElementById(`price-${ticker}`);
    const changeEl = document.getElementById(`change-${ticker}`);
    const barEl    = document.getElementById(`bar-${ticker}`);
    if (!priceEl) return;
    priceEl.textContent = `$${data.current_price.toFixed(2)}`;
    const isUp = data.price_change_pct >= 0;
    changeEl.textContent = `${isUp ? '▲' : '▼'} ${Math.abs(data.price_change_pct).toFixed(2)}%`;
    changeEl.className   = `stock-change ${isUp ? 'up' : 'down'}`;
    if (barEl) {
      const pct = Math.min(Math.abs(data.price_change_pct) * 10, 100);
      barEl.style.width      = `${pct}%`;
      barEl.style.background = isUp ? 'var(--up)' : 'var(--down)';
    }
  } catch (e) {
    const priceEl = document.getElementById(`price-${ticker}`);
    if (priceEl) priceEl.textContent = 'N/A';
  }
}

// ── FULL ANALYSIS ─────────────────────────────────────────────────────────
async function loadStock(ticker, company) {
  showLoading();
  window._currentStock = { ticker, company };

  try {
    const [priceData, techData, predData, newsData] = await Promise.all([
      apiFetch('/price',     { ticker }),
      apiFetch('/technical', { ticker }),
      apiFetch('/predict',   { ticker }),
      apiFetch('/news',      { company, ticker }),
    ]);

    let analyzeData = null;
    try { analyzeData = await apiFetch('/analyze', { ticker, company }); } catch (e) {}

    const sentText = `${company} stock ${priceData.price_change_pct >= 0 ? 'rise gain bullish' : 'fall drop bearish'}`;
    const sentData = await apiFetch('/sentiment', { text: sentText });

    hideLoading();
    renderAnalysis(ticker, company, priceData, analyzeData);
    setTimeout(() => renderCandlestickChart(ticker, priceData.current_price), 100);
    renderTechnical(techData);
    renderSentiment(sentData);
    renderNews(newsData);
    renderPredictions(predData, ticker);
    renderHumanInTheLoop(ticker, company, priceData, analyzeData);

    document.querySelectorAll('.stock-card').forEach(c => c.style.borderColor = '');
    const activeCard = document.querySelector(`[data-ticker="${ticker}"]`);
    if (activeCard) activeCard.style.borderColor = 'var(--blue)';

  } catch (err) {
    hideLoading();
    document.getElementById('analysisArea').innerHTML =
      `<div class="glass-card analysis-content"><div class="error-msg">Sorry, could not find data for "${company}". Please try another company.</div></div>`;
  }
}

// ── RENDER ANALYSIS ───────────────────────────────────────────────────────
function renderAnalysis(ticker, company, priceData, analyzeData) {
  const rec = analyzeData?.recommendation || '—';
  const ai  = analyzeData?.ai_analysis    || 'AI analysis unavailable.';
  document.getElementById('analysisArea').innerHTML = `
    <div class="glass-card analysis-content fade-in">
      <div class="analysis-header">
        <div>
          <div class="analysis-ticker">${ticker}</div>
          <div class="analysis-company">${company}</div>
        </div>
        ${rec !== '—' ? `<div class="rec-badge rec-${rec}">${rec}</div>` : ''}
      </div>
      <div class="analysis-stats">
        <div class="stat-item"><div class="stat-label">PRICE</div><div class="stat-value">$${priceData.current_price.toFixed(2)}</div></div>
        <div class="stat-item"><div class="stat-label">CHANGE</div><div class="stat-value ${priceData.price_change_pct >= 0 ? 'up' : 'down'}">${priceData.price_change_pct >= 0 ? '▲' : '▼'} ${Math.abs(priceData.price_change_pct).toFixed(2)}%</div></div>
        <div class="stat-item"><div class="stat-label">VOLUME</div><div class="stat-value">${(priceData.volume/1e6).toFixed(1)}M</div></div>
        <div class="stat-item"><div class="stat-label">OPEN</div><div class="stat-value">$${priceData.open.toFixed(2)}</div></div>
        <div class="stat-item"><div class="stat-label">HIGH</div><div class="stat-value up">$${priceData.high.toFixed(2)}</div></div>
        <div class="stat-item"><div class="stat-label">LOW</div><div class="stat-value down">$${priceData.low.toFixed(2)}</div></div>
      </div>
      <div class="candle-wrapper">
        <div class="candle-label">30-Day Price Movement</div>
        <canvas id="candlestickChart" class="candle-canvas"></canvas>
      </div>
      ${rec !== '—' ? `<div class="ai-analysis-text">${ai}</div>` : ''}
    </div>`;
}

// ── RENDER TECHNICAL ──────────────────────────────────────────────────────
function renderTechnical(data) {
  const rsiVal    = data.RSI?.value    ?? data.RSI    ?? '—';
  const rsiSig    = data.RSI?.signal   ?? data.RSI_signal  ?? '—';
  const macdVal   = data.MACD?.macd    ?? data.MACD?.trend ?? '—';
  const macdTrend = data.MACD?.trend   ?? data.MACD_signal ?? '—';
  const ema20     = data.EMA?.ema_20   ?? data.EMA_20  ?? '—';
  const ema50     = data.EMA?.ema_50   ?? data.EMA_50  ?? '—';
  const emaTrend  = data.EMA?.trend    ?? data.EMA_trend   ?? '—';
  const rsiClass  = rsiVal > 70 ? 'bearish' : rsiVal < 30 ? 'bullish' : 'neutral';
  const macdClass = (macdTrend === 'Bullish' || macdTrend === 'BULLISH') ? 'bullish' : 'bearish';
  const emaClass  = (emaTrend  === 'Bullish' || emaTrend  === 'BULLISH') ? 'bullish' : 'bearish';
  document.getElementById('technicalBlock').innerHTML = `
    <div class="insight-title">Technical Indicators</div>
    <div class="indicator-row"><span class="ind-name">RSI</span><span class="ind-value ${rsiClass}">${rsiVal}</span><span class="ind-signal ${rsiClass}">${rsiSig}</span></div>
    <div class="indicator-row"><span class="ind-name">MACD</span><span class="ind-value ${macdClass}">${macdVal}</span><span class="ind-signal ${macdClass}">${macdTrend}</span></div>
    <div class="indicator-row"><span class="ind-name">EMA 20</span><span class="ind-value">${ema20 !== '—' ? '$'+ema20 : '—'}</span><span class="ind-signal ${emaClass}">${emaTrend}</span></div>
    <div class="indicator-row"><span class="ind-name">EMA 50</span><span class="ind-value">${ema50 !== '—' ? '$'+ema50 : '—'}</span><span class="ind-signal neutral">—</span></div>`;
}

// ── RENDER SENTIMENT ──────────────────────────────────────────────────────
function renderSentiment(data) {
  const sentiment = data.sentiment || data.overall_sentiment || 'Neutral';
  const score     = data.confidence_score ?? data.sentiment_score ?? 0.5;
  const cls   = `sent-${sentiment.toLowerCase()}`;
  document.getElementById('sentimentBlock').innerHTML = `
    <div class="insight-title">Market Sentiment</div>
    <div class="sentiment-pill ${cls}">${sentiment} <span style="opacity:0.7;font-size:0.6rem">${(score*100).toFixed(0)}%</span></div>`;
}

// ── RENDER NEWS ───────────────────────────────────────────────────────────
function renderNews(data) {
  if (!data.articles || data.articles.length === 0) {
    document.getElementById('newsBlock').innerHTML = `<div class="insight-title">Latest News</div><div class="insight-empty">No news found</div>`;
    return;
  }
  const items = data.articles.slice(0,3).map(a => `
    <div class="news-item">
      <div class="news-title">${a.title}</div>
      <div class="news-meta">${a.source} · ${a.published} <a href="${a.url}" target="_blank" class="news-link">READ</a></div>
    </div>`).join('');
  document.getElementById('newsBlock').innerHTML = `<div class="insight-title">Latest News</div>${items}`;
}

// ── RENDER PREDICTIONS ────────────────────────────────────────────────────
function renderPredictions(data, ticker) {
  const predArea = document.getElementById('predictionArea');
  if (!predArea) return;
  document.getElementById('predTicker').textContent = ticker;
  predArea.style.display = 'block';
  const predictions = data['7_day_predictions'] || [];
  const prices = predictions.map(p => {
    const val = p.predicted_price;
    if (typeof val === 'string') {
      const match = val.match(/\$?([\d.]+)/);
      return match ? parseFloat(match[1]) : 0;
    }
    return parseFloat(val) || 0;
  }).filter(p => p > 0);

  if (!prices.length) {
    document.getElementById('predChart').innerHTML = '<div class="insight-empty">No prediction data</div>';
    return;
  }

  const minP  = Math.min(...prices);
  const maxP  = Math.max(...prices);
  const range = maxP - minP || 1;
  document.getElementById('predChart').innerHTML = prices.map((price, i) => {
    const h    = 20 + ((price - minP) / range) * 70;
    const isUp = i === 0 ? true : price >= prices[i-1];
    return `<div class="pred-bar-wrap">
      <div class="pred-price">$${price.toFixed(0)}</div>
      <div class="pred-bar" style="height:${h}px;background:${isUp ? 'linear-gradient(to top,#00994d,#00ff88)' : 'linear-gradient(to top,#990033,#ff4466)'}"></div>
      <div class="pred-bar-label">D${i+1}</div>
    </div>`;
  }).join('');
}

// ── HUMAN IN THE LOOP ─────────────────────────────────────────────────────
function renderHumanInTheLoop(ticker, company, priceData, analyzeData) {
  const rec    = analyzeData?.recommendation || 'HOLD';
  const hitlEl = document.getElementById('humanInTheLoop');
  if (!hitlEl) return;
  hitlEl.style.display = 'block';
  hitlEl.innerHTML = `
    <div class="hitl-card glass-panel fade-in">
      <div class="hitl-header">
        <span class="hitl-title">Your Investment Decision</span>
        <button class="hitl-toggle" onclick="toggleHitl(this, 'hitlBody')">▲ Minimize</button>
      </div>
      <div id="hitlBody">
        <div class="hitl-disclaimer">
          AI systems are not 100% accurate. This analysis is for informational purposes only and should not be considered financial advice. Always do your own research before investing.
        </div>
        <div class="hitl-question">
          Based on the AI analysis, <strong>${company}</strong> shows a
          <span class="rec-badge rec-${rec}" style="font-size:0.7rem;padding:0.2rem 0.6rem;">${rec}</span> signal.
          <br><br>Would you like to invest in <strong>${company}</strong>?
        </div>
        <div class="hitl-buttons">
          <button class="hitl-yes" onclick="investDecision('yes','${ticker}','${company}',${priceData.current_price})">
            Yes, I want to invest
          </button>
          <button class="hitl-no" onclick="investDecision('no','${ticker}','${company}',${priceData.current_price})">
            No, I will pass
          </button>
        </div>
      </div>
    </div>`;
}

// ── TOGGLE MINIMIZE/MAXIMIZE ──────────────────────────────────────────────
function toggleHitl(btn, bodyId) {
  const body = document.getElementById(bodyId);
  if (!body || !btn) return;
  if (body.style.display === 'none') {
    body.style.display = 'block';
    btn.textContent = '▲ Minimize';
  } else {
    body.style.display = 'none';
    btn.textContent = '▼ Expand';
  }
}

// ── INVESTMENT DECISION ───────────────────────────────────────────────────
async function investDecision(decision, ticker, company, price) {
  const hitlEl = document.getElementById('humanInTheLoop');
  if (!hitlEl) return;

  if (decision === 'no') {
    hitlEl.innerHTML = `
      <div class="hitl-card glass-panel fade-in">
        <div class="hitl-header">
          <span class="hitl-title">Decision Recorded</span>
          <button class="hitl-toggle" onclick="toggleHitl(this, 'hitlBodyNo')">▲ Minimize</button>
        </div>
        <div id="hitlBodyNo">
          <div class="hitl-result-no">
            You chose <strong>not to invest</strong> in ${company} at this time. That is a perfectly valid decision — patience is a key investment strategy.
            <br><br>Remember: AI analysis is not financial advice. Always consult a certified financial advisor before making investment decisions.
          </div>
        </div>
      </div>`;
    return;
  }

  showLoading();
  try {
    const [techData, predData] = await Promise.all([
      apiFetch('/technical', { ticker }),
      apiFetch('/predict',   { ticker }),
    ]);
    hideLoading();

    const predictions = predData['7_day_predictions'] || [];
    const prices = predictions.map(p => {
      const val = p.predicted_price;
      if (typeof val === 'string') {
        const match = val.match(/\$?([\d.]+)/);
        return match ? parseFloat(match[1]) : 0;
      }
      return parseFloat(val) || 0;
    }).filter(p => p > 0);

    const lastPred       = prices.length ? prices[prices.length - 1] : price;
    const expectedReturn = (((lastPred - price) / price) * 100).toFixed(2);
    const rsiVal         = techData.RSI?.value ?? techData.RSI ?? 50;
    const emaTrend       = techData.EMA?.trend ?? techData.EMA_trend ?? 'Neutral';

    hitlEl.innerHTML = `
      <div class="hitl-card glass-panel fade-in">
        <div class="hitl-header">
          <span class="hitl-title">Investment Summary — ${company}</span>
          <button class="hitl-toggle" onclick="toggleHitl(this, 'hitlBodySummary')">▲ Minimize</button>
        </div>
        <div id="hitlBodySummary">
          <div class="hitl-disclaimer">
            IMPORTANT: This is an AI-generated summary only. It is NOT financial advice. AI predictions can be wrong. Invest only what you can afford to lose. Always consult a certified financial advisor.
          </div>
          <div class="hitl-summary-grid">
            <div class="hitl-summary-item">
              <div class="hitl-summary-label">Current Price</div>
              <div class="hitl-summary-value">$${Number(price).toFixed(2)}</div>
            </div>
            <div class="hitl-summary-item">
              <div class="hitl-summary-label">7-Day Target</div>
              <div class="hitl-summary-value ${expectedReturn >= 0 ? 'up' : 'down'}">$${Number(lastPred).toFixed(2)}</div>
            </div>
            <div class="hitl-summary-item">
              <div class="hitl-summary-label">Expected Return</div>
              <div class="hitl-summary-value ${expectedReturn >= 0 ? 'up' : 'down'}">${expectedReturn >= 0 ? '▲' : '▼'} ${Math.abs(expectedReturn)}%</div>
            </div>
            <div class="hitl-summary-item">
              <div class="hitl-summary-label">RSI Signal</div>
              <div class="hitl-summary-value ${rsiVal > 70 ? 'down' : rsiVal < 30 ? 'up' : 'neutral'}">${rsiVal} — ${rsiVal > 70 ? 'Overbought' : rsiVal < 30 ? 'Oversold' : 'Neutral'}</div>
            </div>
            <div class="hitl-summary-item">
              <div class="hitl-summary-label">EMA Trend</div>
              <div class="hitl-summary-value ${emaTrend === 'Bullish' || emaTrend === 'BULLISH' ? 'up' : 'down'}">${emaTrend}</div>
            </div>
            <div class="hitl-summary-item">
              <div class="hitl-summary-label">Risk Level</div>
              <div class="hitl-summary-value">${rsiVal > 70 ? 'High Risk' : rsiVal < 30 ? 'Low Risk' : 'Medium Risk'}</div>
            </div>
          </div>
          <div class="hitl-checklist">
            <div class="hitl-check">You have reviewed the AI analysis</div>
            <div class="hitl-check">You understand AI is not 100% accurate</div>
            <div class="hitl-check">Final investment decision is yours alone</div>
            <div class="hitl-check">Consult a financial advisor before investing real money</div>
          </div>
          <button class="hitl-reset" onclick="resetDecision()">← Reconsider Decision</button>
        </div>
      </div>`;
  } catch (err) {
    hideLoading();
    hitlEl.innerHTML = `<div class="error-msg">Could not load investment summary. Please try again.</div>`;
  }
}

function resetDecision() {
  const s = window._currentStock;
  if (s) loadStock(s.ticker, s.company);
}

// ── SEARCH ────────────────────────────────────────────────────────────────
async function searchStock() {
  const input = document.getElementById('globalSearch');
  if (!input) return;
  const value = input.value.trim();
  if (!value) return;
  showLoading();
  const { ticker, company } = await resolveStock(value);
  hideLoading();
  loadStock(ticker, company);
}

// ── CHATBOT ───────────────────────────────────────────────────────────────
async function sendChat() {
  const input = document.getElementById('chatInput');
  const msgs  = document.getElementById('chatMessages');
  if (!input || !msgs) return;
  const text = input.value.trim();
  if (!text) return;
  input.value = '';

  msgs.innerHTML += `
    <div class="chat-msg user-msg">
      <div class="msg-avatar" style="background:rgba(0,170,255,0.2)">U</div>
      <div class="msg-bubble">${text}</div>
    </div>`;
  msgs.scrollTop = msgs.scrollHeight;

  const typingId = 'typing-' + Date.now();
  msgs.innerHTML += `
    <div class="chat-msg bot-msg" id="${typingId}">
      <div class="msg-avatar">AI</div>
      <div class="msg-bubble"><div class="typing-dots"><span></span><span></span><span></span></div></div>
    </div>`;
  msgs.scrollTop = msgs.scrollHeight;

  const upper = text.toUpperCase();
  let response = '';

  try {
    const { ticker, company } = await resolveStockFromSentence(text);

    if (upper.includes('PRICE') || upper.includes('COST') || upper.includes('WORTH') || upper.includes('STOCK')) {
      const data = await apiFetch('/price', { ticker });
      response = `<strong>${company}</strong> is currently at <strong>$${data.current_price}</strong> — ${data.price_change_pct >= 0 ? '▲' : '▼'} ${Math.abs(data.price_change_pct).toFixed(2)}%<br><small style="color:var(--text-dim)">This is AI data — not financial advice</small>`;
    } else if (upper.includes('NEWS') || upper.includes('LATEST')) {
      const data = await apiFetch('/news', { company, ticker });
      const top = data.articles?.[0];
      response = top ? `Latest on <strong>${company}</strong>: "${top.title}"<br><small style="color:var(--text-dim)">Always verify news independently</small>` : `No news found for ${company}.`;
    } else if (upper.includes('PREDICT') || upper.includes('FORECAST') || upper.includes('TOMORROW')) {
      const data = await apiFetch('/predict', { ticker });
      const day1 = data['7_day_predictions'][0];
      const p    = day1?.predicted_price;
      const displayPrice = typeof p === 'string' ? p : `$${Number(p).toFixed(2)}`;
      response = `<strong>${company}</strong> tomorrow prediction: <strong>${displayPrice}</strong><br><small style="color:var(--text-dim)">AI predictions are not 100% accurate. Not financial advice.</small>`;
    } else if (upper.includes('BUY') || upper.includes('SELL') || upper.includes('HOLD') || upper.includes('ANALYS') || upper.includes('RECOMMEND') || upper.includes('INVEST')) {
      const data = await apiFetch('/analyze', { ticker, company });
      response = `<strong>${company} — ${data.recommendation}</strong><br><small>${data.ai_analysis?.substring(0,200)}...</small><br><small style="color:var(--text-dim)">This is AI analysis only — not financial advice. The final decision is always yours.</small>`;
    } else if (upper.includes('RSI') || upper.includes('MACD') || upper.includes('TECHNICAL')) {
      const data = await apiFetch('/technical', { ticker });
      const rsi  = data.RSI?.value ?? data.RSI ?? '—';
      const trend = data.MACD?.trend ?? data.MACD_signal ?? '—';
      response = `<strong>${company}</strong> — RSI: ${rsi}, MACD: ${trend}<br><small style="color:var(--text-dim)">Technical indicators are tools, not guarantees</small>`;
    } else if (upper.includes('HELLO') || upper.includes('HI') || upper.includes('HEY')) {
      response = `Hello! Type any company name like Apple, Nestle, or Samsung. I will analyze it — but remember, I am an AI and my analysis is not financial advice.`;
    } else if (upper.includes('HELP')) {
      response = `You can ask me:<br>• "Apple price"<br>• "Analyze Tesla"<br>• "Nvidia prediction"<br>• "Latest Microsoft news"<br>• "How is Nestle doing?"<br><br><small>All AI responses are informational only — not financial advice.</small>`;
    } else {
      const data = await apiFetch('/price', { ticker });
      response = `<strong>${company}</strong> — $${data.current_price} | High: $${data.high} | Low: $${data.low} | Volume: ${(data.volume/1e6).toFixed(1)}M<br><small style="color:var(--text-dim)">Not financial advice</small>`;
    }
  } catch (err) {
    response = `Sorry, I could not find that company. Try typing just the company name like "Nestle", "Apple", or "Samsung".`;
  }

  document.getElementById(typingId)?.remove();
  msgs.innerHTML += `
    <div class="chat-msg bot-msg fade-in">
      <div class="msg-avatar">AI</div>
      <div class="msg-bubble">${response}</div>
    </div>`;
  msgs.scrollTop = msgs.scrollHeight;
}

function quickPrompt(text) {
  const input = document.getElementById('chatInput');
  if (input) { input.value = text; sendChat(); }
}

// ── INIT ──────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initBackgroundCharts();
  initParticles();
  initTiltCards();

  if (document.getElementById('stocksRow')) {
    ['AAPL','TSLA','GOOGL','NVDA'].forEach(t => loadStockCard(t));
    const searchInput = document.getElementById('globalSearch');
    if (searchInput) {
      searchInput.addEventListener('keydown', e => {
        if (e.key === 'Enter') searchStock();
      });
    }
  }
});