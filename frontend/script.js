const API_BASE = 'http://localhost:8000';

// Cache so we don't search same company twice
const searchCache = {};

// ── RESOLVE ANY COMPANY NAME TO TICKER ───────────────────────────────────
async function resolveStock(input) {
  const cleaned = input.trim();
  const key = cleaned.toLowerCase();

  // Return from cache if already searched
  if (searchCache[key]) return searchCache[key];

  try {
    const res = await fetch(`${API_BASE}/search?q=${encodeURIComponent(cleaned)}`);
    if (res.ok) {
      const data = await res.json();
      const result = { ticker: data.ticker, company: data.company };
      searchCache[key] = result;
      return result;
    }
  } catch (e) {}

  // Fallback — treat input as ticker directly
  const ticker = cleaned.toUpperCase();
  return { ticker, company: ticker };
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
    card.addEventListener('mouseleave', () => {
      card.style.transform = '';
    });
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
    renderTechnical(techData);
    renderSentiment(sentData);
    renderNews(newsData);
    renderPredictions(predData, ticker);

    document.querySelectorAll('.stock-card').forEach(c => c.style.borderColor = '');
    const activeCard = document.querySelector(`[data-ticker="${ticker}"]`);
    if (activeCard) activeCard.style.borderColor = 'var(--blue)';

  } catch (err) {
    hideLoading();
    document.getElementById('analysisArea').innerHTML =
      `<div class="glass-card analysis-content"><div class="error-msg">⚠ Sorry, could not find data for "${company}". Please try another company.</div></div>`;
  }
}

// ── RENDER FUNCTIONS ──────────────────────────────────────────────────────
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
      ${rec !== '—' ? `<div class="ai-analysis-text">${ai}</div>` : ''}
    </div>`;
}

function renderTechnical(data) {
  const rsiClass  = data.RSI.value > 70 ? 'bearish' : data.RSI.value < 30 ? 'bullish' : 'neutral';
  const macdClass = data.MACD.trend === 'Bullish' ? 'bullish' : 'bearish';
  const emaClass  = data.EMA.trend  === 'Bullish' ? 'bullish' : 'bearish';
  document.getElementById('technicalBlock').innerHTML = `
    <div class="insight-title">Technical Indicators</div>
    <div class="indicator-row"><span class="ind-name">RSI</span><span class="ind-value ${rsiClass}">${data.RSI.value}</span><span class="ind-signal ${rsiClass}">${data.RSI.signal}</span></div>
    <div class="indicator-row"><span class="ind-name">MACD</span><span class="ind-value ${macdClass}">${data.MACD.macd}</span><span class="ind-signal ${macdClass}">${data.MACD.trend}</span></div>
    <div class="indicator-row"><span class="ind-name">EMA 20</span><span class="ind-value">$${data.EMA.ema_20}</span><span class="ind-signal ${emaClass}">${data.EMA.trend}</span></div>
    <div class="indicator-row"><span class="ind-name">EMA 50</span><span class="ind-value">$${data.EMA.ema_50}</span><span class="ind-signal neutral">—</span></div>`;
}

function renderSentiment(data) {
  const cls   = `sent-${data.sentiment.toLowerCase()}`;
  const emoji = data.sentiment === 'Positive' ? '📈' : data.sentiment === 'Negative' ? '📉' : '➡️';
  document.getElementById('sentimentBlock').innerHTML = `
    <div class="insight-title">Market Sentiment</div>
    <div class="sentiment-pill ${cls}">${emoji} ${data.sentiment} <span style="opacity:0.7;font-size:0.6rem">${(data.confidence_score*100).toFixed(0)}%</span></div>`;
}

function renderNews(data) {
  if (!data.articles || data.articles.length === 0) {
    document.getElementById('newsBlock').innerHTML = `<div class="insight-title">Latest News</div><div class="insight-empty">No news found</div>`;
    return;
  }
  const items = data.articles.slice(0,3).map(a => `
    <div class="news-item">
      <div class="news-title">${a.title}</div>
      <div class="news-meta">${a.source} · ${a.published} <a href="${a.url}" target="_blank" class="news-link">READ →</a></div>
    </div>`).join('');
  document.getElementById('newsBlock').innerHTML = `<div class="insight-title">Latest News</div>${items}`;
}

function renderPredictions(data, ticker) {
  const predArea = document.getElementById('predictionArea');
  if (!predArea) return;
  document.getElementById('predTicker').textContent = ticker;
  predArea.style.display = 'block';
  const prices  = data['7_day_predictions'].map(p => p.predicted_price);
  const minP    = Math.min(...prices);
  const maxP    = Math.max(...prices);
  const range   = maxP - minP || 1;
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
      <div class="msg-avatar">⬡</div>
      <div class="msg-bubble"><div class="typing-dots"><span></span><span></span><span></span></div></div>
    </div>`;
  msgs.scrollTop = msgs.scrollHeight;

  const upper = text.toUpperCase();
  let response = '';

  try {
    const { ticker, company } = await resolveStock(text);

    if (upper.includes('PRICE') || upper.includes('COST') || upper.includes('WORTH') || upper.includes('STOCK')) {
      const data = await apiFetch('/price', { ticker });
      response = `📈 <strong>${company}</strong> is at <strong>$${data.current_price}</strong> — ${data.price_change_pct >= 0 ? '▲' : '▼'} ${Math.abs(data.price_change_pct).toFixed(2)}%`;
    } else if (upper.includes('NEWS') || upper.includes('LATEST')) {
      const data = await apiFetch('/news', { company, ticker });
      const top = data.articles?.[0];
      response = top ? `📰 Latest on <strong>${company}</strong>: "${top.title}"` : `No news found for ${company}.`;
    } else if (upper.includes('PREDICT') || upper.includes('FORECAST') || upper.includes('TOMORROW')) {
      const data = await apiFetch('/predict', { ticker });
      const day1 = data['7_day_predictions'][0];
      response = `🔮 <strong>${company}</strong> tomorrow: <strong>$${day1.predicted_price}</strong> — ${data.trend}`;
    } else if (upper.includes('BUY') || upper.includes('SELL') || upper.includes('HOLD') || upper.includes('ANALYS') || upper.includes('RECOMMEND')) {
      const data = await apiFetch('/analyze', { ticker, company });
      response = `🧠 <strong>${company} — ${data.recommendation}</strong><br><small>${data.ai_analysis?.substring(0,200)}...</small>`;
    } else if (upper.includes('RSI') || upper.includes('MACD') || upper.includes('TECHNICAL')) {
      const data = await apiFetch('/technical', { ticker });
      response = `📊 <strong>${company}</strong>: RSI=${data.RSI.value} (${data.RSI.signal}), MACD=${data.MACD.macd} (${data.MACD.trend})`;
    } else if (upper.includes('HELLO') || upper.includes('HI') || upper.includes('HEY')) {
      response = `👋 Hello! Type any company name like "Apple", "Nestle", "Samsung". I'll find it automatically!`;
    } else if (upper.includes('HELP')) {
      response = `🤖 Try:<br>• "Apple price"<br>• "Analyze Nestle"<br>• "Samsung prediction"<br>• "Latest Reliance news"<br>• "Should I buy Google?"`;
    } else {
      const data = await apiFetch('/price', { ticker });
      response = `💹 <strong>${company}</strong> — $${data.current_price} | H:$${data.high} | L:$${data.low} | Vol:${(data.volume/1e6).toFixed(1)}M`;
    }
  } catch (err) {
    response = `⚠ Sorry, I couldn't find that company. Try typing the full company name like "Apple", "Nestle", or "Samsung".`;
  }

  document.getElementById(typingId)?.remove();
  msgs.innerHTML += `
    <div class="chat-msg bot-msg fade-in">
      <div class="msg-avatar">⬡</div>
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