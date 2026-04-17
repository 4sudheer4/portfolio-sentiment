import asyncio
import hashlib
import os
import secrets
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import jwt
import requests
import robin_stocks.robinhood as rh
import yfinance as yf
from dotenv import load_dotenv
from fastapi import Cookie, FastAPI, WebSocket
from fastapi.responses import HTMLResponse

# Prevent yfinance SQLite lock conflicts with threading
try:
    from yfinance import set_tz_cache_location
    set_tz_cache_location("/tmp/yfinance")
except Exception:
    pass

load_dotenv()

app = FastAPI()

HF_API_URL = "https://router.huggingface.co/hf-inference/models/ProsusAI/finbert"
HF_TOKEN   = os.getenv("HF_TOKEN")
JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_hex(32))

LABEL_TO_SCORE   = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
SENTIMENT_WEIGHT = 0.55
MOMENTUM_WEIGHT  = 0.45

# ── Cache — keyed by md5(jwt_token)[:8] ──────────────────────────────────────
_caches    = {}
_logged_in = False
_login_lock = None
CACHE_TTL  = 1800
JWT_EXPIRY = 86400

# ── Thread pool ───────────────────────────────────────────────────────────────
_executor = ThreadPoolExecutor(max_workers=20)

# ── JWT helpers ───────────────────────────────────────────────────────────────
def _issue_jwt():
    return jwt.encode({
        "uid": secrets.token_hex(8),
        "iat": int(time.time()),
        "exp": int(time.time()) + JWT_EXPIRY,
    }, JWT_SECRET, algorithm="HS256")

def _verify_jwt(token):
    try:
        jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return True
    except Exception:
        return False

def _cache_key(token):
    return hashlib.md5(token.encode()).hexdigest()[:8]

# ── Robinhood login ───────────────────────────────────────────────────────────
def _do_rh_login():
    username = os.getenv("RH_USERNAME")
    password = os.getenv("RH_PASSWORD")
    if not username or not password:
        raise ValueError("Missing credentials.")
    rh.login(username, password, store_session=True)

async def login_to_robinhood():
    global _logged_in, _login_lock
    if _logged_in:
        return
    if _login_lock is None:
        _login_lock = asyncio.Lock()
    async with _login_lock:
        if _logged_in:
            return
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _do_rh_login)
        _logged_in = True
        print("Logged into Robinhood.")

# ── Step 1: Collect full news items per ticker ────────────────────────────────
def collect_news(tickers: list) -> dict:
    """
    Fetch full news items for all tickers.
    Returns {ticker: [full_item1, full_item2]}
    Stores full items (not just titles) to preserve timestamps
    for recency weighting later.
    """
    def fetch_one(ticker):
        try:
            news = yf.Ticker(ticker).news or []
            return ticker, news[:2]
        except Exception as e:
            print(f"News error {ticker}: {e}")
            return ticker, []
    result = {}
    with ThreadPoolExecutor(max_workers=20) as ex:
        futures = {ex.submit(fetch_one, t):t for t in tickers}
        for future in as_completed(futures):
            ticker, news = future.result()
            result[ticker] = news
    return result


# ── Step 2: Batch score all headlines in ONE API call ─────────────────────────
def batch_score_finbert(headlines: list) -> dict:
    """
    Send ALL headlines to HuggingFace in one POST request.
    Returns {headline_text: compound_score}

    Before: 56 individual API calls → 40-80s
    After:  1 batch API call        → 3-5s
    """
    if not headlines:
        return {}

    for attempt in range(3):
        try:
            response = requests.post(
                HF_API_URL,
                headers={"Authorization": f"Bearer {HF_TOKEN}"},
                json={"inputs": headlines},
                timeout=30
            )

            if not response.text.strip():
                print(f"  HF empty response (attempt {attempt+1})")
                time.sleep(2)
                continue

            results = response.json()

            # Model still loading — HF returns {"error": ..., "estimated_time": N}
            if isinstance(results, dict) and "error" in results:
                wait = min(results.get("estimated_time", 10), 20)
                print(f"  HF model loading, waiting {wait}s...")
                time.sleep(wait)
                continue

            # Parse batch results
            # results = [[{label, score}, ...], [...], [...]]
            # One inner list per headline, in same order as input
            scores = {}
            for headline, label_list in zip(headlines, results):
                if isinstance(label_list, list):
                    best   = max(label_list, key=lambda x: x["score"])
                    label  = best["label"].lower()
                    conf   = best["score"]
                    scores[headline] = LABEL_TO_SCORE.get(label, 0.0) * conf
                else:
                    scores[headline] = 0.0
            return scores

        except Exception as e:
            print(f"  HF batch error (attempt {attempt+1}): {e}")
            if attempt < 2:
                time.sleep(2)

    # All retries failed — return neutral for everything
    return {h: 0.0 for h in headlines}

# ── Step 3: Compute weighted sentiment score per ticker ───────────────────────
def compute_ticker_score(news_items: list, headline_scores: dict) -> float:
    """
    Apply exponential recency decay to pre-scored headlines.

    Half-life = 3 days (259200 seconds)
    age=0d → weight=1.0 (full)
    age=3d → weight=0.5 (half)
    age=6d → weight=0.25 (quarter)

    Returns weighted average in [-1, 1]
    """
    now          = time.time()
    weighted_sum = 0.0
    weight_total = 0.0

    for item in news_items:
        title = (
            item.get("title")
            or item.get("content", {}).get("title")
            or ""
        )
        if not title or title not in headline_scores:
            continue

        pub_time       = item.get("providerPublishTime") or now
        age_seconds    = max(now - pub_time, 0)
        recency_weight = 0.5 ** (age_seconds / 259200)
        score          = headline_scores[title]

        weighted_sum  += score * recency_weight
        weight_total  += recency_weight

    return round(weighted_sum / weight_total, 4) if weight_total else 0.0

# ── Step 4: Price momentum ────────────────────────────────────────────────────
def get_price_momentum(ticker: str) -> dict:
    """
    5-day price % change normalized to [-1, 1].
    Caps at ±20% so outliers don't dominate the blend.
    """
    for attempt in range(3):
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if len(hist) < 2:
                return {"pct": 0.0, "score": 0.0}
            pct   = (hist["Close"].iloc[-1] - hist["Close"].iloc[0]) / hist["Close"].iloc[0]
            score = max(-1.0, min(1.0, pct / 0.20))
            return {"pct": round(pct * 100, 2), "score": round(score, 4)}
        except Exception as e:
            if "database is locked" in str(e) and attempt < 2:
                time.sleep(0.1)
                continue
            print(f"  Momentum error {ticker}: {e}")
            return {"pct": 0.0, "score": 0.0}

# ── Step 5: Combined signal ───────────────────────────────────────────────────
def combined_signal(sentiment_score: float, momentum_score: float) -> float:
    return round(
        SENTIMENT_WEIGHT * sentiment_score + MOMENTUM_WEIGHT * momentum_score, 4
    )

# ── HTML template ─────────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Portfolio Signal</title>
  <link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&display=swap" rel="stylesheet"/>
  <style>
    :root {
      --bg:      #08090d;
      --surface: #0f1117;
      --border:  #1e2130;
      --muted:   #4a5068;
      --text:    #e8eaf0;
      --bull:    #22c55e;
      --bear:    #ef4444;
      --neutral: #f59e0b;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg); color: var(--text);
      font-family: ui-monospace, 'SF Mono', monospace;
      min-height: 100vh; padding: 28px 16px 60px;
    }
    header { max-width: 480px; margin: 0 auto 28px; }
    header h1 {
      font-family: -apple-system, BlinkMacSystemFont, sans-serif;
      font-size: 1.6rem; font-weight: 800; letter-spacing: -0.5px;
    }
    header h1 span { color: var(--bull); }
    .meta { margin-top: 6px; font-size: 0.7rem; color: var(--muted); }
    .status {
      max-width: 480px; margin: 0 auto 16px;
      font-size: 0.72rem; color: var(--muted);
      display: flex; align-items: center; gap: 8px;
    }
    .status-dot {
      width: 7px; height: 7px; border-radius: 50%;
      background: var(--muted); flex-shrink: 0; transition: background 0.3s;
    }
    .status-dot.live { background: var(--bull); animation: pulse 1.5s infinite; }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
    .legend {
      max-width: 480px; margin: 0 auto 20px;
      display: flex; gap: 16px; font-size: 0.68rem; color: var(--muted);
    }
    .legend-dot {
      display: inline-block; width: 7px; height: 7px;
      border-radius: 50%; margin-right: 5px; vertical-align: middle;
    }
    .cards { max-width: 480px; margin: 0 auto; }
    .card {
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 14px; padding: 16px 18px; margin-bottom: 10px;
      position: relative; overflow: hidden;
      opacity: 0; transform: translateY(12px);
      transition: opacity 0.4s ease, transform 0.4s ease;
    }
    .card.visible { opacity: 1; transform: translateY(0); }
    .card::before {
      content: ''; position: absolute;
      left: 0; top: 0; bottom: 0; width: 3px;
      border-radius: 14px 0 0 14px;
    }
    .card.bull::before  { background: var(--bull); }
    .card.bear::before  { background: var(--bear); }
    .card.neutral::before { background: var(--neutral); }
    .card-row { display: flex; justify-content: space-between; align-items: baseline; }
    .ticker {
      font-family: -apple-system, BlinkMacSystemFont, sans-serif;
      font-size: 1.25rem; font-weight: 800;
    }
    .final-score { font-size: 1.1rem; font-weight: 500; }
    .sub-row { display: flex; gap: 14px; margin-top: 10px; font-size: 0.7rem; color: var(--muted); }
    .badge {
      display: inline-flex; align-items: center; gap: 5px;
      margin-top: 10px; padding: 4px 10px;
      border-radius: 20px; font-size: 0.72rem; font-weight: 500;
    }
    .badge-dot { width: 6px; height: 6px; border-radius: 50%; }
    .action { margin-top: 6px; font-size: 0.72rem; color: var(--muted); }
    .bar-track {
      margin-top: 12px; height: 4px;
      background: var(--border); border-radius: 99px; overflow: hidden;
    }
    .bar-fill { height: 100%; border-radius: 99px; }
    footer {
      text-align: center; font-size: 0.68rem;
      color: var(--muted); margin-top: 32px;
    }
    footer a { color: var(--muted); text-decoration: underline; }
  </style>
</head>
<body>
<header>
  <h1>Portfolio <span>Signal</span></h1>
  <p class="meta">FinBERT NLP · 5-day momentum · blended 55/45</p>
</header>
<div class="status">
  <div class="status-dot" id="status-dot"></div>
  <span id="status-text">Connecting...</span>
</div>
<div class="legend">
  <span><span class="legend-dot" style="background:var(--bull)"></span>Bullish ≥ 0.15</span>
  <span><span class="legend-dot" style="background:var(--neutral)"></span>Neutral</span>
  <span><span class="legend-dot" style="background:var(--bear)"></span>Bearish ≤ -0.15</span>
</div>
<div class="cards" id="cards"></div>
<footer>
  Scores are informational only — not financial advice.<br/>
  <a href="/">Refresh</a> to pull latest data.
</footer>
<script>
  const dot    = document.getElementById('status-dot');
  const status = document.getElementById('status-text');
  const cards  = document.getElementById('cards');

  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(`${protocol}//${location.host}/ws`);

  ws.onopen = () => {
    dot.classList.add('live');
    status.textContent = 'Analyzing portfolio...';
    ws.send('start_analysis');
  };

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === 'status') {
      status.textContent = msg.message;
    }
    if (msg.type === 'ticker_result') {
      const data = msg.data;
      status.textContent = `Analyzed ${data.ticker}...`;
      const bar  = Math.round(((data.final_score + 1) / 2) * 100);
      const card = document.createElement('div');
      card.className = `card ${data.tier}`;
      card.innerHTML = `
        <div class="card-row">
          <span class="ticker">${data.ticker}</span>
          <span class="final-score" style="color:${data.color}">${data.final_score}</span>
        </div>
        <div class="sub-row">
          <span>NLP <strong>${data.sent_score}</strong></span>
          <span>5d <strong style="color:${data.mom_pct>=0?'var(--bull)':'var(--bear)'}">
            ${data.mom_pct>=0?'+':''}${data.mom_pct}%
          </strong></span>
        </div>
        <div class="bar-track">
          <div class="bar-fill" style="width:${bar}%;background:${data.color}44;border:1px solid ${data.color};"></div>
        </div>
        <span class="badge" style="background:${data.color}18;color:${data.color};">
          <span class="badge-dot" style="background:${data.color}"></span>
          ${data.sentiment}
        </span>
        <div class="action">${data.action}</div>
      `;
      cards.appendChild(card);
      requestAnimationFrame(() => card.classList.add('visible'));
    }
    if (msg.type === 'analysis_complete') {
      dot.classList.remove('live');
      status.textContent = 'Analysis complete';
      const allCards = [...cards.querySelectorAll('.card')];
      allCards.sort((a, b) =>
        parseFloat(b.querySelector('.final-score').textContent) -
        parseFloat(a.querySelector('.final-score').textContent)
      );
      allCards.forEach(c => cards.appendChild(c));
    }
    if (msg.type === 'error') {
      dot.style.background = 'var(--bear)';
      status.textContent = `Error: ${msg.message}`;
    }
  };

  ws.onerror = () => {
    dot.style.background = 'var(--bear)';
    status.textContent = 'Connection error';
  };

  ws.onclose = () => {
    dot.classList.remove('live');
    if (status.textContent !== 'Analysis complete') {
      status.textContent = 'Disconnected — refresh to reconnect';
    }
  };
</script>
</body>
</html>
"""

# ── HTTP route ────────────────────────────────────────────────────────────────
@app.get("/")
async def index(auth_token: str = Cookie(None)):
    response = HTMLResponse(content=HTML)
    if not auth_token or not _verify_jwt(auth_token):
        token = _issue_jwt()
        response.set_cookie(
            "auth_token", token,
            httponly=True, samesite="lax",
            max_age=JWT_EXPIRY
        )
    return response

# ── WebSocket endpoint ────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    # Auth check
    token = ws.cookies.get("auth_token")
    if not token or not _verify_jwt(token):
        await ws.send_json({"type": "error", "message": "Session expired. Please refresh."})
        await ws.close()
        return

    try:
        msg = await ws.receive_text()
        if msg != "start_analysis":
            return
    except Exception:
        return

    # Per-user cache check
    ckey       = _cache_key(token)
    user_cache = _caches.get(ckey, {"data": None, "ts": 0})
    if user_cache["data"] and (time.time() - user_cache["ts"]) < CACHE_TTL:
        print(f"Serving from cache [{ckey}]")
        for row in user_cache["data"]:
            await ws.send_json({"type": "ticker_result", "data": row})
            await asyncio.sleep(0.1)
        await ws.send_json({"type": "analysis_complete"})
        return
    # ── Total timer ───────────────────────────────────────────────────────────
    t_total = time.time()

    # Login
    try:
        t0 = time.time()
        await login_to_robinhood()
        print(f"[TIMER] login:           {time.time()-t0:.2f}s")
    except Exception as e:
        await ws.send_json({"type": "error", "message": str(e)})
        return

    # Fetch holdings
    t0       = time.time()
    loop     = asyncio.get_event_loop()
    holdings = await loop.run_in_executor(None, rh.account.build_holdings)
    print(f"[TIMER] build_holdings:  {time.time()-t0:.2f}s  ({len(holdings)} tickers)")
    if not holdings:
        await ws.send_json({"type": "error", "message": "No holdings found."})
        return

    tickers = list(holdings.keys())

    # ── Step 1: Collect all news in parallel ─────────────────────────────────
    t0 = time.time()
    await ws.send_json({"type": "status", "message": "Fetching news..."})
    news_by_ticker = await loop.run_in_executor(
        _executor, collect_news, tickers
    )
    total_items = sum(len(v) for v in news_by_ticker.values())
    print(f"[TIMER] collect_news:    {time.time()-t0:.2f}s  ({total_items} items across {len(tickers)} tickers)")

    # ── Step 2: Deduplicate headlines ─────────────────────────────────────────
    t0 = time.time()
    unique_headlines = list({
        title
        for items in news_by_ticker.values()
        for item in items
        for title in [
            item.get("title")
            or item.get("content", {}).get("title")
            or ""
        ]
        if title
    })
    print(f"[TIMER] deduplicate:     {time.time()-t0:.2f}s  ({total_items} → {len(unique_headlines)} unique headlines)")
    print(f"Scoring {len(unique_headlines)} unique headlines in one batch...")

    # ── Step 3: One HuggingFace batch call ────────────────────────────────────
    t0 = time.time()
    await ws.send_json({"type": "status", "message": f"Scoring {len(unique_headlines)} headlines..."})
    headline_scores = await loop.run_in_executor(
        _executor, batch_score_finbert, unique_headlines
    )
    print(f"[TIMER] batch_finbert:   {time.time()-t0:.2f}s  ({len(headline_scores)} headlines scored)")

    # ── Step 4: Analyze each ticker (momentum + score) in parallel ────────────
    def analyze_ticker(ticker):
        t_sent = time.time()
        sent_score = compute_ticker_score(
            news_by_ticker.get(ticker, []),
            headline_scores
        )
        t_mom = time.time()
        mom   = get_price_momentum(ticker)
        t_end = time.time()
        print(f"[TIMER]   {ticker}: sentiment={t_mom-t_sent:.2f}s momentum={t_end-t_mom:.2f}s total={t_end-t_sent:.2f}s")

        final = combined_signal(sent_score, mom["score"])

        if final >= 0.15:
            sentiment, action, color, tier = "Bullish",  "HODL / Accumulate",       "#22c55e", "bull"
        elif final <= -0.15:
            sentiment, action, color, tier = "Bearish",  "Review Sell / Stop-Loss", "#ef4444", "bear"
        else:
            sentiment, action, color, tier = "Neutral",  "Maintain Position",       "#f59e0b", "neutral"

        return {
            "ticker":      ticker,
            "sentiment":   sentiment,
            "action":      action,
            "color":       color,
            "tier":        tier,
            "sent_score":  sent_score,
            "mom_pct":     mom["pct"],
            "final_score": final,
        }

    # Run all ticker analyses concurrently
    t0    = time.time()
    tasks = [
        loop.run_in_executor(_executor, analyze_ticker, ticker)
        for ticker in tickers
    ]

    results = []
    for coro in asyncio.as_completed(tasks):
        try:
            row = await coro
            results.append(row)
            await ws.send_json({"type": "ticker_result", "data": row})
            await asyncio.sleep(0.1)
        except Exception as e:
            print(f"  Error: {e}")
    print(f"[TIMER] all_tickers:     {time.time()-t0:.2f}s")
    # ── Streaming delay ───────────────────────────────────────────────────────
    t_stream = len(results) * 0.1
    print(f"[TIMER] streaming delay: ~{t_stream:.1f}s  ({len(results)} cards × 0.1s)")

    # ── Summary ───────────────────────────────────────────────────────────────
    t_elapsed = time.time() - t_total
    print(f"[TIMER] ─────────────────────────────────")
    print(f"[TIMER] TOTAL:           {t_elapsed:.2f}s")

    # Cache and complete
    _caches[ckey] = {"data": results, "ts": time.time()}
    await ws.send_json({"type": "analysis_complete"})


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8082))

    if sys.platform == "darwin":
        import selectors

        async def _serve():
            config = uvicorn.Config(app, host="0.0.0.0", port=port)
            server = uvicorn.Server(config)
            await server.serve()

        _loop = asyncio.SelectorEventLoop(selectors.SelectSelector())
        asyncio.set_event_loop(_loop)
        try:
            _loop.run_until_complete(_serve())
        finally:
            _loop.close()
    else:
        uvicorn.run(app, host="0.0.0.0", port=port)