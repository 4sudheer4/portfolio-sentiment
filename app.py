import asyncio
import hashlib
import os
import secrets
import sys
import time
from concurrent.futures import ThreadPoolExecutor

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

# ── Cache — keyed by md5(jwt_token)[:8] ─────────────────────────────────────
_caches    = {}   # cache_key → {"data": [...], "ts": float}
_logged_in = False
CACHE_TTL  = 1800
JWT_EXPIRY = 86400  # 24 hours

# ── Thread pool for blocking I/O ─────────────────────────────────────────────
_executor = ThreadPoolExecutor(max_workers=5)

# ── Login lock — initialized lazily inside event loop ────────────────────────
_login_lock = None

# ── JWT helpers ──────────────────────────────────────────────────────────────
def _issue_jwt():
    payload = {
        "uid": secrets.token_hex(8),
        "iat": int(time.time()),
        "exp": int(time.time()) + JWT_EXPIRY,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def _verify_jwt(token):
    try:
        jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return True
    except Exception:
        return False

def _cache_key(token):
    return hashlib.md5(token.encode()).hexdigest()[:8]

# ── Robinhood login ──────────────────────────────────────────────────────────
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
        if _logged_in:  # re-check after acquiring
            return
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _do_rh_login)
        _logged_in = True
        print("Logged into Robinhood.")

# ── FinBERT via HuggingFace API ──────────────────────────────────────────────
def query_finbert(text: str) -> dict:
    print(f"  [HF] querying: {text[:60]}")
    for attempt in range(3):
        try:
            print(f"  [HF] attempt {attempt + 1} — sending request")
            response = requests.post(
                HF_API_URL,
                headers={"Authorization": f"Bearer {HF_TOKEN}"},
                json={"inputs": text},
                timeout=20
            )
            print(f"  [HF] status={response.status_code} body={response.text[:200]}")

            # Model loading — HF returns 503 or {"error": "loading"}
            if response.status_code == 503 or (
                isinstance(response.json(), dict) and "error" in response.json()
            ):
                wait = response.json().get("estimated_time", 5) if response.content else 5
                print(f"  [HF] model loading, waiting {wait}s")
                time.sleep(min(wait, 10))
                continue

            result = response.json()
            print(f"  [HF] parsed result: {result}")
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], list):
                    result = result[0]
                best = max(result, key=lambda x: x["score"])
                print(f"  [HF] best label={best['label']} score={best['score']:.4f}")
                return {"label": best["label"].lower(), "score": best["score"]}
            print(f"  [HF] unexpected result shape, returning neutral")
        except Exception as e:
            print(f"  [HF] exception (attempt {attempt + 1}): {type(e).__name__}: {e}")
            time.sleep(2)
    print(f"  [HF] all attempts failed, returning neutral")
    return {"label": "neutral", "score": 1.0}

# ── FinBERT sentiment ────────────────────────────────────────────────────────
def get_finbert_sentiment(ticker: str) -> float:
    try:
        stock      = yf.Ticker(ticker)
        news_items = stock.news or []
        if not news_items:
            return 0.0

        now          = time.time()
        weighted_sum = 0.0
        weight_total = 0.0

        for item in news_items[:5]:
            title = (
                item.get("title")
                or item.get("content", {}).get("title")
                or ""
            )
            if not title:
                continue

            pub_time       = item.get("providerPublishTime") or now
            age_seconds    = max(now - pub_time, 0)
            recency_weight = 0.5 ** (age_seconds / 259200)

            result     = query_finbert(title)
            label      = result["label"]
            confidence = result["score"]

            score           = LABEL_TO_SCORE.get(label, 0.0) * confidence
            combined_weight = recency_weight * confidence

            weighted_sum  += score * combined_weight
            weight_total  += combined_weight

        return round(weighted_sum / weight_total, 4) if weight_total else 0.0

    except Exception as e:
        print(f"  FinBERT error for {ticker}: {e}")
        return 0.0

# ── Price momentum ───────────────────────────────────────────────────────────
def get_price_momentum(ticker: str) -> dict:
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
                time.sleep(0.5)
                continue
            print(f"  Momentum error for {ticker}: {e}")
            return {"pct": 0.0, "score": 0.0}

# ── Combined signal ──────────────────────────────────────────────────────────
SENTIMENT_WEIGHT = 0.55
MOMENTUM_WEIGHT  = 0.45

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
      background: var(--muted); flex-shrink: 0;
      transition: background 0.3s;
    }
    .status-dot.live { background: var(--bull); animation: pulse 1.5s infinite; }
    @keyframes pulse {
      0%, 100% { opacity: 1; } 50% { opacity: 0.4; }
    }
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
          <span>5d momentum <strong style="color:${data.mom_pct >= 0 ? 'var(--bull)' : 'var(--bear)'}">${data.mom_pct >= 0 ? '+' : ''}${data.mom_pct}%</strong></span>
        </div>
        <div class="bar-track">
          <div class="bar-fill" style="width:${bar}%; background:${data.color}44; border:1px solid ${data.color};"></div>
        </div>
        <span class="badge" style="background:${data.color}18; color:${data.color};">
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
      allCards.sort((a, b) => {
        const scoreA = parseFloat(a.querySelector('.final-score').textContent);
        const scoreB = parseFloat(b.querySelector('.final-score').textContent);
        return scoreB - scoreA;
      });
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
    t_start = time.time()
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

    ckey       = _cache_key(token)
    user_cache = _caches.get(ckey, {"data": None, "ts": 0})

    if user_cache["data"] and (time.time() - user_cache["ts"]) < CACHE_TTL:
        print(f"Serving from cache [{ckey}]")
        for row in user_cache["data"]:
            await ws.send_json({"type": "ticker_result", "data": row})
        await ws.send_json({"type": "analysis_complete"})
        return

    try:
        await login_to_robinhood()
    except Exception as e:
        await ws.send_json({"type": "error", "message": str(e)})
        return

    loop     = asyncio.get_event_loop()
    holdings = await loop.run_in_executor(None, rh.account.build_holdings)
    if not holdings:
        await ws.send_json({"type": "error", "message": "No holdings found."})
        return

    def analyze_ticker(ticker):
        t0 = time.time()
        sent_score = get_finbert_sentiment(ticker)
        t1 = time.time()
        mom        = get_price_momentum(ticker)
        t2 = time.time()
        print(f"{ticker}: sentiment={t1-t0:.2f}s momentum={t2-t1:.2f}s total={t2-t0:.2f}s")

        final      = combined_signal(sent_score, mom["score"])
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

    results = []
    tasks   = [
        loop.run_in_executor(_executor, analyze_ticker, ticker)
        for ticker in holdings.keys()
    ]
    for coro in asyncio.as_completed(tasks):
        try:
            row = await coro
            results.append(row)
            await ws.send_json({"type": "ticker_result", "data": row})
            await asyncio.sleep(0.1)
        except Exception as e:
            print(f"  Error: {e}")

    _caches[ckey] = {"data": results, "ts": time.time()}
    await ws.send_json({"type": "analysis_complete"})
    print(f"Total analysis time: {time.time() - t_start:.2f}s")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8082))

    if sys.platform == "darwin":
        # macOS Python 3.9: kqueue selector is buggy.
        # Bypass uvicorn's loop creation — run server.serve() in our own
        # SelectSelector-based loop so kqueue is never touched.
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