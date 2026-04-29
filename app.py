import asyncio
import hashlib
import json
import os
import secrets
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import asyncpg
import jwt
import redis
import requests
import robin_stocks.robinhood as rh
import yfinance as yf
from dotenv import load_dotenv
from fastapi import Cookie, FastAPI, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse

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

JWT_EXPIRY = 86400
CACHE_TTL  = 1800   # per-user analysis cache (s)
LOGIN_TTL  = 3600   # rh:logged_in key TTL (s)
LOCK_TTL   = 30     # rh:login_lock TTL — auto-expires if server crashes (s)

# ── Redis client ──────────────────────────────────────────────────────────────
_redis = redis.from_url(
    os.getenv("REDIS_URL", "redis://localhost:6379"),
    decode_responses=True
)

# ── DB pool ───────────────────────────────────────────────────────────────────
_db_pool = None  # asyncpg.Pool — set on startup

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
    # Fast path — already logged in
    if _redis.get("rh:logged_in"):
        return

    # SETNX — only one worker wins the lock
    acquired = _redis.set("rh:login_lock", "1", nx=True, ex=LOCK_TTL)
    if not acquired:
        # Another worker is logging in — poll until done
        for _ in range(15):
            await asyncio.sleep(1)
            if _redis.get("rh:logged_in"):
                return
        raise Exception("Login lock timeout — Robinhood login took too long")

    # Double-check after acquiring
    if _redis.get("rh:logged_in"):
        _redis.delete("rh:login_lock")
        return

    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _do_rh_login)
        _redis.setex("rh:logged_in", LOGIN_TTL, "1")
        print("Logged into Robinhood.")
    finally:
        _redis.delete("rh:login_lock")

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
    print(f"[DEBUG] News collected: {[(t, len(items)) for t, items in result.items()]}")
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
            print(f"[DEBUG] HF raw results type: {type(results)}")
            print(f"[DEBUG] First result sample: {results[0] if results else 'None'}")

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
            for headline, result in zip(headlines, results):
                if isinstance(result, dict) and 'label' in result:
                    label = result["label"].lower()
                    conf  = result["score"]
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
    print(f"[DEBUG] compute_ticker_score: {len(news_items)} news items")
    print(f"[DEBUG] headline_scores has {len(headline_scores)} entries")
    matched_count = 0
    for item in news_items:
        title = (
            item.get("title")
            or item.get("content", {}).get("title")
            or ""
        )
        print(f"[DEBUG] News title: '{title}'")
        print(f"[DEBUG] Title in scores: {title in headline_scores}")

        if not title or title not in headline_scores:
            continue
        matched_count += 1
        pub_time       = item.get("providerPublishTime") or now
        age_seconds    = max(now - pub_time, 0)
        recency_weight = 0.5 ** (age_seconds / 259200)
        score          = headline_scores[title]
        print(f"[DEBUG] Matched title score: {score}, weight: {recency_weight}")
        weighted_sum  += score * recency_weight
        weight_total  += recency_weight

    final_score = round(weighted_sum / weight_total, 4) if weight_total else 0.0
    print(f"[DEBUG] Final ticker score: {final_score} (matched {matched_count} titles)")
    return final_score

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

# ── DB helpers ────────────────────────────────────────────────────────────────
async def _init_db_pool():
    global _db_pool
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("DATABASE_URL not set — skipping PostgreSQL init")
        return
    try:
        _db_pool = await asyncpg.create_pool(db_url, min_size=1, max_size=5)
        print("PostgreSQL pool created.")
    except Exception as e:
        print(f"PostgreSQL unavailable — skipping (non-fatal): {e}")

async def store_analysis_with_history(
    token: str,
    tickers: list,
    results: list,
    duration_seconds: int,
):
    """Dual-write: Redis cache + PostgreSQL permanent history."""
    ckey = f"cache:{_cache_key(token)}"
    _redis.setex(ckey, CACHE_TTL, json.dumps(results))

    if _db_pool is None:
        return
    try:
        async with _db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO analyses
                    (user_token_hash, tickers, results, analysis_duration_seconds)
                VALUES ($1, $2, $3, $4)
                """,
                _cache_key(token),
                tickers,
                json.dumps(results),
                duration_seconds,
            )
    except Exception as e:
        print(f"[DB] store error (non-fatal): {e}")

# ── Startup / shutdown ────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    await _init_db_pool()
    _redis.delete("rh:logged_in")   # clear stale key — fresh container needs fresh login
    try:
        await login_to_robinhood()
        loop     = asyncio.get_event_loop()
        holdings = await loop.run_in_executor(None, rh.account.build_holdings)
        _redis.setex("rh:holdings", 600, json.dumps(list(holdings.keys())))
        print(f"Startup: pre-warmed {len(holdings)} tickers")
    except Exception as e:
        print(f"Startup warm-up failed (non-fatal): {e}")

@app.on_event("shutdown")
async def shutdown():
    if _db_pool:
        await _db_pool.close()
        print("PostgreSQL pool closed.")

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

# ── History API ───────────────────────────────────────────────────────────────
@app.get("/api/history")
async def api_history(auth_token: str = Cookie(None), limit: int = 10):
    """Last N analyses for the current user."""
    if not auth_token or not _verify_jwt(auth_token):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if _db_pool is None:
        return JSONResponse({"error": "Database not available"}, status_code=503)
    async with _db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, created_at, tickers, results, analysis_duration_seconds
            FROM analyses
            WHERE user_token_hash = $1
            ORDER BY created_at DESC
            LIMIT $2
            """,
            _cache_key(auth_token),
            limit,
        )
    return JSONResponse([
        {
            "id":       r["id"],
            "created_at": r["created_at"].isoformat(),
            "tickers":  r["tickers"],
            "results":  json.loads(r["results"]),
            "duration": r["analysis_duration_seconds"],
        }
        for r in rows
    ])

@app.get("/api/trends/{ticker}")
async def api_trends(ticker: str, auth_token: str = Cookie(None), limit: int = 30):
    """Score history for one ticker across past runs for the current user."""
    if not auth_token or not _verify_jwt(auth_token):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if _db_pool is None:
        return JSONResponse({"error": "Database not available"}, status_code=503)
    async with _db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT created_at,
                   elem->>'final_score'  AS final_score,
                   elem->>'sentiment'    AS sentiment
            FROM analyses,
                 jsonb_array_elements(results) AS elem
            WHERE user_token_hash = $1
              AND elem->>'ticker' = $2
            ORDER BY created_at DESC
            LIMIT $3
            """,
            _cache_key(auth_token),
            ticker.upper(),
            limit,
        )
    return JSONResponse([
        {
            "created_at":  r["created_at"].isoformat(),
            "final_score": float(r["final_score"]),
            "sentiment":   r["sentiment"],
        }
        for r in rows
    ])

@app.get("/api/stats")
async def api_stats(auth_token: str = Cookie(None)):
    """Aggregate stats for the current user."""
    if not auth_token or not _verify_jwt(auth_token):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if _db_pool is None:
        return JSONResponse({"error": "Database not available"}, status_code=503)
    async with _db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT COUNT(*)                                   AS total_runs,
                   MIN(created_at)                           AS first_run,
                   MAX(created_at)                           AS last_run,
                   ROUND(AVG(analysis_duration_seconds), 1)  AS avg_duration
            FROM analyses
            WHERE user_token_hash = $1
            """,
            _cache_key(auth_token),
        )
    return JSONResponse({
        "total_runs":   row["total_runs"],
        "first_run":    row["first_run"].isoformat() if row["first_run"] else None,
        "last_run":     row["last_run"].isoformat() if row["last_run"] else None,
        "avg_duration": float(row["avg_duration"]) if row["avg_duration"] else None,
    })

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
    ckey   = f"cache:{_cache_key(token)}"
    cached = _redis.get(ckey)
    if cached:
        print(f"[CACHE] hit [{ckey}]")
        for row in json.loads(cached):
            await ws.send_json({"type": "ticker_result", "data": row})
            await asyncio.sleep(0.1)
        await ws.send_json({"type": "analysis_complete"})
        return
    print(f"[CACHE] miss [{ckey}]")
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
        print(f"[DEBUG] Starting analysis for ticker: {ticker}")
    
        t_sent = time.time()
        news_items = news_by_ticker.get(ticker, [])
        print(f"[DEBUG] {ticker} has {len(news_items)} news items")

        sent_score = compute_ticker_score(news_items, headline_scores)
        print(f"[DEBUG] {ticker} computed sentiment score: {sent_score}")
        
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

    # Dual-write: Redis cache + PostgreSQL history
    duration = int(time.time() - t_total)
    await store_analysis_with_history(token, tickers, results, duration)
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