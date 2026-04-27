# Portfolio Sentiment Analyzer

## What this is
Flask-based real-time sentiment analysis app for Robinhood portfolio.
Uses FinBERT via HuggingFace Inference API + 5-day price momentum.
WebSocket streaming — cards appear one by one as analysis completes.
Deployed on AWS EC2 via Docker + Nginx + GitHub Actions CI/CD.

## Current phase
E1 — WebSockets (in progress)
  ✓ E1a — WebSocket streaming (cards stream one by one)
  → E1b — sid targeting (fix duplicate cards across clients)
  → E1c — JWT auth + per-user cache
  → E1d — Login semaphore
Next: E2 — FastAPI migration

## Stack
- Backend:     Python · Flask + flask-socketio
- Async:       eventlet (Linux/EC2) · threading (Mac)
- Concurrency: GreenPool on EC2 · sequential on Mac
- Sentiment:   HuggingFace Inference API (ProsusAI/finbert)
- Price data:  yfinance (5-day momentum)
- Portfolio:   robin-stocks (Robinhood API)
- Server:      Gunicorn + eventlet worker
- Container:   Docker (python:3.11-slim)
- Cloud:       AWS EC2 t2.micro · Ubuntu 22.04
- Proxy:       Nginx (port 80 → 8080)
- CI/CD:       GitHub Actions → EC2 SSH deploy

## Architecture
Phone → Nginx (80) → Docker (8080) → Gunicorn → Flask/SocketIO
                                                       │
                                    ┌──────────────────┤
                                    ├── Robinhood API
                                    ├── yfinance
                                    └── HuggingFace API (FinBERT)

## Key decisions — do not change without asking
- HuggingFace API not local FinBERT — t2.micro only has 1GB RAM
- GreenPool not ThreadPoolExecutor — eventlet cooperative threads,
  ThreadPoolExecutor bypasses eventlet and buffers all emits
- to=sid on ALL socketio.emit() calls — prevents duplicate cards
  when multiple clients connect simultaneously
- JWT token hash as cache key — stable across reconnects
  (request.sid is ephemeral, dies on disconnect)
- Sequential emit with socketio.sleep(0.3) — yields to event loop
  so each card streams immediately instead of buffering
- yfinance cache at /tmp/yfinance — prevents SQLite lock with threads
- Platform detection at top of app.py — eventlet on Linux, threading on Mac
- eventlet.monkey_patch() only on Linux — kqueue bug on macOS
- Port 8080 external, 5000 internal (Gunicorn binds 5000)
- PYTHONUNBUFFERED=1 in Dockerfile — shows print logs in docker logs
- Cache TTL 30 minutes in-memory (→ Redis in E3)

## Current app.py structure
- Platform detection + eventlet.monkey_patch() (top)
- Imports + load_dotenv()
- Flask + SocketIO init
- HF_API_URL, HF_TOKEN globals
- _cache, _logged_in, CACHE_TTL globals
- login_to_robinhood() — skips if _logged_in
- query_finbert(text) — calls HuggingFace API
- get_finbert_sentiment(ticker) — fetches news, scores headlines
- get_price_momentum(ticker) — yfinance 5d history, retry on lock
- combined_signal(sentiment, momentum) — 55/45 weighted blend
- HTML template (module level string)
- @app.route("/") → index()
- @socketio.on("start_analysis") → handle_analysis()
- __main__ → socketio.run()

## Credentials (never commit)
Stored in .env — RH_USERNAME, RH_PASSWORD, HF_TOKEN
On EC2: /home/ubuntu/portfolio-sentiment/.env
Volume mount for session: /home/ubuntu/rh_session:/app/rh_session

## How to run locally
docker build -t sentiment-app .
docker run -p 8080:5000 --env-file .env --name sentiment-app sentiment-app
Open: http://localhost:8080

## How to deploy
git push → GitHub Actions auto-deploys to EC2
Manual SSH: ssh -i ~/.ssh/sentiment-key.pem ubuntu@3.88.63.58

## EC2 details
IP: 3.88.63.58
User: ubuntu
Key: ~/.ssh/sentiment-key.pem
Docker port mapping: 8080:5000

## Common commands
# Local Docker
docker stop sentiment-app && docker rm sentiment-app
docker build -t sentiment-app .
docker run -d -p 8080:5000 --env-file .env --name sentiment-app --restart always sentiment-app
docker logs -f sentiment-app

# EC2
ssh -i ~/.ssh/sentiment-key.pem ubuntu@3.88.63.58
sudo systemctl restart nginx
sudo nginx -t
sudo cat /var/log/nginx/error.log

# Health check
curl -s -o /dev/null -w "%{http_code}" http://3.88.63.58

## Enhancement roadmap

### E1 — WebSockets (in progress)
- E1a ✓ WebSocket streaming — cards stream one by one via socketio.emit
- E1b → sid targeting — request.sid + to=sid on all emits
          fixes duplicate cards when multiple clients connect
- E1c → JWT auth + per-user cache
          pyjwt · httpOnly cookie · 15-min expiry
          _cache keyed by hashlib.md5(token)[:8]
          each browser runs independent analysis
- E1d → Login semaphore
          eventlet.semaphore.Semaphore(1)
          prevents race condition on concurrent logins
          replaced by Redis SETNX in E3

### E2 — FastAPI migration (~3 days)
- Remove: Flask, flask-socketio, eventlet, gunicorn+eventlet
- Add: FastAPI, Uvicorn, native WebSocket, async/await, Pydantic
- JWT stays — cleaner with FastAPI dependency injection
- Auto Swagger docs at /docs
- No more semaphore needed — asyncio handles concurrency natively

### E3 — Redis (~2 days)
- Replace _cache dict with Redis (persistent, survives restarts)
- Replace login semaphore with Redis SETNX (atomic, works across workers)
- Per-user cache: Redis key = "cache:{token_key}", TTL = 1800s
- Multiple Gunicorn workers now safe
- Redis pub/sub for future multi-client broadcasts

### E4 — PostgreSQL (~4 days)
- Store every analysis run permanently
- Schema: ticker, score, sentiment, momentum, timestamp
- Track score history per ticker over time
- SQLAlchemy async ORM + Alembic migrations
- Foundation for GraphQL queries

### E5 — GraphQL — Strawberry (~4 days)
- Replace REST /report endpoint with /graphql
- Query current portfolio, filter by sentiment
- Query score history for a ticker, compare over time
- GraphQL Playground at /graphql

### E6 — React + TypeScript (~1 week)
- Replace Jinja HTML template with React app
- TypeScript from day one
- Apollo Client for GraphQL queries
- WebSocket subscriptions for live updates
- Recharts for score history charts
- Playwright MCP — E2E browser testing
- Figma MCP — design to code workflow
- Feature-dev plugin — structured feature development

### Post E6
- T4: Multi-agent parallel runs for large refactors
- L1: TypeScript (during E6)
- L2: Rust (after E6)

## Resume line (after E6 complete)
Tech: Python · FastAPI · WebSockets · GraphQL · Strawberry ·
      PostgreSQL · Redis · React · TypeScript · Docker ·
      AWS EC2 · Nginx · GitHub Actions · JWT · Playwright

## Interview talking points
- GreenPool vs ThreadPoolExecutor: cooperative vs preemptive threads
- to=sid vs broadcast: transport-layer vs application-layer targeting
- JWT token key vs sid: stable vs ephemeral client identity
- Redis SETNX vs semaphore: infrastructure vs application coordination
- Monolith vs microservices: why monolith is right for this scale
- Nginx as reverse proxy: not load balancer (single backend)
- HuggingFace API vs local model: RAM constraint tradeoff on t2.micro
- eventlet monkey_patch: why Linux only (kqueue bug on macOS)
-------------
updated after e2:
# Portfolio Sentiment Analyzer

## What this is
FastAPI real-time sentiment analysis app for Robinhood portfolio.
Uses FinBERT via HuggingFace Inference API + 5-day price momentum.
WebSocket streaming — cards appear one by one as analysis completes.
Deployed on AWS EC2 via Docker + Nginx + GitHub Actions CI/CD.

## Current phase
E3 — Redis (in progress)
  → Replace _caches dict with Redis cache
  → Replace asyncio.Lock with Redis SETNX
  → Replace _logged_in flag with Redis key
  → Add holdings cache in Redis
  → Add startup pre-warm

Previously completed:
  ✓ E1a — WebSocket streaming
  ✓ E1b — sid targeting (native WebSocket — no sid needed, ws object per connection)
  ✓ E1c — JWT auth + per-user cache keyed by md5(token)[:8]
  ✓ E1d — Login semaphore (asyncio.Lock — being replaced by Redis SETNX in E3)
  ✓ E2  — FastAPI migration (native WebSocket, Uvicorn, asyncio, batch HF API)

## Stack
- Backend:     Python · FastAPI
- Server:      Uvicorn
- Async:       asyncio + ThreadPoolExecutor(max_workers=20)
- Sentiment:   HuggingFace Inference API (ProsusAI/finbert) — batch call
- Price data:  yfinance (5-day momentum) — parallel fetch
- News:        yfinance.news — parallel fetch via ThreadPoolExecutor
- Portfolio:   robin-stocks (Robinhood API)
- Cache:       Redis (E3) — replacing in-memory dict
- Container:   Docker (python:3.11-slim)
- Cloud:       AWS EC2 t2.micro · Ubuntu 22.04
- Proxy:       Nginx (port 80 → 8082)
- CI/CD:       GitHub Actions → EC2 SSH deploy

## Architecture
Phone → Nginx (80) → Docker (8082) → Uvicorn → FastAPI
                                                    │
                                    ┌───────────────┤
                                    ├── Redis        │ ← E3
                                    ├── Robinhood API
                                    ├── yfinance
                                    └── HuggingFace API

## E3 — Redis implementation plan

### What Redis replaces
_caches = {}          → Redis key: cache:{token_key}   TTL=1800s
_logged_in = False    → Redis key: rh:logged_in        TTL=3600s
asyncio.Lock          → Redis SETNX: rh:login_lock     TTL=30s
(future) holdings     → Redis key: rh:holdings         TTL=600s

### Redis keys used
cache:{token_key}   string   1800s   per-user analysis results (JSON)
rh:logged_in        string   3600s   login state across workers
rh:login_lock       string   30s     SETNX atomic login lock
rh:holdings         string   600s    holdings cache (optional E3 addition)

### Redis client setup
import redis
import json

_redis = redis.from_url(
    os.getenv("REDIS_URL", "redis://localhost:6379"),
    decode_responses=True
)

CACHE_TTL  = 1800
LOGIN_TTL  = 3600
LOCK_TTL   = 30

### Change 1 — Remove these globals
# DELETE these three lines:
_caches    = {}
_logged_in = False
_login_lock = None

### Change 2 — login_to_robinhood() new implementation
async def login_to_robinhood():
    # Fast path
    if _redis.get("rh:logged_in"):
        return

    # SETNX atomic lock — only one worker wins
    acquired = _redis.set("rh:login_lock", "1", nx=True, ex=LOCK_TTL)

    if not acquired:
        # Another worker is logging in — poll until done
        for _ in range(15):
            await asyncio.sleep(1)
            if _redis.get("rh:logged_in"):
                return
        raise Exception("Login lock timeout")

    # Double-check after acquiring (another worker may have finished)
    if _redis.get("rh:logged_in"):
        _redis.delete("rh:login_lock")
        return

    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _do_rh_login)
        _redis.setex("rh:logged_in", LOGIN_TTL, "1")
        print("Logged into Robinhood.")
    finally:
        _redis.delete("rh:login_lock")  # always release

### Change 3 — Cache check in websocket_endpoint
# Find this block and replace:
# BEFORE:
ckey       = _cache_key(token)
user_cache = _caches.get(ckey, {"data": None, "ts": 0})
if user_cache["data"] and (time.time() - user_cache["ts"]) < CACHE_TTL:
    print(f"Serving from cache [{ckey}]")
    for row in user_cache["data"]:
        await ws.send_json({"type": "ticker_result", "data": row})
        await asyncio.sleep(0.1)
    await ws.send_json({"type": "analysis_complete"})
    return

# AFTER:
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

### Change 4 — Cache write at end of websocket_endpoint
# Find this line and replace:
# BEFORE:
_caches[ckey] = {"data": results, "ts": time.time()}

# AFTER:
_redis.setex(ckey, CACHE_TTL, json.dumps(results))

### Change 5 — Startup pre-warm (add after HTML template)
@app.on_event("startup")
async def startup():
    try:
        loop = asyncio.get_event_loop()
        await login_to_robinhood()
        holdings = await loop.run_in_executor(None, rh.account.build_holdings)
        _redis.setex("rh:holdings", 600, json.dumps(list(holdings.keys())))
        print(f"Startup: pre-warmed {len(holdings)} tickers")
    except Exception as e:
        print(f"Startup warm-up failed: {e}")

## Key decisions — do not change without asking
- HuggingFace batch API — all headlines in ONE POST request
  Before: 56 individual calls → 40-80s
  After:  1 batch call → 3-5s
- collect_news() parallelized with ThreadPoolExecutor(20)
  Stores FULL news items (not just titles) — preserves timestamps
  for recency weighting: 0.5 ** (age_seconds / 259200)
- fetch holdings via rh.account.build_holdings() — slow (16s)
  mitigated by Redis holdings cache (TTL=600s)
- Port: 8082 external → 5000 internal Docker → Uvicorn binds 5000
- JWT: httpOnly cookie, 24hr expiry, HS256
- Cache key: md5(token)[:8] — stable across reconnects
- PYTHONUNBUFFERED=1 in Dockerfile
- Mac: asyncio.SelectorEventLoop(SelectSelector()) — kqueue workaround
- Linux/EC2: uvicorn.run() directly — no kqueue issue

## Current app.py structure
- Imports (asyncio, hashlib, jwt, redis, requests, rh, yf, fastapi)
- FastAPI app init
- HF_API_URL, HF_TOKEN, JWT_SECRET globals
- TTL constants (CACHE_TTL, LOGIN_TTL, LOCK_TTL)
- _redis client
- _executor = ThreadPoolExecutor(max_workers=20)
- JWT helpers (_issue_jwt, _verify_jwt, _cache_key)
- _do_rh_login() — sync, called via run_in_executor
- login_to_robinhood() — async, SETNX lock
- collect_news(tickers) — parallel ThreadPoolExecutor
- batch_score_finbert(headlines) — ONE HF API call
- compute_ticker_score(news_items, headline_scores) — recency decay
- get_price_momentum(ticker) — yfinance 5d history
- combined_signal(sentiment, momentum) — 55/45 blend
- HTML template
- @app.on_event("startup") — pre-warm
- @app.get("/") — JWT cookie
- @app.websocket("/ws") — full analysis pipeline
- __main__ — platform-aware uvicorn start

## Credentials (never commit)
.env file:
  RH_USERNAME=...
  RH_PASSWORD=...
  HF_TOKEN=hf_xxxxxxxx
  JWT_SECRET=...
  REDIS_URL=redis://localhost:6379

On EC2: /home/ubuntu/portfolio-sentiment/.env
  REDIS_URL=redis://127.0.0.1:6379

## How to run locally
redis-cli ping  # verify Redis running
python app.py
Open: http://localhost:8082

## How to deploy
git push → GitHub Actions auto-deploys to EC2
Manual SSH: ssh -i ~/.ssh/sentiment-key.pem ubuntu@3.88.63.58

## EC2 details
IP: 3.88.63.58
Docker port: -p 8082:5000
Nginx: proxy_pass http://127.0.0.1:8082
Redis: runs on EC2, accessible at 127.0.0.1:6379

## Common commands
# Local
redis-cli ping
redis-cli keys '*'           ← see all Redis keys
redis-cli get rh:logged_in   ← check login state
redis-cli ttl cache:abc123   ← check TTL remaining
redis-cli flushall           ← clear all Redis data

# Docker
docker stop sentiment-app && docker rm sentiment-app
docker build -t sentiment-app .
docker run -d -p 8082:5000 --env-file .env \
  --name sentiment-app --restart always sentiment-app
docker logs -f sentiment-app

# EC2
ssh -i ~/.ssh/sentiment-key.pem ubuntu@3.88.63.58
sudo systemctl restart nginx
sudo systemctl status redis

## Enhancement roadmap
E1a ✓  WebSocket streaming
E1b ✓  sid targeting (native WS — each ws object is unique per connection)
E1c ✓  JWT auth + per-user cache
E1d ✓  Login semaphore (asyncio.Lock → replaced by Redis SETNX in E3)
E2  ✓  FastAPI migration
E3  →  Redis (IN PROGRESS — implement changes above)
E4     PostgreSQL history
E5     GraphQL — Strawberry
E6     React + TypeScript
       └── Playwright MCP
       └── Figma MCP
       └── Feature-dev plugin
T4     Multi-agent parallel runs (post E6)
L1     TypeScript (during E6)
L2     Rust (after E6)

## Interview talking points
- Redis SETNX vs asyncio.Lock: infrastructure-level atomicity vs process-level
- SETNX ex=30: auto-expiry prevents deadlock if server crashes (self-healing)
- Double-check pattern: check → lock → check again (prevents duplicate login)
- Redis TTL: no manual time.time() comparison needed — Redis handles expiry
- Per-user cache keyed by token hash: stable across reconnects (sid is ephemeral)
- Batch HF API: one HTTP request vs 56 — 5-10x speedup
- ThreadPoolExecutor for yfinance: blocking I/O parallelized without asyncio
- recency decay: 0.5^(age/259200) — 3-day half-life weights recent news higher
- collect_news stores full items not just titles — preserves timestamps for decay
- Startup pre-warm: first user gets fast response — holdings already cached
