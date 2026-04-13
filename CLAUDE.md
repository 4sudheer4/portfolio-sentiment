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