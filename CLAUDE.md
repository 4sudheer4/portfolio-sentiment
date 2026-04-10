# Portfolio Sentiment Analyzer

## What this is
Flask sentiment analysis app for Robinhood portfolio.
Uses FinBERT via HuggingFace API + 5-day price momentum + WebSocket streaming.
Deployed on AWS EC2 via Docker + Nginx + GitHub Actions CI/CD.

## Current phase
E1 — WebSocket streaming with flask-socketio (in progress)
Next: E2 — FastAPI migration (replacing Flask + eventlet — both deprecated)

## Stack
- Backend: Python · Flask + flask-socketio (→ FastAPI in E2)
- Sentiment: HuggingFace Inference API (ProsusAI/finbert)
- Price data: yfinance
- Portfolio: robin-stocks
- Server: Gunicorn + eventlet worker (→ Uvicorn in E2)
- Container: Docker (python:3.11-slim)
- Cloud: AWS EC2 t2.micro · Ubuntu 22.04
- Proxy: Nginx (port 80 → 8080)
- CI/CD: GitHub Actions → EC2 SSH deploy

## Key decisions — do not change without asking
- HuggingFace API not local FinBERT — t2.micro only has 1GB RAM
- Port 8080 external, 5000 internal (Gunicorn binds 5000)
- Sequential socketio loop not ThreadPoolExecutor — eventlet conflict on t2.micro
- yfinance cache at /tmp/yfinance — prevents SQLite lock with threading
- Cache TTL 30 minutes in-memory (→ Redis in E3)
- PYTHONUNBUFFERED=1 in Dockerfile — shows print logs in docker logs

## Architecture
Phone → Nginx (80) → Docker (8080) → Gunicorn → Flask/SocketIO
                                                       │
                                    ┌──────────────────┤
                                    ├── Robinhood API
                                    ├── yfinance
                                    └── HuggingFace API

## Credentials (never commit)
Stored in .env — RH_USERNAME, RH_PASSWORD, HF_TOKEN
On EC2: /home/ubuntu/portfolio-sentiment/.env

## How to run locally
docker build -t sentiment-app .
docker run -p 8080:5000 --env-file .env --name sentiment-app sentiment-app
Open: http://localhost:8080

## How to deploy
git push → GitHub Actions auto-deploys to EC2
Manual: ssh -i ~/.ssh/sentiment-key.pem ubuntu@3.88.63.58

## Enhancement roadmap
E1 → WebSockets (in progress)
E2 → FastAPI migration
E3 → Redis caching
E4 → PostgreSQL history
E5 → GraphQL (Strawberry)
E6 → React + TypeScript
     └── Playwright MCP (E2E testing)
     └── Figma MCP (design to code)
     └── Feature-dev plugin
T4 → Multi-agent parallel runs (post E6)
L1 → TypeScript (during E6)
L2 → Rust (after E6)

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
