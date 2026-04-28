# Robinhood Portfolio Sentiment Analyzer

Analyzes sentiment of your Robinhood holdings using Yahoo Finance news and VADER NLP.

## Stack
Python · Flask · Docker · AWS EC2 · Nginx · GitHub Actions

## Run locally
```bash
cp .env.example .env
# Fill in your credentials in .env
docker build -t sentiment-app .
docker run -p 5000:5000 --env-file .env sentiment-app

All E3 changes done. Summary:

  - json + redis imports added
  - _caches, _logged_in, _login_lock globals removed — Redis handles all of it
  - login_to_robinhood() — SETNX atomic lock, polls if another worker holds it, auto-expires in 30s if server crashes
  - Cache check — _redis.get(f"cache:{ckey}") instead of dict lookup
  - Cache write — _redis.setex(ckey, 1800, json.dumps(results))
  - Startup pre-warm — logs into Robinhood + caches holdings on boot
  - requirements.txt — added redis
  - docker-compose.yml — Redis Alpine sidecar, no persistence, internal network only

  To run locally you'll need Redis: brew install redis && brew services start redis, then python app.py.