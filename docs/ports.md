# Port Reference

## Full request flow

```
Your Phone / Browser
        │
        │ HTTP :80
        ▼
   Nginx (EC2)
        │
        │ HTTP :8080 (proxy_pass)
        ▼
  Docker Container
        │
        │ binds :5000 internally
        ▼
  Uvicorn (app:app)
        │
        ├── GET  /        → serves HTML page
        └── WS   /ws      → WebSocket analysis stream
```

## Port breakdown

| Port | Where | Bound by | Purpose |
|------|-------|----------|---------|
| 80   | EC2 host | Nginx | Public HTTP — what your phone hits |
| 8080 | EC2 host | Docker | External Docker port (Nginx proxies here) |
| 5000 | Inside container | Uvicorn | Internal app port (never exposed publicly) |
| 6379 | Inside container (E3+) | Redis | Cache + login lock (internal only) |

## Local development ports

| Port | Purpose |
|------|---------|
| 8080 | Docker Desktop local run (`-p 8080:5000`) |
| 8082 | `python app.py` local run (current default) |

## Why 8080 external and 5000 internal?

- **5000** is the standard Flask/Uvicorn default — kept internally for consistency
- **8080** is the external Docker port — avoids conflict with other services on :80 or :443
- **Nginx** sits in front and translates :80 → :8080, so the public URL stays clean (`http://3.88.63.58` not `http://3.88.63.58:8080`)

## WebSocket path

```
Browser opens: ws://3.88.63.58/ws
    → Nginx upgrades HTTP → WS, proxies to :8080
    → Docker maps :8080 → :5000
    → Uvicorn handles /ws endpoint
    → FastAPI WebSocket handler streams ticker results
```

## Nginx config (relevant excerpt)

```nginx
location / {
    proxy_pass         http://localhost:8080;
    proxy_http_version 1.1;
    proxy_set_header   Upgrade $http_upgrade;
    proxy_set_header   Connection "upgrade";  # required for WebSocket
    proxy_set_header   Host $host;
}
```

`Upgrade` and `Connection` headers are required — without them Nginx closes
the WebSocket handshake and the browser falls back to polling.

## E3 — Redis port (coming next)

| Port | Where | Purpose |
|------|-------|---------|
| 6379 | Docker internal network | Redis cache + SETNX login lock |

Redis will run as a sidecar container via docker-compose.
It is **not** exposed on the host — only the app container can reach it
via the internal Docker network (`redis:6379`).
