# Docker EC2 Optimization Notes

## The biggest win — CPU-only PyTorch

```dockerfile
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu
```

This single line makes the most difference on EC2 because it affects both **image size** and **runtime RAM**.

The default PyTorch build includes CUDA (GPU support). On EC2 t3.small you have no GPU — so CUDA is dead weight.

| | Full PyTorch | CPU-only PyTorch |
|---|---|---|
| Download size | ~2.5 GB | ~700 MB |
| RAM when running | ~2.5 GB | ~700 MB |
| Savings | — | ~1.8 GB |

## Why RAM matters more than image size on EC2

- FinBERT loads into RAM on first request: ~440 MB
- PyTorch CPU-only in RAM: ~700 MB
- Total: ~1.14 GB

This means a **t3.small (2 GB RAM)** is enough instead of a t3.medium (4 GB RAM) — roughly **half the monthly cost**.

## Multi-stage build — good practice, modest impact

The idea: use one stage to build/install everything, then copy only the final artifacts into a clean image. Build tools never make it into the image you ship.

```dockerfile
# ── Stage 1: install deps ──────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: lean runtime image ───────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY --from=builder /install /usr/local
COPY . .

EXPOSE 5000
CMD ["gunicorn", "--workers=1", "--threads=2", "--timeout=120", "--bind=0.0.0.0:5000", "app:app"]
```

### What each stage does

**Stage 1 (builder):**
- Installs `gcc` and build tools needed to compile some Python packages
- Installs all Python dependencies into `/install`
- This stage is thrown away after the build — never shipped

**Stage 2 (runtime):**
- Starts from a clean `python:3.11-slim` with no build tools
- Copies only the installed packages from stage 1
- Copies your app code
- This is the final image that gets pushed to ECR and pulled by EC2

### Key flag — `--prefix=/install`
Tells pip to install packages into `/install` instead of the system Python path.
This makes it easy to copy just the packages into the runtime stage with `COPY --from=builder /install /usr/local`.

### Also add a `.dockerignore`
Stops unnecessary files being copied into the image:

```
.git
.env
__pycache__
*.pyc
*.pyo
.venv
venv
*.md
.pytest_cache
```

### Impact on EC2
- Saves ~200–400 MB from the final image (gcc + build artifacts removed)
- Slightly faster image pulls from ECR
- Lower ECR storage cost
- Not as dramatic as the CPU-only PyTorch change, but good practice

## Full memory savings summary

| Optimization | RAM / Size Saved |
|---|---|
| CPU-only PyTorch | ~1,800 MB |
| 1 gunicorn worker (FinBERT loads once) | ~440 MB |
| Multi-stage build | ~200–400 MB |
| python:3.11-slim base image | ~870 MB (image only) |
| **Total** | **~3,100+ MB** |

## EC2 instance recommendation

**t3.small (2 GB RAM)** — minimum viable, has headroom after optimizations.
t2.micro (1 GB) will OOM when FinBERT loads. Avoid.