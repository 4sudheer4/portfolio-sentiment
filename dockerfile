FROM python:3.11-slim

WORKDIR /app

# Minimal system deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# ⚡ CPU-only PyTorch — biggest memory saving (~1.8 GB less than full build)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# Install everything else
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

# 1 worker = FinBERT loads once, not once per worker
# 2 threads = handles concurrent requests without extra RAM
# timeout=120 = gives FinBERT time to load on cold start
CMD ["gunicorn", "--workers=1", "--threads=2", "--timeout=120", "--bind=0.0.0.0:5000", "app:app"]