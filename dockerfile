# ── Base image ────────────────────────────────────────────────────────────────
# Python 3.11 slim keeps the image small while supporting all dependencies
FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
# gcc/g++ needed to compile some transformers/torch native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Pre-download FinBERT model at build time ──────────────────────────────────
# This bakes the ~440 MB model into the image so the first request is fast.
# Remove this block if you prefer to download at runtime to keep image smaller.
RUN python -c "\
from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
AutoTokenizer.from_pretrained('ProsusAI/finbert'); \
AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')"

# ── App source ────────────────────────────────────────────────────────────────
COPY app.py .
COPY .env .

# ── Port ──────────────────────────────────────────────────────────────────────
EXPOSE 8080

# ── Run ───────────────────────────────────────────────────────────────────────
CMD ["python", "app.py"]
