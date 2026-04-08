FROM python:3.10-slim

WORKDIR /app

# Install system deps cleanly
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch FIRST (before other requirements)
# This prevents pip from pulling the 2.5GB CUDA version
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install transformers and other deps
COPY requirements.txt .

# Make sure requirements.txt does NOT have torch in it
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

EXPOSE 5000

CMD ["gunicorn", "--workers", "1", "--threads", "2", "--timeout", "120", "--bind", "0.0.0.0:5000", "app:app"]