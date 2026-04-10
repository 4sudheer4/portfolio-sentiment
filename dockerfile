FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 5000

CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--timeout", "120", \
     "--workers", "1", \
     "--worker-class", "gthread", \
     "--threads", "4", \
     "--keep-alive", "5", \
     "app:app"]