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
```