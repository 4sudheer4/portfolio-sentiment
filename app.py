import robin_stocks.robinhood as rh
import pandas as pd
import yfinance as yf
import nltk
import os
from flask import Flask, render_template_string
from dotenv import load_dotenv
from nltk.sentiment.vader import SentimentIntensityAnalyzer

load_dotenv()
nltk.download('vader_lexicon', quiet=True)

app = Flask(__name__)

def login_to_robinhood():
    username = os.getenv("RH_USERNAME")
    password = os.getenv("RH_PASSWORD")

    if not username or not password:
        raise ValueError(
            "Missing credentials. Create a .env file with:\n"
            "  RH_USERNAME=your_email\n"
            "  RH_PASSWORD=your_password"
        )
    rh.login(username, password, store_session=True)

def get_sentiment_score(ticker):
    analyzer = SentimentIntensityAnalyzer()
    scores = []
    try:
        stock = yf.Ticker(ticker)
        news_items = stock.news
        if not news_items:
            return 0.0
        for item in news_items[:10]:
            title = (
                item.get("title")
                or item.get("content", {}).get("title")
            )
            if not title:
                continue
            scores.append(analyzer.polarity_scores(title)['compound'])
        return round(sum(scores) / len(scores), 4) if scores else 0.0
    except Exception as e:
        print(f"  Error analyzing {ticker}: {e}")
        return 0.0

@app.route("/")
def report():
    try:
        login_to_robinhood()
    except ValueError as e:
        return f"<pre style='color:red'>{e}</pre>", 500

    holdings = rh.account.build_holdings()
    if not holdings:
        return "<p>No holdings found or session expired.</p>", 404

    rows = []
    for ticker in holdings.keys():
        print(f"Analyzing {ticker}...")
        score = get_sentiment_score(ticker)

        if score >= 0.05:
            sentiment, action, color = "Bullish", "HODL / Accumulate", "#16a34a"
        elif score <= -0.05:
            sentiment, action, color = "Bearish", "Review Sell / Stop-Loss", "#dc2626"
        else:
            sentiment, action, color = "Neutral", "Maintain Position", "#d97706"

        rows.append({
            "ticker": ticker,
            "score": score,
            "sentiment": sentiment,
            "action": action,
            "color": color
        })

    rows.sort(key=lambda x: x["score"], reverse=True)

    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Portfolio Sentiment</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, sans-serif;
                background: #0f172a;
                color: #f1f5f9;
                padding: 20px;
                max-width: 480px;
                margin: 0 auto;
            }
            h1 {
                font-size: 1.2rem;
                font-weight: 600;
                margin-bottom: 6px;
                color: #f8fafc;
            }
            .subtitle {
                font-size: 0.8rem;
                color: #64748b;
                margin-bottom: 20px;
            }
            .card {
                background: #1e293b;
                border-radius: 12px;
                padding: 16px;
                margin-bottom: 12px;
                border: 1px solid #334155;
            }
            .card-top {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 8px;
            }
            .ticker {
                font-size: 1.3rem;
                font-weight: 700;
                letter-spacing: 0.5px;
            }
            .score {
                font-size: 0.85rem;
                color: #94a3b8;
            }
            .badge {
                display: inline-block;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 0.78rem;
                font-weight: 600;
                margin-bottom: 6px;
            }
            .action {
                font-size: 0.82rem;
                color: #94a3b8;
            }
            .footer {
                text-align: center;
                font-size: 0.75rem;
                color: #475569;
                margin-top: 24px;
            }
        </style>
    </head>
    <body>
        <h1>📊 Portfolio Sentiment</h1>
        <p class="subtitle">Based on latest Yahoo Finance headlines · VADER NLP</p>

        {% for row in rows %}
        <div class="card">
            <div class="card-top">
                <span class="ticker">{{ row.ticker }}</span>
                <span class="score">Score: {{ row.score }}</span>
            </div>
            <span class="badge" style="background: {{ row.color }}22; color: {{ row.color }};">
                {{ row.sentiment }}
            </span>
            <div class="action">{{ row.action }}</div>
        </div>
        {% endfor %}

        <p class="footer">Refresh the page to pull latest scores</p>
    </body>
    </html>
    """
    return render_template_string(html, rows=rows)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)