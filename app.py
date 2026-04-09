import robin_stocks.robinhood as rh
import yfinance as yf
import os
import time
from flask import Flask, render_template_string
from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed


load_dotenv()

app = Flask(__name__)

# ── FinBERT setup ────────────────────────────────────────────────────────────
# ProsusAI/finbert is trained specifically on financial text.
# Labels: positive / negative / neutral  (maps cleanly to compound score)
_finbert_pipeline = None

def get_finbert():
    global _finbert_pipeline
    if _finbert_pipeline is None:
        print("Loading FinBERT model (first run may take ~30 s)…")
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        _finbert_pipeline = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            truncation=True,
            max_length=512,
        )
    return _finbert_pipeline

LABEL_TO_SCORE = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}

# ── Robinhood login ──────────────────────────────────────────────────────────
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

# ── FinBERT sentiment ────────────────────────────────────────────────────────
def get_finbert_sentiment(ticker: str) -> float:
    """
    Returns a blended score in [-1, 1].
    Each headline is scored by FinBERT and weighted by its confidence,
    then recency-decayed so that older news matters less.
    """
    finbert = get_finbert()
    try:
        stock = yf.Ticker(ticker)
        news_items = stock.news or []
        if not news_items:
            return 0.0

        now = time.time()
        weighted_sum = 0.0
        weight_total = 0.0

        for item in news_items[:20]:                       # up to 20 headlines
            title = (
                item.get("title")
                or item.get("content", {}).get("title")
                or ""
            )
            if not title:
                continue

            # Recency weight: half-life of ~3 days (259200 s)
            pub_time = item.get("providerPublishTime") or now
            age_seconds = max(now - pub_time, 0)
            recency_weight = 0.5 ** (age_seconds / 259200)

            result = finbert(title)[0]
            label  = result["label"].lower()
            confidence = result["score"]                   # model confidence [0,1]

            score = LABEL_TO_SCORE.get(label, 0.0) * confidence
            combined_weight = recency_weight * confidence

            weighted_sum  += score * combined_weight
            weight_total  += combined_weight

        return round(weighted_sum / weight_total, 4) if weight_total else 0.0

    except Exception as e:
        print(f"  FinBERT error for {ticker}: {e}")
        return 0.0

# ── Price momentum ───────────────────────────────────────────────────────────
def get_price_momentum(ticker: str) -> dict:
    """
    Returns 5-day % change and a normalised momentum score in [-1, 1].
    Caps at ±20 % so one outlier can't dominate the blend.
    """
    try:
        hist = yf.Ticker(ticker).history(period="5d")
        if len(hist) < 2:
            return {"pct": 0.0, "score": 0.0}
        pct = (hist["Close"].iloc[-1] - hist["Close"].iloc[0]) / hist["Close"].iloc[0]
        score = max(-1.0, min(1.0, pct / 0.20))           # normalise to [-1,1]
        return {"pct": round(pct * 100, 2), "score": round(score, 4)}
    except Exception as e:
        print(f"  Momentum error for {ticker}: {e}")
        return {"pct": 0.0, "score": 0.0}

# ── Combined signal ──────────────────────────────────────────────────────────
SENTIMENT_WEIGHT = 0.55
MOMENTUM_WEIGHT  = 0.45

def combined_signal(sentiment_score: float, momentum_score: float) -> float:
    return round(
        SENTIMENT_WEIGHT * sentiment_score + MOMENTUM_WEIGHT * momentum_score, 4
    )

# ── Flask route ──────────────────────────────────────────────────────────────
@app.route("/")
def report():
    try:
        login_to_robinhood()
    except ValueError as e:
        return f"<pre style='color:red'>{e}</pre>", 500

    holdings = rh.account.build_holdings()
    if not holdings:
        return "<p>No holdings found or session expired.</p>", 404



        
    def analyze_ticker(ticker):
        sent_score = get_finbert_sentiment(ticker)
        mom        = get_price_momentum(ticker)
        final      = combined_signal(sent_score, mom["score"])
        if final >= 0.15:
            sentiment, action, color, tier = "Bullish",  "HODL / Accumulate",       "#22c55e", "bull"
        elif final <= -0.15:
            sentiment, action, color, tier = "Bearish",  "Review Sell / Stop-Loss", "#ef4444", "bear"
        else:
            sentiment, action, color, tier = "Neutral",  "Maintain Position",       "#f59e0b", "neutral"
        return {
            "ticker":      ticker,
            "sentiment":   sentiment,
            "action":      action,
            "color":       color,
            "tier":        tier,
            "sent_score":  sent_score,
            "mom_pct":     mom["pct"],
            "final_score": final,
        }
    
    rows = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(analyze_ticker, t): t for t in holdings.keys()}
        for future in as_completed(futures):
            try:
                rows.append(future.result())
            except Exception as e:
                print(f"  Error analyzing ticker: {e}")

    rows.sort(key=lambda x: x["final_score"], reverse=True)
    

    html = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Portfolio Signal</title>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&display=swap" rel="stylesheet"/>
  <style>
    :root {
      --bg:      #08090d;
      --surface: #0f1117;
      --border:  #1e2130;
      --muted:   #4a5068;
      --text:    #e8eaf0;
      --bull:    #22c55e;
      --bear:    #ef4444;
      --neutral: #f59e0b;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: ui-monospace, 'SF Mono', monospace;
      min-height: 100vh;
      padding: 28px 16px 60px;
    }

    /* ── header ── */
    header {
      max-width: 480px;
      margin: 0 auto 28px;
    }
    header h1 {
      font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif;
      font-size: 1.6rem;
      font-weight: 800;
      letter-spacing: -0.5px;
      line-height: 1.1;
    }
    header h1 span { color: var(--bull); }
    .meta {
      margin-top: 6px;
      font-size: 0.7rem;
      color: var(--muted);
      letter-spacing: 0.04em;
    }

    /* ── legend ── */
    .legend {
      max-width: 480px;
      margin: 0 auto 20px;
      display: flex;
      gap: 16px;
      font-size: 0.68rem;
      color: var(--muted);
    }
    .legend-dot {
      display: inline-block;
      width: 7px; height: 7px;
      border-radius: 50%;
      margin-right: 5px;
      vertical-align: middle;
    }

    /* ── card grid ── */
    .cards { max-width: 480px; margin: 0 auto; }

    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 16px 18px;
      margin-bottom: 10px;
      position: relative;
      overflow: hidden;
      transition: border-color 0.2s;
    }
    .card:hover { border-color: var(--muted); }

    /* coloured left accent bar */
    .card::before {
      content: '';
      position: absolute;
      left: 0; top: 0; bottom: 0;
      width: 3px;
      border-radius: 14px 0 0 14px;
    }
    .card.bull::before  { background: var(--bull); }
    .card.bear::before  { background: var(--bear); }
    .card.neutral::before { background: var(--neutral); }

    .card-row {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
    }
    .ticker {
      font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif;
      font-size: 1.25rem;
      font-weight: 800;
      letter-spacing: 0.5px;
    }
    .final-score {
      font-size: 1.1rem;
      font-weight: 500;
    }

    .sub-row {
      display: flex;
      gap: 14px;
      margin-top: 10px;
      font-size: 0.7rem;
      color: var(--muted);
    }
    .sub-item strong { color: var(--text); }

    .badge {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      margin-top: 10px;
      padding: 4px 10px;
      border-radius: 20px;
      font-size: 0.72rem;
      font-weight: 500;
      letter-spacing: 0.03em;
    }
    .badge-dot {
      width: 6px; height: 6px;
      border-radius: 50%;
      flex-shrink: 0;
    }

    .action {
      margin-top: 6px;
      font-size: 0.72rem;
      color: var(--muted);
    }

    /* ── bar chart mini ── */
    .bar-track {
      margin-top: 12px;
      height: 4px;
      background: var(--border);
      border-radius: 99px;
      overflow: hidden;
    }
    .bar-fill {
      height: 100%;
      border-radius: 99px;
      transition: width 0.4s ease;
    }

    footer {
      text-align: center;
      font-size: 0.68rem;
      color: var(--muted);
      margin-top: 32px;
    }
    footer a { color: var(--muted); text-decoration: underline; }
  </style>
</head>
<body>

<header>
  <h1>Portfolio <span>Signal</span></h1>
  <p class="meta">FinBERT NLP · 5-day momentum · blended 55/45</p>
</header>

<div class="legend">
  <span><span class="legend-dot" style="background:var(--bull)"></span>Bullish ≥ 0.15</span>
  <span><span class="legend-dot" style="background:var(--neutral)"></span>Neutral</span>
  <span><span class="legend-dot" style="background:var(--bear)"></span>Bearish ≤ -0.15</span>
</div>

<div class="cards">
{% for r in rows %}
<div class="card {{ r.tier }}">
  <div class="card-row">
    <span class="ticker">{{ r.ticker }}</span>
    <span class="final-score" style="color:{{ r.color }}">{{ r.final_score }}</span>
  </div>

  <div class="sub-row">
    <span>NLP <strong>{{ r.sent_score }}</strong></span>
    <span>5d momentum <strong style="color:{% if r.mom_pct >= 0 %}var(--bull){% else %}var(--bear){% endif %}">
      {% if r.mom_pct >= 0 %}+{% endif %}{{ r.mom_pct }}%
    </strong></span>
  </div>

  <!-- normalised score bar: map [-1,1] → [0,100%] -->
  <div class="bar-track">
    <div class="bar-fill"
         style="width:{{ ((r.final_score + 1) / 2 * 100)|round|int }}%;
                background:{{ r.color }}44;
                border:1px solid {{ r.color }};">
    </div>
  </div>

  <span class="badge" style="background:{{ r.color }}18; color:{{ r.color }};">
    <span class="badge-dot" style="background:{{ r.color }}"></span>
    {{ r.sentiment }}
  </span>
  <div class="action">{{ r.action }}</div>
</div>
{% endfor %}
</div>

<footer>
  Scores are informational only — not financial advice.<br/>
  <a href="/">Refresh</a> to pull latest data.
</footer>

</body>
</html>
"""
    return render_template_string(html, rows=rows)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)