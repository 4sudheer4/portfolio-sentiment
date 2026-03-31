import robin_stocks.robinhood as rh
import pandas as pd
import yfinance as yf
import nltk
import os
from dotenv import load_dotenv
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Load credentials from .env file
load_dotenv()

nltk.download('vader_lexicon', quiet=True)

def login_to_robinhood():
    username = os.getenv("RH_USERNAME")
    password = os.getenv("RH_PASSWORD")

    # Fail fast with a clear message if credentials are missing
    if not username or not password:
        raise ValueError(
            "Missing credentials. Create a .env file with:\n"
            "  RH_USERNAME=your_email\n"
            "  RH_PASSWORD=your_password"
        )

    rh.login(username, password, store_session=True)
    print("Successfully logged into Robinhood.")

def get_sentiment_score(ticker):
    """
    Fetches recent news from Yahoo Finance via yfinance
    and returns average VADER compound sentiment score.
    """
    analyzer = SentimentIntensityAnalyzer()
    scores = []

    try:
        stock = yf.Ticker(ticker)
        news_items = stock.news

        if not news_items:
            print(f"  No news found for {ticker}")
            return 0.0

        for item in news_items[:10]:
            title = (
                item.get("title")
                or item.get("content", {}).get("title")
            )
            if not title:
                continue
            score = analyzer.polarity_scores(title)['compound']
            scores.append(score)

        return round(sum(scores) / len(scores), 4) if scores else 0.0

    except Exception as e:
        print(f"  Error analyzing {ticker}: {e}")
        return 0.0

def analyze_my_portfolio():
    login_to_robinhood()

    print("Fetching portfolio holdings...")
    holdings = rh.account.build_holdings()

    if not holdings:
        print("No holdings found or session expired.")
        return pd.DataFrame()

    report_data = []

    for ticker in holdings.keys():
        print(f"Analyzing {ticker}...")
        avg_score = get_sentiment_score(ticker)

        if avg_score >= 0.05:
            sentiment = "Bullish"
            action = "HODL / Accumulate"
        elif avg_score <= -0.05:
            sentiment = "Bearish"
            action = "Review Sell / Stop-Loss"
        else:
            sentiment = "Neutral"
            action = "Maintain Position"

        report_data.append({
            'Ticker': ticker,
            'Score': avg_score,
            'Sentiment': sentiment,
            'Suggested Action': action
        })

    df = pd.DataFrame(report_data)
    df.sort_values('Score', ascending=False, inplace=True)
    return df

if __name__ == "__main__":
    final_report = analyze_my_portfolio()

    print("\n--- PORTFOLIO SENTIMENT REPORT ---")
    print(final_report.to_string(index=False))

    # Uncomment to save:
    # final_report.to_csv('portfolio_analysis.csv', index=False)

# Score guide:
# +1.0 = extremely positive/bullish news
#  0.0 = completely neutral news
# -1.0 = extremely negative/bearish news
