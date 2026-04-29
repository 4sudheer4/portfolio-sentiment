import yfinance as yf

def debug_momentum(ticker: str):
    stock = yf.Ticker(ticker)
    
    print(f"\n{'='*50}")
    print(f"DEBUG: {ticker} Momentum Calculation")
    print(f"{'='*50}")

    # Raw fetch
    hist_raw = stock.history(period="5d", interval="1d", prepost=False)
    print(f"\n[1] Raw yfinance rows (period=5d):")
    #print(hist_raw[["Open", "Close", "Volume"]])
    

    # After volume filter
    hist_clean = hist_raw[hist_raw["Volume"] > 0]
    print(f"\n[2] After Volume > 0 filter:")
    print(hist_clean[["Open", "Close", "Volume"]])
    print(f"Total rows: {len(hist_clean)}")

    # OLD method (iloc[0] vs iloc[-1])
    old_start = hist_clean["Close"].iloc[0]
    old_end   = hist_clean["Close"].iloc[-1]
    old_pct   = (old_end - old_start) / old_start * 100
    print(f"\n[3] OLD method (iloc[0] vs iloc[-1]):")
    print(f"  Start price : ${old_start:.2f}  ({hist_clean.index[0].date()})")
    print(f"  End price   : ${old_end:.2f}  ({hist_clean.index[-1].date()})")
    print(f"  Change      : {old_pct:.2f}%")

    # NEW method (tail(5))
    last_5 = hist_clean.tail(5)
    new_start = last_5["Close"].iloc[0]
    new_end   = last_5["Close"].iloc[-1]
    new_pct   = (new_end - new_start) / new_start * 100
    print(f"\n[4] NEW method (tail(5)):")
    print(f"  Start price : ${new_start:.2f}  ({last_5.index[0].date()})")
    print(f"  End price   : ${new_end:.2f}  ({last_5.index[-1].date()})")
    print(f"  Change      : {new_pct:.2f}%")

    # 10d fetch for comparison
    hist_10d = stock.history(period="10d", interval="1d", prepost=False)
    hist_10d = hist_10d[hist_10d["Volume"] > 0]
    print(f"\n[5] Last 5 rows from period=10d fetch:")
    print(hist_10d[["Close", "Volume"]].tail(5))

    print(f"\n[6] All closing prices (10d):")
    for date, row in hist_10d.iterrows():
        print(f"  {date.date()}  Close: ${row['Close']:.2f}  Volume: {int(row['Volume']):,}")

debug_momentum("HYMC")