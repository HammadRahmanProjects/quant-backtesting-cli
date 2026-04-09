import yfinance as yf
import pandas as pd

def validate_ticker(ticker: str) -> bool:
    try:
        df = yf.Ticker(ticker).history(period="5d", auto_adjust=False)
        return not df.empty
    except Exception:
        return False

def pull_market_data(portfolio):
    market_data = {}

    for ticker in portfolio.tickers:
    
        df = yf.download(
            ticker,
            start=portfolio.start_date,
            end=portfolio.end_date,
            interval=portfolio.interval,
            progress=False,
            auto_adjust=False,
            threads=False
        )

        if df.empty:
            raise ValueError(f"No data returned for {ticker}.")

        df = df.reset_index()

        # Flatten columns if yfinance returns MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

        # Normalize column names
        df.columns = [str(col).strip().lower() for col in df.columns]

        # Handle both "date" and "datetime"
        if "date" in df.columns:
            df = df.rename(columns={"date": "datetime"})
        elif "datetime" not in df.columns:
            raise ValueError(f"{ticker} data missing datetime/date column.")

        required_columns = ["datetime", "open", "high", "low", "close", "volume"]

        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            print(f"\nNormalized columns for {ticker}: {df.columns.tolist()}")
            raise ValueError(f"{ticker} data missing columns: {missing}")

        df = df[required_columns].copy()

        market_data[ticker] = df

    return market_data

