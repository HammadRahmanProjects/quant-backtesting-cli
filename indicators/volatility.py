import numpy as np
import pandas as pd

def add_bollinger_bands(
    df: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    if window <= 0:
        raise ValueError("window must be a positive integer.")
    if num_std <= 0:
        raise ValueError("num_std must be positive.")
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain a 'close' column.")

    result = df.copy()

    rolling_mean = result["close"].rolling(window=window).mean()
    rolling_std  = result["close"].rolling(window=window).std(ddof=1)

    result["bb_middle"] = rolling_mean
    result["bb_upper"]  = rolling_mean + num_std * rolling_std
    result["bb_lower"]  = rolling_mean - num_std * rolling_std
    result["bb_width"]  = (result["bb_upper"] - result["bb_lower"]) / result["bb_middle"]
    result["bb_pct"]    = (result["close"] - result["bb_lower"]) / (
        result["bb_upper"] - result["bb_lower"]
    )

    return result

def add_atr(
    df: pd.DataFrame,
    window: int = 14,
) -> pd.DataFrame:
    if window <= 0:
        raise ValueError("window must be a positive integer.")

    required = {"high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required}")

    result = df.copy()

    prev_close = result["close"].shift(1)

    tr = pd.concat([
        result["high"] - result["low"],
        (result["high"] - prev_close).abs(),
        (result["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)

    result["true_range"] = tr
    result["atr"]        = tr.rolling(window=window).mean()

    return result

def add_historical_volatility(
    df: pd.DataFrame,
    window: int = 20,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    if window <= 0:
        raise ValueError("window must be a positive integer.")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain a 'close' column.")

    result = df.copy()

    result["log_return"] = np.log(result["close"] / result["close"].shift(1))

    result["historical_volatility"] = (
        result["log_return"]
        .rolling(window=window)
        .std(ddof=1)
        * np.sqrt(periods_per_year)
    )

    return result