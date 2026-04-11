import pandas as pd

def add_rsi(
    df: pd.DataFrame,
    window: int = 14,
) -> pd.DataFrame:
    if window <= 0:
        raise ValueError("window must be a positive integer.")
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain a 'close' column.")

    result = df.copy()

    delta = result["close"].diff()

    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # Wilder's smoothing = EMA with alpha = 1/window
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))

    result["rsi"] = 100 - (100 / (1 + rs))

    return result

def add_macd(
    df: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    if fast_period <= 0 or slow_period <= 0 or signal_period <= 0:
        raise ValueError("All periods must be positive integers.")
    if fast_period >= slow_period:
        raise ValueError("fast_period must be less than slow_period.")
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain a 'close' column.")

    result = df.copy()

    ema_fast = result["close"].ewm(span=fast_period, adjust=False).mean()
    ema_slow = result["close"].ewm(span=slow_period, adjust=False).mean()

    result["macd"]        = ema_fast - ema_slow
    result["macd_signal"] = result["macd"].ewm(span=signal_period, adjust=False).mean()
    result["macd_hist"]   = result["macd"] - result["macd_signal"]

    return result

def add_roc(
    df: pd.DataFrame,
    window: int = 10,
) -> pd.DataFrame:
    if window <= 0:
        raise ValueError("window must be a positive integer.")
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain a 'close' column.")

    result = df.copy()

    result["roc"] = result["close"].pct_change(periods=window) * 100

    return result

def add_momentum(
    df: pd.DataFrame,
    window: int = 10,
) -> pd.DataFrame:
    if window <= 0:
        raise ValueError("window must be a positive integer.")
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain a 'close' column.")

    result = df.copy()

    result["momentum"] = result["close"].diff(periods=window)

    return result
