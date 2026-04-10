import pandas as pd

def add_moving_averages(
    df: pd.DataFrame,
    short_window: int,
    long_window: int
) -> pd.DataFrame:
    if short_window <= 0 or long_window <= 0:
        raise ValueError("Moving average windows must be positive integers.")

    if short_window >= long_window:
        raise ValueError("short_window must be less than long_window.")

    if "close" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'close' column.")

    result = df.copy()

    result["short_ma"] = result["close"].rolling(window=short_window).mean()
    result["long_ma"] = result["close"].rolling(window=long_window).mean()

    return result

def add_price_vs_moving_average(
    df: pd.DataFrame,
    window: int,
    column_name: str = "price_vs_ma"
) -> pd.DataFrame:
    if window <= 0:
        raise ValueError("window must be positive.")

    if "close" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'close' column.")

    result = df.copy()

    moving_average = result["close"].rolling(window=window).mean()
    result[column_name] = result["close"] / moving_average

    return result
