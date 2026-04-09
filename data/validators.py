import pandas as pd

REQUIRED_COLUMNS = ["datetime", "open", "high", "low", "close", "volume"]
NUMERIC_COLUMNS = ["open", "high", "low", "close", "volume"]

def validate_data(df: pd.DataFrame):
    if df.empty:
        raise ValueError("DataFrame is empty.")

    working_df = df.copy()
    working_df.columns = [str(col).strip().lower() for col in working_df.columns]

    missing = [col for col in REQUIRED_COLUMNS if col not in working_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    working_df["error_reason"] = ""

    duplicate_mask = working_df["datetime"].duplicated(keep=False)
    working_df.loc[duplicate_mask, "error_reason"] += "duplicate_timestamp; "

    zero_mask = (working_df[NUMERIC_COLUMNS] == 0).any(axis=1)
    working_df.loc[zero_mask, "error_reason"] += "zero_value; "

    high_low_mask = working_df["high"] < working_df["low"]
    working_df.loc[high_low_mask, "error_reason"] += "high_less_than_low; "

    open_range_mask = (
        (working_df["open"] < working_df["low"]) |
        (working_df["open"] > working_df["high"])
    )
    working_df.loc[open_range_mask, "error_reason"] += "open_outside_range; "

    close_range_mask = (
        (working_df["close"] < working_df["low"]) |
        (working_df["close"] > working_df["high"])
    )
    working_df.loc[close_range_mask, "error_reason"] += "close_outside_range; "

    negative_volume_mask = working_df["volume"] < 0
    working_df.loc[negative_volume_mask, "error_reason"] += "negative_volume; "

    working_df["error_reason"] = working_df["error_reason"].str.strip("; ")

    invalid_mask = working_df["error_reason"] != ""

    valid_df = working_df.loc[~invalid_mask].copy()
    invalid_df = working_df.loc[invalid_mask].copy()

    valid_df = valid_df.reset_index(drop=True)
    invalid_df = invalid_df.reset_index(drop=True)

    return valid_df, invalid_df