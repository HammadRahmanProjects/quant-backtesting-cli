import pandas as pd

REQUIRED_COLUMNS = ["datetime", "open", "high", "low", "close", "volume"]
NUMERIC_COLUMNS = ["open", "high", "low", "close", "volume"]

def clean_dataframe(df: pd.DataFrame):
    if df.empty:
        raise ValueError("The DataFrame contains no rows.")

    working_df = df.copy()

    working_df.columns = [str(col).strip().lower() for col in working_df.columns]

    missing = [col for col in REQUIRED_COLUMNS if col not in working_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    original_df = working_df[REQUIRED_COLUMNS].copy()
    parsed_df = original_df.copy()

    parsed_df["datetime"] = pd.to_datetime(parsed_df["datetime"], errors="coerce")

    for col in NUMERIC_COLUMNS:
        parsed_df[col] = pd.to_numeric(parsed_df[col], errors="coerce")

    bad_mask = parsed_df[REQUIRED_COLUMNS].isna().any(axis=1)

    clean_df = parsed_df.loc[~bad_mask].copy()
    original_bad_df = original_df.loc[bad_mask].copy()
    parsed_bad_df = parsed_df.loc[bad_mask].copy()

    clean_df = clean_df.sort_values("datetime").reset_index(drop=True)
    original_bad_df = original_bad_df.reset_index(drop=True)
    parsed_bad_df = parsed_bad_df.reset_index(drop=True)

    return clean_df, original_bad_df, parsed_bad_df