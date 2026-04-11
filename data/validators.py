import logging

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["datetime", "open", "high", "low", "close", "volume"]
NUMERIC_COLUMNS  = ["open", "high", "low", "close", "volume"]


def validate_data(df: pd.DataFrame):
    logger.debug("validate_data called — input shape: %s", df.shape)

    if df.empty:
        logger.error("validate_data received an empty DataFrame")
        raise ValueError("DataFrame is empty.")

    working_df = df.copy()
    working_df.columns = [str(col).strip().lower() for col in working_df.columns]

    missing = [col for col in REQUIRED_COLUMNS if col not in working_df.columns]
    if missing:
        logger.error("validate_data — missing required columns: %s", missing)
        raise ValueError(f"Missing required columns: {missing}")

    working_df["error_reason"] = ""

    # --- Individual validation checks ---

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

    invalid_mask  = working_df["error_reason"] != ""
    valid_df      = working_df.loc[~invalid_mask].copy().reset_index(drop=True)
    invalid_df    = working_df.loc[invalid_mask].copy().reset_index(drop=True)

    n_valid   = len(valid_df)
    n_invalid = len(invalid_df)

    if n_invalid > 0:
        # Log a breakdown of exactly which validation rules were triggered
        reason_counts = (
            invalid_df["error_reason"]
            .str.split("; ")
            .explode()
            .value_counts()
        )
        logger.warning(
            "validate_data — %d invalid row(s) found out of %d total",
            n_invalid,
            len(df),
        )
        for reason, count in reason_counts.items():
            logger.warning("  %-30s — %d row(s)", reason, count)
    else:
        logger.debug(
            "validate_data — all %d rows passed validation",
            n_valid,
        )

    logger.debug(
        "validate_data complete — valid: %d, invalid: %d",
        n_valid,
        n_invalid,
    )

    return valid_df, invalid_df