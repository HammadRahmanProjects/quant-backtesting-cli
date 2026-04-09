import pandas as pd
from typing import Dict

from data.loaders import clean_dataframe
from data.repair import repair_bad_rows
from data.validators import validate_data

def process_market_data(market_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Full data pipeline:
    - cleaning
    - repair (loader-level)
    - validation
    - repair (validation-level)

    Returns:
        Dict[ticker -> cleaned DataFrame]
    """
    processed_data = {}

    for ticker, df in market_data.items():
        clean_df, original_bad_df, parsed_bad_df = clean_dataframe(df)

        repaired_loader_df = repair_bad_rows(original_bad_df, parsed_bad_df)

        if not repaired_loader_df.empty:
            loader_final_df = pd.concat(
                [clean_df, repaired_loader_df],
                ignore_index=True
            )
        else:
            loader_final_df = clean_df.copy()

        loader_final_df = loader_final_df.sort_values("datetime").reset_index(drop=True)

        if loader_final_df.empty:
            raise ValueError(f"{ticker}: DataFrame is empty after cleaning/repair.")

        valid_df, invalid_df = validate_data(loader_final_df)

        if not invalid_df.empty:
            repaired_invalid_df = repair_bad_rows(invalid_df, invalid_df)

            if not repaired_invalid_df.empty:
                final_df = pd.concat(
                    [valid_df, repaired_invalid_df],
                    ignore_index=True
                )
            else:
                final_df = valid_df.copy()
        else:
            final_df = valid_df.copy()

        final_df = final_df.sort_values("datetime").reset_index(drop=True)

        if final_df.empty:
            raise ValueError(f"{ticker}: DataFrame is empty after validation/repair.")

        processed_data[ticker] = final_df

    return processed_data