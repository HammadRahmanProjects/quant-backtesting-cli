import logging
from typing import Dict

import pandas as pd

from data.loaders import clean_dataframe
from data.repair import repair_bad_rows
from data.validators import validate_data

logger = logging.getLogger(__name__)

def process_market_data(market_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Full data pipeline per ticker:
        1. Clean  — parse types, drop unparseable rows
        2. Repair — prompt user to fix any bad rows from cleaning
        3. Validate — business logic checks (OHLCV sanity)
        4. Repair — prompt user to fix any rows that failed validation
    """
    logger.info(
        "Starting data pipeline for %d ticker(s): %s",
        len(market_data),
        list(market_data.keys()),
    )

    processed_data = {}

    for ticker, df in market_data.items():
        logger.info("%s — pipeline start | input rows: %d", ticker, len(df))

        clean_df, original_bad_df, parsed_bad_df = clean_dataframe(df)
        logger.debug(
            "%s — cleaning complete | clean: %d, bad: %d",
            ticker,
            len(clean_df),
            len(original_bad_df),
        )

        repaired_loader_df = repair_bad_rows(original_bad_df, parsed_bad_df)

        if not repaired_loader_df.empty:
            logger.info(
                "%s — %d loader-level row(s) repaired",
                ticker,
                len(repaired_loader_df),
            )
            loader_final_df = pd.concat(
                [clean_df, repaired_loader_df],
                ignore_index=True,
            )
        else:
            loader_final_df = clean_df.copy()

        loader_final_df = loader_final_df.sort_values("datetime").reset_index(drop=True)

        if loader_final_df.empty:
            logger.error("%s — DataFrame empty after cleaning/repair", ticker)
            raise ValueError(f"{ticker}: DataFrame is empty after cleaning/repair.")

        valid_df, invalid_df = validate_data(loader_final_df)
        logger.debug(
            "%s — validation complete | valid: %d, invalid: %d",
            ticker,
            len(valid_df),
            len(invalid_df),
        )

        if not invalid_df.empty:
            logger.info(
                "%s — %d validation-level row(s) flagged for repair",
                ticker,
                len(invalid_df),
            )
            repaired_invalid_df = repair_bad_rows(invalid_df, invalid_df)

            if not repaired_invalid_df.empty:
                logger.info(
                    "%s — %d validation-level row(s) repaired",
                    ticker,
                    len(repaired_invalid_df),
                )
                final_df = pd.concat(
                    [valid_df, repaired_invalid_df],
                    ignore_index=True,
                )
            else:
                logger.warning(
                    "%s — no rows recovered from validation repair",
                    ticker,
                )
                final_df = valid_df.copy()
        else:
            final_df = valid_df.copy()

        final_df = final_df.sort_values("datetime").reset_index(drop=True)

        if final_df.empty:
            logger.error("%s — DataFrame empty after validation/repair", ticker)
            raise ValueError(f"{ticker}: DataFrame is empty after validation/repair.")

        processed_data[ticker] = final_df

        logger.info(
            "%s — pipeline complete | final rows: %d",
            ticker,
            len(final_df),
        )

    logger.info(
        "Data pipeline complete for all %d ticker(s)",
        len(processed_data),
    )

    return processed_data