import logging
from typing import Dict

import pandas as pd

from strategies.registry import AVAILABLE_STRATEGIES

logger = logging.getLogger(__name__)

def generate_signals(
    processed_data: Dict[str, pd.DataFrame],
    portfolio,
    strategy_params_map: Dict[str, Dict],
) -> Dict[str, pd.DataFrame]:
    logger.info(
        "Generating signals for %d ticker(s): %s",
        len(processed_data),
        list(processed_data.keys()),
    )

    signals_data = {}

    for ticker, df in processed_data.items():
        strategy_info = portfolio.strategy_map[ticker]
        strategy_name = strategy_info["name"]
        params        = strategy_params_map.get(ticker, {})

        logger.info(
            "%s — strategy: %s | params: %s | rows: %d",
            ticker,
            strategy_name,
            params,
            len(df),
        )

        strategy_class = AVAILABLE_STRATEGIES.get(strategy_name)

        if strategy_class is None:
            logger.error(
                "%s — unknown strategy '%s'",
                ticker,
                strategy_name,
            )
            raise ValueError(f"{ticker}: Unknown strategy '{strategy_name}'.")

        try:
            strategy   = strategy_class(df, **params)
            signals_df = strategy.generate_signals()

            n_long  = (signals_df["signal"] == 1).sum()
            n_short = (signals_df["signal"] == -1).sum()
            n_flat  = (signals_df["signal"] == 0).sum()

            logger.info(
                "%s — signals generated | long: %d | short: %d | flat: %d",
                ticker,
                n_long,
                n_short,
                n_flat,
            )

            signals_data[ticker] = signals_df

        except Exception as e:
            logger.error(
                "%s — signal generation failed: %s",
                ticker,
                e,
                exc_info=True,
            )
            raise

    logger.info(
        "Signal generation complete for all %d ticker(s)",
        len(signals_data),
    )

    return signals_data