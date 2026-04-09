from typing import Dict

import pandas as pd

from strategies.registry import AVAILABLE_STRATEGIES

def generate_signals(
    processed_data: Dict[str, pd.DataFrame],
    portfolio
) -> Dict[str, pd.DataFrame]:
    signals_data = {}

    for ticker, df in processed_data.items():
        strategy_info = portfolio.strategy_map[ticker]
        strategy_name = strategy_info["name"]
        strategy_params = strategy_info["params"]

        strategy_class = AVAILABLE_STRATEGIES.get(strategy_name)

        if strategy_class is None:
            raise ValueError(f"{ticker}: Unknown strategy '{strategy_name}'.")

        strategy = strategy_class(df, **strategy_params)
        signals_df = strategy.generate_signals()

        signals_data[ticker] = signals_df

    return signals_data