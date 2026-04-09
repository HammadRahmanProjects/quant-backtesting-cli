from typing import Dict

import pandas as pd

from engine.backtester import Backtester
from engine.portfolio_builder import get_allocated_cash

def run_backtests(
    signals_data: Dict[str, pd.DataFrame],
    portfolio
) -> Dict[str, pd.DataFrame]:
    backtest_results = {}

    for ticker, signals_df in signals_data.items():
        backtester = Backtester(
            signals_df,
            initial_cash=get_allocated_cash(portfolio, ticker),
            commission_rate=portfolio.commission_rate,
            slippage_rate=portfolio.slippage_rate
        )

        results_df = backtester.run()
        backtest_results[ticker] = results_df

    return backtest_results