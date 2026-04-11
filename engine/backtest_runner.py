import logging
from typing import Dict, Optional

import pandas as pd

from engine.backtester import Backtester
from engine.event_backtester import EventBacktester
from engine.portfolio_builder import get_allocated_cash
from engine.position_sizer import SizingMethod

logger = logging.getLogger(__name__)

def run_backtests(
    signals_data  : Dict[str, pd.DataFrame],
    portfolio,
    sizing_method : SizingMethod = SizingMethod.FULL_PORTFOLIO,
    sizing_params : Optional[Dict] = None,
    use_event_backtester: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Run backtests for all tickers in the portfolio.

    Args:
        signals_data         : dict of ticker → DataFrame with signal column
        portfolio            : Portfolio object
        sizing_method        : position sizing method to use
        sizing_params        : kwargs passed to the position sizer
        use_event_backtester : if True, use event-driven backtester (default)
                               if False, use vectorized (used by optimization)
    """
    logger.info(
        "run_backtests — %d ticker(s): %s | engine: %s | sizing: %s",
        len(signals_data),
        list(signals_data.keys()),
        "event-driven" if use_event_backtester else "vectorized",
        sizing_method.value,
    )

    backtest_results = {}

    for ticker, signals_df in signals_data.items():
        allocated_cash = get_allocated_cash(portfolio, ticker)

        logger.info(
            "%s — starting backtest | cash: %.2f | rows: %d",
            ticker,
            allocated_cash,
            len(signals_df),
        )

        try:
            if use_event_backtester:
                backtester = EventBacktester(
                    df              = signals_df,
                    initial_cash    = allocated_cash,
                    commission_rate = portfolio.commission_rate,
                    slippage_rate   = portfolio.slippage_rate,
                    sizing_method   = sizing_method,
                    sizing_params   = sizing_params or {},
                )
            else:
                backtester = Backtester(
                    signals_df,
                    initial_cash    = allocated_cash,
                    commission_rate = portfolio.commission_rate,
                    slippage_rate   = portfolio.slippage_rate,
                )

            results_df             = backtester.run()
            backtest_results[ticker] = results_df

            final_equity = results_df["equity_curve"].iloc[-1]
            total_return = (final_equity / allocated_cash - 1) * 100

            logger.info(
                "%s — backtest complete | final equity: %.2f | return: %.2f%%",
                ticker,
                final_equity,
                total_return,
            )

        except Exception as e:
            logger.error(
                "%s — backtest failed: %s",
                ticker,
                e,
                exc_info=True,
            )
            raise

    logger.info(
        "run_backtests complete — %d ticker(s) processed",
        len(backtest_results),
    )

    return backtest_results