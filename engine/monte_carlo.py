import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from engine.metrics import (
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_total_return,
    get_periods_per_year,
)

logger = logging.getLogger(__name__)

@dataclass
class MonteCarloResult:
    """
    Results from a Monte Carlo simulation on a strategy's return stream.

    All distributions are from n_simulations random shuffles of the
    actual trade returns — same returns, different order each time.
    """
    ticker          : str
    n_simulations   : int
    confidence      : float

    observed_sharpe     : float = 0.0
    observed_max_dd     : float = 0.0
    observed_return     : float = 0.0

    sharpe_distribution : List[float] = field(default_factory=list)
    max_dd_distribution : List[float] = field(default_factory=list)
    return_distribution : List[float] = field(default_factory=list)

    sharpe_ci_lower     : float = 0.0
    sharpe_ci_upper     : float = 0.0

    max_dd_ci_lower     : float = 0.0
    max_dd_ci_upper     : float = 0.0

    prob_outperformance : float = 0.0

    sample_equity_curves: List[List[float]] = field(default_factory=list)

def run_monte_carlo(
    results_df      : pd.DataFrame,
    ticker          : str,
    interval        : str,
    initial_cash    : float,
    n_simulations   : int   = 10_000,
    confidence      : float = 0.95,
    n_sample_curves : int   = 200,
    random_seed     : Optional[int] = None,
) -> MonteCarloResult:
    """
    Run Monte Carlo simulation by randomly shuffling strategy returns.

    This answers the question: "Is our observed Sharpe ratio the result
    of genuine edge, or could it have occurred by random chance given
    the same set of returns in a different order?"

    Process:
        1. Extract net strategy returns from the backtest
        2. Shuffle their order n_simulations times
        3. Rebuild equity curve from each shuffled sequence
        4. Compute Sharpe, max drawdown, total return for each simulation
        5. Build confidence intervals and probability of outperformance

    Args:
        results_df      : backtest results DataFrame
        ticker          : ticker name for logging
        interval        : data interval for annualization
        initial_cash    : starting portfolio value
        n_simulations   : number of Monte Carlo simulations (default 10,000)
        confidence      : confidence interval level (default 95%)
        n_sample_curves : number of equity curves to store for visualization
        random_seed     : optional seed for reproducibility

    Returns:
        MonteCarloResult with full distribution statistics
    """
    logger.info(
        "%s — Monte Carlo start | n_simulations: %d | confidence: %.2f",
        ticker,
        n_simulations,
        confidence,
    )

    if random_seed is not None:
        np.random.seed(random_seed)

    periods_per_year = get_periods_per_year(interval)

    returns = results_df["net_strategy_returns"].dropna().values

    if len(returns) < 10:
        logger.warning(
            "%s — insufficient return history for Monte Carlo (%d bars)",
            ticker,
            len(returns),
        )
        return MonteCarloResult(
            ticker        = ticker,
            n_simulations = n_simulations,
            confidence    = confidence,
        )

    n_bars = len(returns)

    observed_equity  = results_df["equity_curve"].values
    observed_sharpe  = calculate_sharpe_ratio(
        pd.Series(returns), periods_per_year
    )
    observed_max_dd  = calculate_max_drawdown(pd.Series(observed_equity))
    observed_return  = calculate_total_return(pd.Series(observed_equity))

    logger.debug(
        "%s — observed | Sharpe: %.4f | MaxDD: %.4f | Return: %.4f",
        ticker,
        observed_sharpe,
        observed_max_dd,
        observed_return,
    )

    sharpe_dist = np.zeros(n_simulations)
    max_dd_dist = np.zeros(n_simulations)
    return_dist = np.zeros(n_simulations)

    shuffled_indices = np.array([
        np.random.permutation(n_bars)
        for _ in range(n_simulations)
    ])

    sample_interval  = max(1, n_simulations // n_sample_curves)
    sample_curves    = []

    for i in range(n_simulations):
        shuffled_returns = returns[shuffled_indices[i]]

        # Rebuild equity curve from shuffled returns
        equity = initial_cash * np.cumprod(1 + shuffled_returns)

        sharpe_dist[i] = calculate_sharpe_ratio(
            pd.Series(shuffled_returns), periods_per_year
        )
        max_dd_dist[i] = calculate_max_drawdown(pd.Series(equity))
        return_dist[i] = (equity[-1] / initial_cash) - 1

        # Store sample curves for visualization
        if i % sample_interval == 0:
            sample_curves.append(equity.tolist())

    logger.debug(
        "%s — simulations complete | mean Sharpe: %.4f | "
        "mean MaxDD: %.4f",
        ticker,
        sharpe_dist.mean(),
        max_dd_dist.mean(),
    )

    alpha = 1 - confidence
    lower_pct = alpha / 2 * 100
    upper_pct = (1 - alpha / 2) * 100

    sharpe_ci_lower = float(np.percentile(sharpe_dist, lower_pct))
    sharpe_ci_upper = float(np.percentile(sharpe_dist, upper_pct))
    max_dd_ci_lower = float(np.percentile(max_dd_dist, lower_pct))
    max_dd_ci_upper = float(np.percentile(max_dd_dist, upper_pct))

    prob_outperformance = float(
        np.mean(observed_sharpe > sharpe_dist)
    )

    logger.info(
        "%s — Monte Carlo complete | "
        "Sharpe CI: [%.4f, %.4f] | "
        "MaxDD CI: [%.4f, %.4f] | "
        "P(outperform): %.4f",
        ticker,
        sharpe_ci_lower,
        sharpe_ci_upper,
        max_dd_ci_lower,
        max_dd_ci_upper,
        prob_outperformance,
    )

    return MonteCarloResult(
        ticker               = ticker,
        n_simulations        = n_simulations,
        confidence           = confidence,
        observed_sharpe      = float(observed_sharpe),
        observed_max_dd      = float(observed_max_dd),
        observed_return      = float(observed_return),
        sharpe_distribution  = sharpe_dist.tolist(),
        max_dd_distribution  = max_dd_dist.tolist(),
        return_distribution  = return_dist.tolist(),
        sharpe_ci_lower      = sharpe_ci_lower,
        sharpe_ci_upper      = sharpe_ci_upper,
        max_dd_ci_lower      = max_dd_ci_lower,
        max_dd_ci_upper      = max_dd_ci_upper,
        prob_outperformance  = prob_outperformance,
        sample_equity_curves = sample_curves,
    )


def run_monte_carlo_portfolio(
    backtest_results : Dict[str, pd.DataFrame],
    portfolio,
    n_simulations    : int   = 10_000,
    confidence       : float = 0.95,
    n_sample_curves  : int   = 200,
) -> Dict[str, MonteCarloResult]:
    """
    Run Monte Carlo simulation for all tickers in the portfolio.
    """
    logger.info(
        "run_monte_carlo_portfolio — %d ticker(s) | %d simulations",
        len(backtest_results),
        n_simulations,
    )

    mc_results = {}

    for ticker, results_df in backtest_results.items():
        from engine.portfolio_builder import get_allocated_cash
        initial_cash = get_allocated_cash(portfolio, ticker)

        mc_results[ticker] = run_monte_carlo(
            results_df     = results_df,
            ticker         = ticker,
            interval       = portfolio.interval,
            initial_cash   = initial_cash,
            n_simulations  = n_simulations,
            confidence     = confidence,
            n_sample_curves= n_sample_curves,
        )

    return mc_results