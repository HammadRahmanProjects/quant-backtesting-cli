import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from engine.metrics import (
    calculate_max_drawdown,
    calculate_sharpe_ratio,
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
    ticker              : str
    n_simulations       : int
    confidence          : float

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
    n_active_bars       : int   = 0

    sample_equity_curves: List[List[float]] = field(default_factory=list)


def run_monte_carlo(
    results_df      : pd.DataFrame,
    ticker          : str,
    interval        : str,
    initial_cash    : float,
    n_simulations   : int            = 10_000,
    confidence      : float          = 0.95,
    n_sample_curves : int            = 200,
    random_seed     : Optional[int]  = None,
    min_active_bars : int            = 30,
) -> MonteCarloResult:
    """
    Run Monte Carlo simulation by randomly shuffling strategy returns.

    Key fix vs naive implementation:
        We only shuffle ACTIVE bars — bars where the strategy had a
        non-zero position. Shuffling zero returns adds no information
        and causes the CI to collapse to a single value when the
        strategy is mostly flat (which is common with tight signal
        thresholds).

        The active returns are shuffled and then re-embedded back into
        the full return series at their original positions, preserving
        the zero-return flat periods while randomizing the active
        trading sequence.

    Args:
        results_df      : backtest results DataFrame
        ticker          : ticker name for logging
        interval        : data interval for annualization
        initial_cash    : starting portfolio value
        n_simulations   : number of Monte Carlo simulations
        confidence      : confidence interval level (default 95%)
        n_sample_curves : number of equity curves to store
        random_seed     : optional seed for reproducibility
        min_active_bars : minimum active bars required to run simulation
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

    # Full return series including flat periods
    all_returns = results_df["net_strategy_returns"].fillna(0).values

    # Active return indices — bars where strategy was actually trading
    # A bar is active if net_strategy_returns is non-zero
    active_mask    = all_returns != 0
    active_indices = np.where(active_mask)[0]
    active_returns = all_returns[active_mask]
    n_active       = len(active_returns)

    logger.info(
        "%s — Monte Carlo | total bars: %d | active bars: %d (%.1f%%)",
        ticker,
        len(all_returns),
        n_active,
        100 * n_active / len(all_returns) if len(all_returns) > 0 else 0,
    )

    if n_active < min_active_bars:
        logger.warning(
            "%s — insufficient active bars for Monte Carlo "
            "(%d active, need %d). "
            "Strategy may be too infrequent to assess edge meaningfully.",
            ticker,
            n_active,
            min_active_bars,
        )
        return MonteCarloResult(
            ticker        = ticker,
            n_simulations = n_simulations,
            confidence    = confidence,
            n_active_bars = n_active,
        )

    # -----------------------------------------------------------------------
    # Observed statistics
    # -----------------------------------------------------------------------
    observed_equity = results_df["equity_curve"].values
    observed_sharpe = calculate_sharpe_ratio(
        pd.Series(active_returns), periods_per_year
    )
    observed_max_dd = calculate_max_drawdown(pd.Series(observed_equity))
    observed_return = calculate_total_return(pd.Series(observed_equity))

    logger.debug(
        "%s — observed | Sharpe: %.4f | MaxDD: %.4f | Return: %.4f",
        ticker,
        observed_sharpe,
        observed_max_dd,
        observed_return,
    )

    # -----------------------------------------------------------------------
    # Run simulations
    # Pre-generate all shuffled active return arrays at once — vectorized,
    # much faster than shuffling inside the loop
    # -----------------------------------------------------------------------
    sharpe_dist = np.zeros(n_simulations)
    max_dd_dist = np.zeros(n_simulations)
    return_dist = np.zeros(n_simulations)

    # Shape: (n_simulations, n_active)
    shuffled_active = np.array([
        active_returns[np.random.permutation(n_active)]
        for _ in range(n_simulations)
    ])

    sample_interval = max(1, n_simulations // n_sample_curves)
    sample_curves   = []

    for i in range(n_simulations):
        # Re-embed shuffled active returns back into full return series
        # Flat (zero) periods stay in place — only active periods are shuffled
        sim_returns                  = all_returns.copy()
        sim_returns[active_indices]  = shuffled_active[i]

        # Rebuild equity curve
        equity = initial_cash * np.cumprod(1 + sim_returns)

        # Compute metrics on active returns only for Sharpe
        # (same as observed — consistent comparison)
        sim_active_returns = shuffled_active[i]

        sharpe_dist[i] = calculate_sharpe_ratio(
            pd.Series(sim_active_returns), periods_per_year
        )
        max_dd_dist[i] = calculate_max_drawdown(pd.Series(equity))
        return_dist[i] = (equity[-1] / initial_cash) - 1

        if i % sample_interval == 0:
            sample_curves.append(equity.tolist())

    logger.debug(
        "%s — simulations complete | "
        "Sharpe: mean=%.4f std=%.4f | "
        "MaxDD: mean=%.4f std=%.4f",
        ticker,
        sharpe_dist.mean(),
        sharpe_dist.std(),
        max_dd_dist.mean(),
        max_dd_dist.std(),
    )

    # -----------------------------------------------------------------------
    # Confidence intervals
    # -----------------------------------------------------------------------
    alpha       = 1 - confidence
    lower_pct   = alpha / 2 * 100
    upper_pct   = (1 - alpha / 2) * 100

    sharpe_ci_lower = float(np.percentile(sharpe_dist, lower_pct))
    sharpe_ci_upper = float(np.percentile(sharpe_dist, upper_pct))
    max_dd_ci_lower = float(np.percentile(max_dd_dist, lower_pct))
    max_dd_ci_upper = float(np.percentile(max_dd_dist, upper_pct))

    # Probability observed Sharpe beats simulated Sharpes
    prob_outperformance = float(np.mean(observed_sharpe > sharpe_dist))

    logger.info(
        "%s — Monte Carlo complete | active bars: %d | "
        "Sharpe CI: [%.4f, %.4f] | "
        "MaxDD CI: [%.4f, %.4f] | "
        "P(outperform): %.4f",
        ticker,
        n_active,
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
        n_active_bars        = n_active,
        sample_equity_curves = sample_curves,
    )

def run_monte_carlo_portfolio(
    backtest_results : Dict[str, pd.DataFrame],
    portfolio,
    n_simulations    : int   = 10_000,
    confidence       : float = 0.95,
    n_sample_curves  : int   = 200,
) -> Dict[str, "MonteCarloResult"]:
    """Run Monte Carlo simulation for all tickers in the portfolio."""
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
            results_df      = results_df,
            ticker          = ticker,
            interval        = portfolio.interval,
            initial_cash    = initial_cash,
            n_simulations   = n_simulations,
            confidence      = confidence,
            n_sample_curves = n_sample_curves,
        )

    return mc_results