import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from engine.backtester import Backtester
from engine.metrics import compute_all_metrics
from engine.optimization import (
    _build_param_combinations,
    _chunk_list,
    _evaluate_param_chunk,
    _is_valid_param_combo,
)
from engine.portfolio_builder import get_allocated_cash
from strategies.registry import AVAILABLE_STRATEGIES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class WindowResult:
    """Results for a single walk-forward window."""
    window_index     : int
    train_start      : Any
    train_end        : Any
    test_start       : Any
    test_end         : Any
    best_params      : Dict[str, Any]
    best_train_metric: float
    test_metrics     : Dict[str, float]
    test_equity_curve: List[float]
    n_train_bars     : int
    n_test_bars      : int


@dataclass
class WalkForwardResult:
    """Aggregated results across all walk-forward windows for a single ticker."""
    ticker           : str
    strategy_name    : str
    train_period     : int
    test_period      : int
    step_size        : int
    ranking_metric   : str
    windows          : List[WindowResult]      = field(default_factory=list)
    oos_equity_curve : List[float]             = field(default_factory=list)
    param_stability  : List[Dict[str, Any]]    = field(default_factory=list)
    oos_metrics      : Dict[str, float]        = field(default_factory=dict)
    elapsed_seconds  : float                   = 0.0


# ---------------------------------------------------------------------------
# Step 1 helper — backtest a single test window
# (defined before _optimize_window so it's in scope)
# ---------------------------------------------------------------------------

def _backtest_window(
    df             : pd.DataFrame,
    ticker         : str,
    strategy_name  : str,
    params         : Dict[str, Any],
    initial_cash   : float,
    commission_rate: float,
    slippage_rate  : float,
    interval       : str,
) -> Tuple[Dict[str, float], List[float]]:
    """
    Run a single backtest on a test window using the given params.
    Returns (metrics_dict, equity_curve_list).
    """
    strategy_class = AVAILABLE_STRATEGIES.get(strategy_name)

    if strategy_class is None:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    strategy   = strategy_class(df, **params)
    signals_df = strategy.generate_signals()

    backtester = Backtester(
        signals_df,
        initial_cash    = initial_cash,
        commission_rate = commission_rate,
        slippage_rate   = slippage_rate,
    )

    results_df = backtester.run()
    metrics    = compute_all_metrics(results_df, interval=interval)

    return metrics, results_df["equity_curve"].tolist()


# ---------------------------------------------------------------------------
# Step 2 helper — optimize a single training window
# Uses ProcessPoolExecutor for the same parallelism as main optimization
# ---------------------------------------------------------------------------

def _optimize_window(
    df             : pd.DataFrame,
    ticker         : str,
    strategy_name  : str,
    initial_cash   : float,
    commission_rate: float,
    slippage_rate  : float,
    interval       : str,
    ranking_metric : str,
    chunk_size     : int = 250,
    max_workers    : int | None = None,
    progress       = None,
    inner_task     = None,
) -> Tuple[Dict[str, Any], float]:
    """
    Run a parallelized optimization on a single training window.
    Returns (best_params, best_metric_value).
    """
    strategy_class = AVAILABLE_STRATEGIES.get(strategy_name)

    if strategy_class is None:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    param_grid         = strategy_class.get_optimization_grid()
    param_names        = strategy_class.get_param_names()
    param_combinations = _build_param_combinations(param_grid)

    valid_combinations = [
        p for p in param_combinations
        if _is_valid_param_combo(p, param_names, len(df))
    ]

    if not valid_combinations:
        raise ValueError(f"{ticker}: No valid param combinations for this window.")

    n_valid = len(valid_combinations)

    logger.debug(
        "%s — window optimization | valid combos: %d",
        ticker,
        n_valid,
    )

    if progress is not None and inner_task is not None:
        progress.reset(inner_task, total=n_valid)

    chunks      = _chunk_list(valid_combinations, chunk_size)
    all_results = []
    completed   = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {
            executor.submit(
                _evaluate_param_chunk,
                ticker,
                df,
                strategy_name,
                chunk,
                initial_cash,
                commission_rate,
                slippage_rate,
                interval,
            ): chunk
            for chunk in chunks
        }

        for future in as_completed(future_to_chunk):
            try:
                chunk_results  = future.result()
                all_results.extend(chunk_results)
                completed     += len(future_to_chunk[future])

                if progress is not None and inner_task is not None:
                    progress.update(inner_task, completed=completed)

            except Exception as e:
                logger.warning(
                    "%s — chunk failed in window optimization: %s",
                    ticker,
                    e,
                )
                continue

    if not all_results:
        raise ValueError(f"{ticker}: No results generated for window.")

    results_df = pd.DataFrame(all_results)

    if ranking_metric not in results_df.columns:
        raise ValueError(f"Ranking metric '{ranking_metric}' not found.")

    best_row    = results_df.loc[results_df[ranking_metric].idxmax()]
    best_params = {k: best_row[k] for k in param_names}
    best_metric = float(best_row[ranking_metric])

    logger.debug(
        "%s — best %s: %.4f | params: %s",
        ticker,
        ranking_metric,
        best_metric,
        best_params,
    )

    return best_params, best_metric


# ---------------------------------------------------------------------------
# Main walk-forward engine
# ---------------------------------------------------------------------------

def run_walk_forward(
    processed_data : Dict[str, pd.DataFrame],
    portfolio,
    train_period   : int,
    test_period    : int,
    step_size      : Optional[int] = None,
    ranking_metric : str = "sharpe_ratio",
    chunk_size     : int = 250,
    max_workers    : int | None = None,
) -> Dict[str, WalkForwardResult]:
    """
    Run walk-forward validation for all tickers in the portfolio.

    For each ticker:
        1. Slice a training window of `train_period` bars
        2. Optimize on training window (parallel) → find best params
        3. Test on next `test_period` bars using best params
        4. Roll forward by `step_size` bars and repeat

    Args:
        processed_data : dict of ticker → cleaned DataFrame
        portfolio      : Portfolio object
        train_period   : number of bars in each training window
        test_period    : number of bars in each test window
        step_size      : bars to roll forward each iteration
                         defaults to test_period (non-overlapping windows)
        ranking_metric : metric to optimize on in training window
        chunk_size     : param combinations per chunk
        max_workers    : max parallel workers (None = auto)

    Returns:
        Dict[ticker → WalkForwardResult]
    """
    if step_size is None:
        step_size = test_period

    logger.info(
        "run_walk_forward — start | tickers: %s | train: %d | "
        "test: %d | step: %d | metric: %s",
        list(processed_data.keys()),
        train_period,
        test_period,
        step_size,
        ranking_metric,
    )

    results    = {}
    start_time = time.perf_counter()

    for ticker, df in processed_data.items():
        logger.info(
            "%s — walk-forward start | total bars: %d",
            ticker,
            len(df),
        )

        strategy_info = portfolio.strategy_map[ticker]
        strategy_name = strategy_info["name"]
        initial_cash  = get_allocated_cash(portfolio, ticker)

        wf_result = WalkForwardResult(
            ticker         = ticker,
            strategy_name  = strategy_name,
            train_period   = train_period,
            test_period    = test_period,
            step_size      = step_size,
            ranking_metric = ranking_metric,
        )

        # Pre-calculate number of windows for outer progress bar
        total_bars   = len(df)
        n_windows    = 0
        temp_start   = 0
        while temp_start + train_period + test_period <= total_bars:
            n_windows  += 1
            temp_start += step_size

        window_start = 0
        window_index = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        ) as progress:

            outer_task = progress.add_task(
                f"[#2962FF]{ticker} — Walk-Forward Windows",
                total=n_windows,
            )

            inner_task = progress.add_task(
                "[#FF9800]Optimizing window...",
                total=100,
            )

            while True:
                train_start_idx = window_start
                train_end_idx   = window_start + train_period
                test_start_idx  = train_end_idx
                test_end_idx    = test_start_idx + test_period

                if test_end_idx > total_bars:
                    break

                train_df = df.iloc[train_start_idx:train_end_idx].reset_index(drop=True)
                test_df  = df.iloc[test_start_idx:test_end_idx].reset_index(drop=True)

                progress.update(
                    outer_task,
                    description=(
                        f"[#2962FF]{ticker} — Window {window_index + 1}/{n_windows} | "
                        f"Train: {str(train_df['datetime'].iloc[0].date())}→"
                        f"{str(train_df['datetime'].iloc[-1].date())} | "
                        f"Test: {str(test_df['datetime'].iloc[0].date())}→"
                        f"{str(test_df['datetime'].iloc[-1].date())}"
                    )
                )

                progress.update(
                    inner_task,
                    description="[#FF9800]Optimizing training window...",
                )

                logger.info(
                    "%s — window %d | train: %s→%s (%d bars) | "
                    "test: %s→%s (%d bars)",
                    ticker,
                    window_index,
                    train_df["datetime"].iloc[0].date(),
                    train_df["datetime"].iloc[-1].date(),
                    len(train_df),
                    test_df["datetime"].iloc[0].date(),
                    test_df["datetime"].iloc[-1].date(),
                    len(test_df),
                )

                # -----------------------------------------------------------
                # Optimize on training window
                # -----------------------------------------------------------
                try:
                    best_params, best_metric = _optimize_window(
                        df             = train_df,
                        ticker         = ticker,
                        strategy_name  = strategy_name,
                        initial_cash   = initial_cash,
                        commission_rate= portfolio.commission_rate,
                        slippage_rate  = portfolio.slippage_rate,
                        interval       = portfolio.interval,
                        ranking_metric = ranking_metric,
                        chunk_size     = chunk_size,
                        max_workers    = max_workers,
                        progress       = progress,
                        inner_task     = inner_task,
                    )
                except Exception as e:
                    logger.error(
                        "%s — window %d optimization failed: %s",
                        ticker,
                        window_index,
                        e,
                    )
                    window_start += step_size
                    window_index += 1
                    progress.advance(outer_task)
                    continue

                logger.info(
                    "%s — window %d | best train %s: %.4f | params: %s",
                    ticker,
                    window_index,
                    ranking_metric,
                    best_metric,
                    best_params,
                )

                # -----------------------------------------------------------
                # Backtest on test window
                # -----------------------------------------------------------
                progress.update(
                    inner_task,
                    description="[#26A69A]Running test window backtest...",
                )

                try:
                    test_metrics, test_equity = _backtest_window(
                        df             = test_df,
                        ticker         = ticker,
                        strategy_name  = strategy_name,
                        params         = best_params,
                        initial_cash   = initial_cash,
                        commission_rate= portfolio.commission_rate,
                        slippage_rate  = portfolio.slippage_rate,
                        interval       = portfolio.interval,
                    )
                except Exception as e:
                    logger.error(
                        "%s — window %d test backtest failed: %s",
                        ticker,
                        window_index,
                        e,
                    )
                    window_start += step_size
                    window_index += 1
                    progress.advance(outer_task)
                    continue

                logger.info(
                    "%s — window %d | OOS %s: %.4f | OOS return: %.2f%%",
                    ticker,
                    window_index,
                    ranking_metric,
                    test_metrics.get(ranking_metric, 0.0),
                    test_metrics.get("total_return", 0.0) * 100,
                )

                # -----------------------------------------------------------
                # Record window result
                # -----------------------------------------------------------
                window_result = WindowResult(
                    window_index      = window_index,
                    train_start       = train_df["datetime"].iloc[0],
                    train_end         = train_df["datetime"].iloc[-1],
                    test_start        = test_df["datetime"].iloc[0],
                    test_end          = test_df["datetime"].iloc[-1],
                    best_params       = best_params,
                    best_train_metric = best_metric,
                    test_metrics      = test_metrics,
                    test_equity_curve = test_equity,
                    n_train_bars      = len(train_df),
                    n_test_bars       = len(test_df),
                )

                wf_result.windows.append(window_result)
                wf_result.param_stability.append(best_params)

                # Stitch equity curves — normalize each window to start
                # where the previous one ended
                if not wf_result.oos_equity_curve:
                    wf_result.oos_equity_curve.extend(test_equity)
                else:
                    last_equity  = wf_result.oos_equity_curve[-1]
                    first_equity = test_equity[0] if test_equity else initial_cash
                    scale        = last_equity / first_equity if first_equity > 0 else 1.0
                    wf_result.oos_equity_curve.extend(
                        [e * scale for e in test_equity]
                    )

                window_start += step_size
                window_index += 1
                progress.advance(outer_task)

        # -------------------------------------------------------------------
        # Compute aggregate OOS metrics from stitched equity curve
        # -------------------------------------------------------------------
        if wf_result.oos_equity_curve:
            oos_series   = pd.Series(wf_result.oos_equity_curve)
            oos_returns  = oos_series.pct_change().fillna(0)
            synthetic_df = pd.DataFrame({
                "equity_curve"        : oos_series,
                "net_strategy_returns": oos_returns,
            })

            wf_result.oos_metrics = compute_all_metrics(
                synthetic_df,
                interval=portfolio.interval,
            )

            logger.info(
                "%s — walk-forward complete | %d windows | "
                "OOS Sharpe: %.4f | OOS return: %.2f%%",
                ticker,
                len(wf_result.windows),
                wf_result.oos_metrics.get("sharpe_ratio", 0.0),
                wf_result.oos_metrics.get("total_return", 0.0) * 100,
            )
        else:
            logger.warning(
                "%s — walk-forward produced no valid windows",
                ticker,
            )

        wf_result.elapsed_seconds = time.perf_counter() - start_time
        results[ticker] = wf_result

    logger.info(
        "run_walk_forward complete | elapsed: %.2fs",
        time.perf_counter() - start_time,
    )

    return results