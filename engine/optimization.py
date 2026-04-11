import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from engine.portfolio_builder import get_allocated_cash
from strategies.registry import AVAILABLE_STRATEGIES

console = Console()
logger  = logging.getLogger(__name__)


def _run_numpy_chunk(
    close_prices    : np.ndarray,
    volume          : np.ndarray,
    ticker          : str,
    strategy_class,
    param_chunk     : List[Dict[str, Any]],
    initial_cash    : float,
    commission_rate : float,
    slippage_rate   : float,
    interval        : str,
    batch_size      : int,
) -> List[Dict[str, Any]]:
    from engine.numpy_backtester import run_numpy_optimization

    return run_numpy_optimization(
        close_prices       = close_prices,
        volume             = volume,
        ticker             = ticker,
        strategy_class     = strategy_class,
        param_combinations = param_chunk,
        initial_cash       = initial_cash,
        commission_rate    = commission_rate,
        slippage_rate      = slippage_rate,
        interval           = interval,
        batch_size         = batch_size,
        progress           = None,
        task               = None,
    )


def _build_param_combinations(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys   = list(param_grid.keys())
    values = [param_grid[key] for key in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def _is_valid_param_combo(
    params      : Dict[str, Any],
    param_names : List[str],
    data_length : int,
) -> bool:
    if "short_window" in param_names and "long_window" in param_names:
        if params["short_window"] >= params["long_window"]:
            return False
        if params["long_window"] >= data_length:
            return False

    if "window" in param_names:
        if params["window"] >= data_length:
            return False

    if "lower_threshold" in param_names and "upper_threshold" in param_names:
        if params["lower_threshold"] >= params["upper_threshold"]:
            return False

    if "volume_window" in param_names:
        if params["volume_window"] >= data_length:
            return False

    return True


def optimize_portfolio(
    processed_data : Dict[str, pd.DataFrame],
    portfolio,
    ranking_metric : str = "sharpe_ratio",
    batch_size     : int = 5000,
    max_workers    : int | None = None,
    chunk_size     : int = 5000,
) -> Dict[str, pd.DataFrame]:
    logger.info(
        "optimize_portfolio — start | tickers: %s | metric: %s | "
        "batch: %d | chunk: %d | workers: %s",
        list(processed_data.keys()),
        ranking_metric,
        batch_size,
        chunk_size,
        max_workers or "auto",
    )

    optimization_results = {}
    start_time           = time.perf_counter()

    for ticker, df in processed_data.items():
        strategy_info  = portfolio.strategy_map[ticker]
        strategy_name  = strategy_info["name"]
        strategy_class = AVAILABLE_STRATEGIES.get(strategy_name)

        if strategy_class is None:
            logger.error("%s — unknown strategy '%s'", ticker, strategy_name)
            raise ValueError(f"{ticker}: Unknown strategy '{strategy_name}'.")

        if not hasattr(strategy_class, "generate_signals_numpy"):
            raise ValueError(
                f"{ticker}: Strategy '{strategy_name}' does not implement "
                f"generate_signals_numpy()."
            )

        param_grid         = strategy_class.get_optimization_grid()
        param_names        = strategy_class.get_param_names()
        param_combinations = _build_param_combinations(param_grid)

        valid_combinations = [
            p for p in param_combinations
            if _is_valid_param_combo(p, param_names, len(df))
        ]

        n_total  = len(param_combinations)
        n_valid  = len(valid_combinations)
        n_pruned = n_total - n_valid

        logger.info(
            "%s — strategy: %s | total: %d | valid: %d | pruned: %d",
            ticker,
            strategy_name,
            n_total,
            n_valid,
            n_pruned,
        )

        if not valid_combinations:
            logger.warning("%s — no valid parameter combinations", ticker)
            continue

        initial_cash = get_allocated_cash(portfolio, ticker)

        close_prices = df["close"].values.astype(np.float64)
        volume       = (
            df["volume"].values.astype(np.float64)
            if "volume" in df.columns
            else None
        )

        combo_chunks = [
            valid_combinations[i:i + chunk_size]
            for i in range(0, n_valid, chunk_size)
        ]

        logger.info(
            "%s — dispatching %d chunks | %d combos per chunk",
            ticker,
            len(combo_chunks),
            chunk_size,
        )

        ticker_start   = time.perf_counter()
        ticker_results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"[#2962FF]Optimizing {ticker} — {strategy_name}",
                total=n_valid,
            )

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_chunk = {
                    executor.submit(
                        _run_numpy_chunk,
                        close_prices,
                        volume,
                        ticker,
                        strategy_class,
                        chunk,
                        initial_cash,
                        portfolio.commission_rate,
                        portfolio.slippage_rate,
                        portfolio.interval,
                        batch_size,
                    ): chunk
                    for chunk in combo_chunks
                }

                for future in as_completed(future_to_chunk):
                    try:
                        chunk_results = future.result()
                        ticker_results.extend(chunk_results)
                        progress.advance(task, advance=len(chunk_results))
                        logger.debug(
                            "%s — chunk complete | %d results",
                            ticker,
                            len(chunk_results),
                        )
                    except Exception as e:
                        logger.error(
                            "%s — chunk failed: %s",
                            ticker,
                            e,
                            exc_info=True,
                        )
                        raise

        ticker_elapsed = time.perf_counter() - ticker_start
        combos_per_sec = n_valid / ticker_elapsed if ticker_elapsed > 0 else 0

        logger.info(
            "%s — optimization complete | elapsed: %.2fs | %.0f combos/sec",
            ticker,
            ticker_elapsed,
            combos_per_sec,
        )

        if not ticker_results:
            raise ValueError(f"{ticker}: No optimization results generated.")

        results_table = pd.DataFrame(ticker_results)

        if ranking_metric not in results_table.columns:
            raise ValueError(f"Invalid ranking metric: '{ranking_metric}'")

        results_table = results_table.sort_values(
            by        = ranking_metric,
            ascending = False,
        ).reset_index(drop=True)

        best = results_table.iloc[0]
        logger.info(
            "%s — best %s: %.4f | params: %s",
            ticker,
            ranking_metric,
            float(best[ranking_metric]),
            {k: best[k] for k in param_names},
        )

        optimization_results[ticker] = results_table

    total_elapsed = time.perf_counter() - start_time

    logger.info(
        "optimize_portfolio complete | %d ticker(s) | total elapsed: %.2fs",
        len(optimization_results),
        total_elapsed,
    )

    return optimization_results