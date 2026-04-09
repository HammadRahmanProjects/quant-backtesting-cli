from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from typing import Any, Dict, List

import pandas as pd
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from engine.backtester import Backtester
from engine.metrics import compute_all_metrics
from engine.portfolio_builder import get_allocated_cash
from strategies.registry import AVAILABLE_STRATEGIES

console = Console()

def _build_param_combinations(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(param_grid.keys())
    values = [param_grid[key] for key in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]

def _chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

def _is_valid_param_combo(
    params: Dict[str, Any],
    param_names: List[str],
    data_length: int,
) -> bool:
    if "short_window" in param_names and "long_window" in param_names:
        short_w = params["short_window"]
        long_w = params["long_window"]

        if short_w >= long_w:
            return False

        if long_w >= data_length:
            return False

    if "window" in param_names:
        if params["window"] >= data_length:
            return False

    return True

def _evaluate_param_chunk(
    ticker: str,
    df: pd.DataFrame,
    strategy_name: str,
    params_chunk: List[Dict[str, Any]],
    initial_cash: float,
    commission_rate: float,
    slippage_rate: float,
    interval: str,
) -> List[Dict[str, Any]]:
    strategy_class = AVAILABLE_STRATEGIES.get(strategy_name)

    if strategy_class is None:
        raise ValueError(f"{ticker}: Unknown strategy '{strategy_name}'.")

    chunk_results = []

    for params in params_chunk:
        strategy = strategy_class(df, **params)
        signals_df = strategy.generate_signals()

        backtester = Backtester(
            signals_df,
            initial_cash=initial_cash,
            commission_rate=commission_rate,
            slippage_rate=slippage_rate,
        )

        results_df = backtester.run()
        metrics = compute_all_metrics(results_df, interval=interval)

        chunk_results.append(
            {
                "ticker": ticker,
                "strategy": strategy_name,
                **params,
                **metrics,
                "equity_curve_series": results_df["equity_curve"].tolist(),
            }
        )

    return chunk_results

def optimize_portfolio(
    processed_data: Dict[str, pd.DataFrame],
    portfolio,
    ranking_metric: str = "sharpe_ratio",
    max_workers: int | None = None,
    chunk_size: int = 250,
) -> Dict[str, pd.DataFrame]:
    optimization_results = {}
    jobs = []

    for ticker, df in processed_data.items():
        strategy_info = portfolio.strategy_map[ticker]
        strategy_name = strategy_info["name"]

        strategy_class = AVAILABLE_STRATEGIES.get(strategy_name)

        if strategy_class is None:
            raise ValueError(f"{ticker}: Unknown strategy '{strategy_name}'.")

        if not hasattr(strategy_class, "get_optimization_grid"):
            raise ValueError(
                f"{ticker}: Strategy '{strategy_name}' does not define get_optimization_grid()."
            )

        if not hasattr(strategy_class, "get_param_names"):
            raise ValueError(
                f"{ticker}: Strategy '{strategy_name}' does not define get_param_names()."
            )

        param_grid = strategy_class.get_optimization_grid()
        param_names = strategy_class.get_param_names()
        param_combinations = _build_param_combinations(param_grid)

        valid_param_combinations = [
            params
            for params in param_combinations
            if _is_valid_param_combo(params, param_names, len(df))
        ]

        if not valid_param_combinations:
            continue

        initial_cash = get_allocated_cash(portfolio, ticker)
        param_chunks = _chunk_list(valid_param_combinations, chunk_size)

        for params_chunk in param_chunks:
            jobs.append(
                {
                    "ticker": ticker,
                    "df": df,
                    "strategy_name": strategy_name,
                    "params_chunk": params_chunk,
                    "initial_cash": initial_cash,
                    "commission_rate": portfolio.commission_rate,
                    "slippage_rate": portfolio.slippage_rate,
                    "interval": portfolio.interval,
                }
            )

    if not jobs:
        raise ValueError("No valid optimization jobs were generated.")

    results_by_ticker = {ticker: [] for ticker in processed_data.keys()}

    total_param_combos = sum(len(job["params_chunk"]) for job in jobs)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running optimization...", total=total_param_combos)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _evaluate_param_chunk,
                    job["ticker"],
                    job["df"],
                    job["strategy_name"],
                    job["params_chunk"],
                    job["initial_cash"],
                    job["commission_rate"],
                    job["slippage_rate"],
                    job["interval"],
                )
                for job in jobs
            ]

            for future, job in zip(as_completed(futures), [None] * len(futures)):
                chunk_rows = future.result()

                if chunk_rows:
                    ticker = chunk_rows[0]["ticker"]
                    results_by_ticker[ticker].extend(chunk_rows)
                    progress.advance(task, advance=len(chunk_rows))

    for ticker, ticker_results in results_by_ticker.items():
        results_table = pd.DataFrame(ticker_results)

        if results_table.empty:
            raise ValueError(f"{ticker}: No optimization results were generated.")

        if ranking_metric not in results_table.columns:
            raise ValueError(f"Invalid ranking metric: {ranking_metric}")

        results_table = results_table.sort_values(
            by=ranking_metric,
            ascending=False,
        ).reset_index(drop=True)

        optimization_results[ticker] = results_table

    return optimization_results