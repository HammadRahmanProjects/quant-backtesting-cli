from rich.console import Console

from engine.metrics import compute_all_metrics
from engine.portfolio_builder import get_allocated_cash

import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

console = Console()

def _get_optimization_param_columns(results_table):
    excluded_columns = {
        "ticker",
        "strategy",
        "total_return",
        "cagr",
        "volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "calmar_ratio",
        "equity_curve_series",
    }

    return [col for col in results_table.columns if col not in excluded_columns]

def _build_param_label(row, param_columns):
    label_parts = []

    for col in param_columns:
        value = row[col]

        if pd.isna(value):
            continue

        if isinstance(value, float):
            if float(value).is_integer():
                label_parts.append(f"{col}={int(value)}")
            else:
                label_parts.append(f"{col}={value:.2f}")
        else:
            label_parts.append(f"{col}={value}")

    return ", ".join(label_parts) if label_parts else "strategy"

def plot_optimization_curves(optimization_results, top_n=10, bottom_n=10):
    if not optimization_results:
        console.print("[#FF9800]No optimization results to plot.[/#FF9800]")
        return

    for ticker, results_table in optimization_results.items():
        if results_table is None or results_table.empty:
            console.print(f"[red]{ticker}: No optimization data available.[/red]")
            continue

        param_columns = _get_optimization_param_columns(results_table)

        top_strategies = results_table.head(top_n)
        bottom_strategies = results_table.tail(bottom_n)

        plt.figure(figsize=(12, 6))
        for _, row in top_strategies.iterrows():
            if "equity_curve_series" not in row or row["equity_curve_series"] is None:
                continue

            label = _build_param_label(row, param_columns)

            plt.plot(
                row["equity_curve_series"],
                label=label,
                alpha=0.7
            )

        plt.title(f"{ticker} Top {top_n} Strategies")
        plt.xlabel("Time Step")
        plt.ylabel("Equity")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

        plt.figure(figsize=(12, 6))
        for _, row in bottom_strategies.iterrows():
            if "equity_curve_series" not in row or row["equity_curve_series"] is None:
                continue

            label = _build_param_label(row, param_columns)

            plt.plot(
                row["equity_curve_series"],
                label=label,
                alpha=0.7
            )

        plt.title(f"{ticker} Bottom {bottom_n} Strategies")
        plt.xlabel("Time Step")
        plt.ylabel("Equity")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

def print_backtest_results(backtest_results, portfolio):
    if not backtest_results:
        console.print("[#FF9800]No backtest results to display.[/#FF9800]")
        return

    console.print("\n[#2962FF]Backtest Results[/#2962FF]")

    for ticker, results_df in backtest_results.items():
        if results_df is None or results_df.empty:
            console.print(f"[red]{ticker}: No data available.[/red]")
            continue

        allocated_cash = get_allocated_cash(portfolio, ticker)
        weight = portfolio.weights[portfolio.tickers.index(ticker)]
        final_equity = results_df["equity_curve"].iloc[-1]
        pnl = final_equity - allocated_cash
        total_return = (final_equity / allocated_cash - 1) * 100

        console.print(f"\n[bold]{ticker}[/bold]")
        console.print(f"Weight: {weight}")
        console.print(f"Allocated Cash: {allocated_cash:,.2f}")
        console.print(f"Final Equity: {final_equity:,.2f}")
        console.print(f"PnL: {pnl:,.2f}")
        console.print(f"Total Return: {total_return:.2f}%")
        console.print(f"First Date: {results_df['datetime'].iloc[0]}")
        console.print(f"Last Date: {results_df['datetime'].iloc[-1]}")
        console.print(f"Rows: {len(results_df)}")

        console.print("\nSignal counts:")
        console.print(results_df["signal"].value_counts(dropna=False).to_string())

        console.print("\nPosition counts:")
        console.print(results_df["position"].value_counts(dropna=False).to_string())

        console.print("\nLast 5 rows:")
        console.print(
            results_df[[
                "datetime",
                "close",
                "signal",
                "position",
                "returns",
                "strategy_returns",
                "transaction_cost",
                "net_strategy_returns",
                "equity_curve"
            ]].tail().to_string(index=False)
        )

def print_risk_analysis(backtest_results, portfolio):
    if not backtest_results:
        console.print("[#FF9800]No backtest results to analyze.[/#FF9800]")
        return

    console.print("\n[#AB47BC]Risk Analysis[/#AB47BC]")

    for ticker, results_df in backtest_results.items():
        if results_df is None or results_df.empty:
            console.print(f"[red]{ticker}: No data available.[/red]")
            continue

        metrics = compute_all_metrics(results_df, interval=portfolio.interval)

        console.print(f"\n[bold]{ticker}[/bold]")
        console.print(f"Total Return: {metrics['total_return'] * 100:.2f}%")
        console.print(f"CAGR: {metrics['cagr'] * 100:.2f}%")
        console.print(f"Volatility: {metrics['volatility'] * 100:.2f}%")
        console.print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        console.print(f"Sortino Ratio: {metrics['sortino_ratio']:.4f}")
        console.print(f"Max Drawdown: {metrics['max_drawdown'] * 100:.2f}%")
        console.print(f"Calmar Ratio: {metrics['calmar_ratio']:.4f}")

def print_optimization_results(optimization_results):
    if not optimization_results:
        console.print("[#FF9800]No optimization results to display.[/#FF9800]")
        return

    console.print("\n[#2962FF]Optimization Results[/#2962FF]")

    for ticker, results_table in optimization_results.items():
        if results_table is None or results_table.empty:
            console.print(f"[red]{ticker}: No optimization data available.[/red]")
            continue

        console.print(f"\n[bold]{ticker}[/bold]")
        console.print(f"Tested parameter combinations: {len(results_table)}")

        display_table = results_table.drop(columns=["equity_curve_series"], errors="ignore")

        console.print("\nTop 10 parameter sets:")
        console.print(display_table.head(10).to_string(index=False))

def plot_optimization_heatmap(optimization_results, metric="sharpe_ratio"):
    if not optimization_results:
        console.print("[#FF9800]No optimization results to plot.[/#FF9800]")
        return

    for ticker, results_table in optimization_results.items():
        if results_table is None or results_table.empty:
            console.print(f"[red]{ticker}: No optimization data available.[/red]")
            continue

        param_columns = _get_optimization_param_columns(results_table)

        if len(param_columns) < 2:
            console.print(f"[red]{ticker}: Not enough parameters for heatmap.[/red]")
            continue

        x_col = param_columns[0]
        y_col = param_columns[1]

        if metric not in results_table.columns:
            console.print(f"[red]{ticker}: Metric '{metric}' not found.[/red]")
            continue

        heatmap_data = results_table.pivot(
            index=y_col,
            columns=x_col,
            values=metric
        )

        plt.figure(figsize=(12, 8))
        plt.imshow(
            heatmap_data,
            aspect="auto",
            origin="lower",
            interpolation="nearest"
        )
        plt.colorbar(label=metric)
        plt.title(f"{ticker} Optimization Heatmap ({metric})")
        plt.xlabel(x_col)
        plt.ylabel(y_col)

        x_step = max(1, len(heatmap_data.columns) // 10)
        y_step = max(1, len(heatmap_data.index) // 10)

        x_ticks = range(0, len(heatmap_data.columns), x_step)
        y_ticks = range(0, len(heatmap_data.index), y_step)

        plt.xticks(
            x_ticks,
            [heatmap_data.columns[i] for i in x_ticks],
            rotation=45
        )
        plt.yticks(
            y_ticks,
            [heatmap_data.index[i] for i in y_ticks]
        )

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

def plot_backtest_visuals(backtest_results, portfolio):
    if not backtest_results:
        console.print("[#FF9800]No backtest results to plot.[/#FF9800]")
        return

    for ticker, results_df in backtest_results.items():
        if results_df is None or results_df.empty:
            console.print(f"[red]{ticker}: No backtest data available.[/red]")
            continue

        if "equity_curve" not in results_df.columns or "close" not in results_df.columns:
            console.print(f"[red]{ticker}: Missing required columns for backtest plot.[/red]")
            continue

        allocated_cash = get_allocated_cash(portfolio, ticker)

        buy_and_hold = (results_df["close"] / results_df["close"].iloc[0]) * allocated_cash

        plt.figure(figsize=(12, 6))
        plt.plot(results_df["equity_curve"], label="Strategy")
        plt.plot(buy_and_hold, label="Buy & Hold", alpha=0.8)
        plt.title(f"{ticker} Backtest: Strategy vs Buy & Hold")
        plt.xlabel("Time Step")
        plt.ylabel("Equity")
        plt.legend()
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

        running_max = results_df["equity_curve"].cummax()
        drawdown = (results_df["equity_curve"] / running_max) - 1

        plt.figure(figsize=(12, 4))
        plt.plot(drawdown)
        plt.title(f"{ticker} Drawdown")
        plt.xlabel("Time Step")
        plt.ylabel("Drawdown")
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

def plot_risk_visuals(backtest_results, portfolio):
    if not backtest_results:
        console.print("[#FF9800]No backtest results to analyze.[/#FF9800]")
        return

    for ticker, results_df in backtest_results.items():
        if results_df is None or results_df.empty:
            console.print(f"[red]{ticker}: No risk data available.[/red]")
            continue

        if "net_strategy_returns" not in results_df.columns or "equity_curve" not in results_df.columns:
            console.print(f"[red]{ticker}: Missing required columns for risk plots.[/red]")
            continue

        strategy_returns = results_df["net_strategy_returns"].dropna()

        plt.figure(figsize=(12, 4))
        plt.hist(strategy_returns, bins=30)
        plt.title(f"{ticker} Distribution of Strategy Returns")
        plt.xlabel("Return")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

        cumulative_returns = (1 + strategy_returns).cumprod()

        plt.figure(figsize=(12, 4))
        plt.plot(cumulative_returns)
        plt.title(f"{ticker} Cumulative Strategy Returns")
        plt.xlabel("Time Step")
        plt.ylabel("Growth of $1")
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

        running_max = results_df["equity_curve"].cummax()
        drawdown = (results_df["equity_curve"] / running_max) - 1

        plt.figure(figsize=(12, 4))
        plt.plot(drawdown)
        plt.title(f"{ticker} Risk Analysis Drawdown")
        plt.xlabel("Time Step")
        plt.ylabel("Drawdown")
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)