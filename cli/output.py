from rich.console import Console

from engine.metrics import compute_all_metrics
from engine.portfolio_builder import get_allocated_cash

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

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

        top_fig = go.Figure()
        for _, row in top_strategies.iterrows():
            if "equity_curve_series" not in row or row["equity_curve_series"] is None:
                continue

            label = _build_param_label(row, param_columns)

            top_fig.add_trace(
                go.Scatter(
                    y=row["equity_curve_series"],
                    mode="lines",
                    name=label,
                )
            )

        top_fig.update_layout(
            title=f"{ticker} Top {top_n} Strategies",
            xaxis_title="Time Step",
            yaxis_title="Equity",
            template="plotly_dark",
        )
        top_fig.show()

        bottom_fig = go.Figure()
        for _, row in bottom_strategies.iterrows():
            if "equity_curve_series" not in row or row["equity_curve_series"] is None:
                continue

            label = _build_param_label(row, param_columns)

            bottom_fig.add_trace(
                go.Scatter(
                    y=row["equity_curve_series"],
                    mode="lines",
                    name=label,
                )
            )

        bottom_fig.update_layout(
            title=f"{ticker} Bottom {bottom_n} Strategies",
            xaxis_title="Time Step",
            yaxis_title="Equity",
            template="plotly_dark",
        )
        bottom_fig.show()

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
            results_df[
                [
                    "datetime",
                    "close",
                    "signal",
                    "position",
                    "returns",
                    "strategy_returns",
                    "transaction_cost",
                    "net_strategy_returns",
                    "equity_curve",
                ]
            ].tail().to_string(index=False)
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
            values=metric,
        )

        fig = go.Figure(
            data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale="Viridis",
                colorbar=dict(title=metric),
            )
        )

        fig.update_layout(
            title=f"{ticker} Optimization Heatmap ({metric})",
            xaxis_title=x_col,
            yaxis_title=y_col,
            template="plotly_dark",
        )

        fig.show()

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

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=results_df["datetime"] if "datetime" in results_df.columns else list(range(len(results_df))),
                y=results_df["equity_curve"],
                mode="lines",
                name="Strategy",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=results_df["datetime"] if "datetime" in results_df.columns else list(range(len(results_df))),
                y=buy_and_hold,
                mode="lines",
                name="Buy & Hold",
            )
        )

        fig.update_layout(
            title=f"{ticker} Backtest: Strategy vs Buy & Hold",
            xaxis_title="Time",
            yaxis_title="Equity",
            template="plotly_dark",
        )

        fig.show()

        running_max = results_df["equity_curve"].cummax()
        drawdown = (results_df["equity_curve"] / running_max) - 1

        dd_fig = go.Figure()
        dd_fig.add_trace(
            go.Scatter(
                x=results_df["datetime"] if "datetime" in results_df.columns else list(range(len(results_df))),
                y=drawdown,
                mode="lines",
                name="Drawdown",
            )
        )

        dd_fig.update_layout(
            title=f"{ticker} Drawdown",
            xaxis_title="Time",
            yaxis_title="Drawdown",
            template="plotly_dark",
        )

        dd_fig.show()

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

        hist_fig = px.histogram(
            x=strategy_returns,
            nbins=30,
            title=f"{ticker} Distribution of Strategy Returns",
            template="plotly_dark",
        )
        hist_fig.update_layout(
            xaxis_title="Return",
            yaxis_title="Frequency",
        )
        hist_fig.show()

        cumulative_returns = (1 + strategy_returns).cumprod()

        cum_fig = go.Figure()
        cum_fig.add_trace(
            go.Scatter(
                x=results_df.loc[strategy_returns.index, "datetime"]
                if "datetime" in results_df.columns
                else list(range(len(cumulative_returns))),
                y=cumulative_returns,
                mode="lines",
                name="Cumulative Returns",
            )
        )

        cum_fig.update_layout(
            title=f"{ticker} Cumulative Strategy Returns",
            xaxis_title="Time",
            yaxis_title="Growth of $1",
            template="plotly_dark",
        )
        cum_fig.show()

        running_max = results_df["equity_curve"].cummax()
        drawdown = (results_df["equity_curve"] / running_max) - 1

        dd_fig = go.Figure()
        dd_fig.add_trace(
            go.Scatter(
                x=results_df["datetime"] if "datetime" in results_df.columns else list(range(len(results_df))),
                y=drawdown,
                mode="lines",
                name="Drawdown",
            )
        )

        dd_fig.update_layout(
            title=f"{ticker} Risk Analysis Drawdown",
            xaxis_title="Time",
            yaxis_title="Drawdown",
            template="plotly_dark",
        )

        dd_fig.show()