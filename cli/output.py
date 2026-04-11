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

        # Aggregate duplicate (x, y) combinations before pivoting
        # This happens when the grid has more than 2 params — we average
        # the metric across all combinations that share the same x/y values
        heatmap_df = (
            results_table
            .groupby([y_col, x_col], as_index=False)[metric]
            .mean()
        )

        heatmap_data = heatmap_df.pivot(
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

def print_walk_forward_results(wf_results):
    """Print walk-forward validation summary to terminal."""
    if not wf_results:
        console.print("[#FF9800]No walk-forward results to display.[/#FF9800]")
        return

    console.print("\n[#2962FF]Walk-Forward Validation Results[/#2962FF]")

    for ticker, wf_result in wf_results.items():
        console.print(f"\n[bold]{ticker}[/bold]")
        console.print(f"Strategy:      {wf_result.strategy_name}")
        console.print(f"Train period:  {wf_result.train_period} bars")
        console.print(f"Test period:   {wf_result.test_period} bars")
        console.print(f"Step size:     {wf_result.step_size} bars")
        console.print(f"Windows:       {len(wf_result.windows)}")
        console.print(f"Elapsed:       {wf_result.elapsed_seconds:.1f}s")

        if wf_result.oos_metrics:
            console.print("\n[#26A69A]Out-of-Sample Metrics[/#26A69A]")
            console.print(f"Total Return:  {wf_result.oos_metrics.get('total_return', 0) * 100:.2f}%")
            console.print(f"CAGR:          {wf_result.oos_metrics.get('cagr', 0) * 100:.2f}%")
            console.print(f"Sharpe Ratio:  {wf_result.oos_metrics.get('sharpe_ratio', 0):.4f}")
            console.print(f"Max Drawdown:  {wf_result.oos_metrics.get('max_drawdown', 0) * 100:.2f}%")
            console.print(f"Calmar Ratio:  {wf_result.oos_metrics.get('calmar_ratio', 0):.4f}")

        console.print("\n[#AB47BC]Per-Window Results[/#AB47BC]")

        for w in wf_result.windows:
            # Clean numpy types from params for display
            clean_params = {
                k: v.item() if hasattr(v, "item") else v
                for k, v in w.best_params.items()
            }
            console.print(
                f"  Window {w.window_index} | "
                f"Train: {str(w.train_start.date())}→{str(w.train_end.date())} | "
                f"Test: {str(w.test_start.date())}→{str(w.test_end.date())} | "
                f"Train {wf_result.ranking_metric}: {w.best_train_metric:.4f} | "
                f"OOS {wf_result.ranking_metric}: {w.test_metrics.get(wf_result.ranking_metric, 0):.4f} | "
                f"Params: {clean_params}"
            )

def plot_walk_forward_results(wf_results, portfolio):
    """Plot walk-forward equity curve and parameter stability."""
    if not wf_results:
        console.print("[#FF9800]No walk-forward results to plot.[/#FF9800]")
        return

    for ticker, wf_result in wf_results.items():
        if not wf_result.oos_equity_curve:
            console.print(f"[red]{ticker}: No OOS equity curve to plot.[/red]")
            continue

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            y    = wf_result.oos_equity_curve,
            mode = "lines",
            name = "Walk-Forward OOS",
            line = dict(color="#2962FF", width=2),
        ))

        cumulative_bars = 0
        for w in wf_result.windows:
            fig.add_vline(
                x           = cumulative_bars,
                line_dash   = "dash",
                line_color  = "#FF9800",
                opacity     = 0.5,
                annotation_text = f"W{w.window_index}",
            )
            cumulative_bars += w.n_test_bars

        fig.update_layout(
            title        = f"{ticker} Walk-Forward Out-of-Sample Equity Curve",
            xaxis_title  = "Bar",
            yaxis_title  = "Equity",
            template     = "plotly_dark",
        )
        fig.show()

        if wf_result.param_stability and len(wf_result.windows) > 1:
            param_names = list(wf_result.param_stability[0].keys())
            stability_fig = go.Figure()

            for param in param_names:
                values = [p.get(param, 0) for p in wf_result.param_stability]
                stability_fig.add_trace(go.Scatter(
                    x    = list(range(len(values))),
                    y    = values,
                    mode = "lines+markers",
                    name = param,
                ))

            stability_fig.update_layout(
                title       = f"{ticker} Parameter Stability Across Windows",
                xaxis_title = "Window",
                yaxis_title = "Parameter Value",
                template    = "plotly_dark",
            )
            stability_fig.show()

def print_monte_carlo_results(mc_results):
    if not mc_results:
        console.print("[#FF9800]No Monte Carlo results to display.[/#FF9800]")
        return

    console.print("\n[#2962FF]Monte Carlo Simulation Results[/#2962FF]")

    for ticker, mc in mc_results.items():
        console.print(f"\n[bold]{ticker}[/bold]")
        console.print(f"Simulations:        {mc.n_simulations:,}")
        console.print(f"Confidence level:   {mc.confidence * 100:.0f}%")
        console.print(f"Active bars:        {mc.n_active_bars:,}")

        if mc.n_active_bars == 0:
            console.print(
                "[#FF9800]Insufficient active bars — "
                "run a backtest with more frequent signals first.[/#FF9800]"
            )
            continue

        console.print(f"\n[#26A69A]Observed vs Simulated[/#26A69A]")
        console.print(f"Observed Sharpe:    {mc.observed_sharpe:.4f}")
        console.print(f"Simulated Sharpe:   [{mc.sharpe_ci_lower:.4f}, {mc.sharpe_ci_upper:.4f}]")
        console.print(f"Observed Max DD:    {mc.observed_max_dd * 100:.2f}%")
        console.print(f"Simulated Max DD:   [{mc.max_dd_ci_lower * 100:.2f}%, {mc.max_dd_ci_upper * 100:.2f}%]")

        console.print(f"\n[#AB47BC]Edge Assessment[/#AB47BC]")
        console.print(f"P(genuine edge):    {mc.prob_outperformance * 100:.1f}%")

        if mc.prob_outperformance >= 0.95:
            console.print("[#26A69A]Strong evidence of genuine edge[/#26A69A]")
        elif mc.prob_outperformance >= 0.80:
            console.print("[#FF9800]Moderate evidence of edge — use caution[/#FF9800]")
        else:
            console.print("[red]Weak evidence of edge — likely noise[/red]")

def plot_monte_carlo_results(mc_results):
    """Plot Monte Carlo distributions."""
    if not mc_results:
        console.print("[#FF9800]No Monte Carlo results to plot.[/#FF9800]")
        return

    for ticker, mc in mc_results.items():
        if not mc.sharpe_distribution:
            continue

        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x       = mc.sharpe_distribution,
            nbinsx  = 100,
            name    = "Simulated Sharpe",
            opacity = 0.7,
        ))

        fig.add_vline(
            x                = mc.observed_sharpe,
            line_dash        = "solid",
            line_color       = "#26A69A",
            line_width       = 2,
            annotation_text  = f"Observed: {mc.observed_sharpe:.4f}",
            annotation_position = "top right",
        )

        fig.add_vline(
            x               = mc.sharpe_ci_lower,
            line_dash       = "dash",
            line_color      = "#FF9800",
            annotation_text = f"{mc.confidence*100:.0f}% CI lower",
        )

        fig.add_vline(
            x               = mc.sharpe_ci_upper,
            line_dash       = "dash",
            line_color      = "#FF9800",
            annotation_text = f"{mc.confidence*100:.0f}% CI upper",
        )

        fig.update_layout(
            title       = f"{ticker} Monte Carlo — Sharpe Ratio Distribution ({mc.n_simulations:,} simulations)",
            xaxis_title = "Sharpe Ratio",
            yaxis_title = "Frequency",
            template    = "plotly_dark",
        )
        fig.show()

        if mc.sample_equity_curves:
            eq_fig = go.Figure()

            for i, curve in enumerate(mc.sample_equity_curves[:100]):
                eq_fig.add_trace(go.Scatter(
                    y       = curve,
                    mode    = "lines",
                    opacity = 0.1,
                    line    = dict(color="#2962FF", width=1),
                    showlegend = False,
                ))

            eq_fig.update_layout(
                title       = f"{ticker} Monte Carlo — Sample Equity Curves",
                xaxis_title = "Bar",
                yaxis_title = "Equity",
                template    = "plotly_dark",
            )
            eq_fig.show()