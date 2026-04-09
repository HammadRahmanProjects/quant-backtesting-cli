import numpy as np
import pandas as pd

INTERVAL_TO_PERIODS = {
    "1m": 252 * 390,
    "2m": 252 * (390 / 2),
    "5m": 252 * (390 / 5),
    "15m": 252 * (390 / 15),
    "30m": 252 * (390 / 30),
    "60m": 252 * 6.5,
    "1h": 252 * 6.5,
    "90m": 252 * (6.5 / 1.5),
    "1d": 252,
    "5d": 252 / 5,
    "1wk": 52,
    "1mo": 12,
    "3mo": 4,
}

def get_periods_per_year(interval: str) -> float:
    interval = interval.strip().lower()
    return INTERVAL_TO_PERIODS.get(interval, 252)

def calculate_total_return(equity_curve: pd.Series) -> float:
    if len(equity_curve) < 2:
        return 0.0
    return equity_curve.iloc[-1] / equity_curve.iloc[0] - 1

def calculate_cagr(equity_curve: pd.Series, periods_per_year: float) -> float:
    if len(equity_curve) < 2:
        return 0.0

    total_return_multiple = equity_curve.iloc[-1] / equity_curve.iloc[0]
    n_periods = len(equity_curve)
    years = n_periods / periods_per_year

    if years <= 0:
        return 0.0

    return total_return_multiple ** (1 / years) - 1

def calculate_volatility(returns: pd.Series, periods_per_year: float) -> float:
    returns = returns.dropna()
    if len(returns) < 2:
        return 0.0
    return returns.std(ddof=1) * np.sqrt(periods_per_year)

def calculate_sharpe_ratio(
    returns: pd.Series,
    periods_per_year: float,
    risk_free_rate: float = 0.0
) -> float:
    returns = returns.dropna()
    if len(returns) < 2:
        return 0.0

    rf_per_period = risk_free_rate / periods_per_year
    excess_returns = returns - rf_per_period
    std = excess_returns.std(ddof=1)

    if std == 0 or np.isnan(std):
        return 0.0

    return (excess_returns.mean() / std) * np.sqrt(periods_per_year)

def calculate_sortino_ratio(
    returns: pd.Series,
    periods_per_year: float,
    risk_free_rate: float = 0.0
) -> float:
    returns = returns.dropna()
    if len(returns) < 2:
        return 0.0

    rf_per_period = risk_free_rate / periods_per_year
    excess_returns = returns - rf_per_period
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        return 0.0

    downside_std = downside_returns.std(ddof=1)

    if downside_std == 0 or np.isnan(downside_std):
        return 0.0

    return (excess_returns.mean() / downside_std) * np.sqrt(periods_per_year)

def calculate_drawdown_series(equity_curve: pd.Series) -> pd.Series:
    rolling_peak = equity_curve.cummax()
    return (equity_curve - rolling_peak) / rolling_peak

def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    if len(equity_curve) < 2:
        return 0.0
    drawdown = calculate_drawdown_series(equity_curve)
    return drawdown.min()

def calculate_calmar_ratio(cagr: float, max_drawdown: float) -> float:
    if max_drawdown == 0:
        return 0.0
    return cagr / abs(max_drawdown)

def compute_all_metrics(
    results_df: pd.DataFrame,
    interval: str,
    risk_free_rate: float = 0.0
) -> dict:
    equity_curve = results_df["equity_curve"]
    returns = results_df["net_strategy_returns"]

    periods_per_year = get_periods_per_year(interval)

    total_return = calculate_total_return(equity_curve)
    cagr = calculate_cagr(equity_curve, periods_per_year)
    volatility = calculate_volatility(returns, periods_per_year)
    sharpe = calculate_sharpe_ratio(returns, periods_per_year, risk_free_rate)
    sortino = calculate_sortino_ratio(returns, periods_per_year, risk_free_rate)
    max_drawdown = calculate_max_drawdown(equity_curve)
    calmar = calculate_calmar_ratio(cagr, max_drawdown)

    return {
        "total_return": total_return,
        "cagr": cagr,
        "volatility": volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
    }