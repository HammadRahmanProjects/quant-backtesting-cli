import logging
from typing import Any, Dict, List, Optional

import numpy as np
from numba import jit, prange

logger = logging.getLogger(__name__)


@jit(nopython=True, cache=True)
def _rolling_mean_vectorized(arr, windows):
    n_bars    = len(arr)
    n_windows = len(windows)
    result    = np.full((n_windows, n_bars), np.nan)
    cumsum    = np.cumsum(arr)

    for i in range(n_windows):
        w = int(windows[i])
        if w > n_bars:
            continue
        padded     = np.empty(n_bars - w + 1)
        padded[0]  = cumsum[w - 1]
        for j in range(1, n_bars - w + 1):
            padded[j] = cumsum[w + j - 1] - cumsum[j - 1]
        for j in range(n_bars - w + 1):
            result[i, w - 1 + j] = padded[j] / w

    return result


@jit(nopython=True, cache=True)
def _rolling_std_vectorized(arr, windows, ddof=1):
    n_bars    = len(arr)
    n_windows = len(windows)
    result    = np.full((n_windows, n_bars), np.nan)
    cumsum    = np.cumsum(arr)
    cumsum_sq = np.cumsum(arr ** 2)

    for i in range(n_windows):
        w = int(windows[i])
        if w < 2 or w > n_bars:
            continue

        for j in range(n_bars - w + 1):
            if j == 0:
                sx  = cumsum[w - 1]
                ssq = cumsum_sq[w - 1]
            else:
                sx  = cumsum[w + j - 1] - cumsum[j - 1]
                ssq = cumsum_sq[w + j - 1] - cumsum_sq[j - 1]

            variance = (ssq - sx * sx / w) / (w - ddof)
            if variance < 0.0:
                variance = 0.0
            result[i, w - 1 + j] = variance ** 0.5

    return result


@jit(nopython=True, cache=True)
def _rolling_mean_single(arr, window):
    n_bars = len(arr)
    result = np.full(n_bars, np.nan)

    if window > n_bars:
        return result

    cumsum = np.cumsum(arr)
    result[window - 1] = cumsum[window - 1] / window

    for i in range(1, n_bars - window + 1):
        result[window - 1 + i] = (cumsum[window + i - 1] - cumsum[i - 1]) / window

    return result


@jit(nopython=True, cache=True, parallel=True)
def _compute_equity_curves_jit(
    signal_matrix   : np.ndarray,
    close_prices    : np.ndarray,
    initial_cash    : float,
    commission_rate : float,
    slippage_rate   : float,
) -> tuple:
    n_combos = signal_matrix.shape[0]
    n_bars   = signal_matrix.shape[1]

    bar_returns = np.zeros(n_bars)
    for i in range(1, n_bars):
        denom = close_prices[i - 1]
        if denom > 1e-10:
            bar_returns[i] = (close_prices[i] - denom) / denom

    position_matrix = np.zeros((n_combos, n_bars))
    trade_matrix    = np.zeros((n_combos, n_bars))
    net_returns     = np.zeros((n_combos, n_bars))
    equity_matrix   = np.zeros((n_combos, n_bars))

    cost_rate = commission_rate + slippage_rate

    for c in prange(n_combos):
        for b in range(1, n_bars):
            position_matrix[c, b] = signal_matrix[c, b - 1]

        for b in range(1, n_bars):
            diff = position_matrix[c, b] - position_matrix[c, b - 1]
            if diff < 0.0:
                trade_matrix[c, b] = -diff
            else:
                trade_matrix[c, b] = diff

        cumprod = 1.0
        for b in range(n_bars):
            sr = position_matrix[c, b] * bar_returns[b]
            tc = trade_matrix[c, b] * cost_rate
            nr = sr - tc
            net_returns[c, b]   = nr
            cumprod            *= (1.0 + nr)
            equity_matrix[c, b] = initial_cash * cumprod

    return equity_matrix, net_returns


@jit(nopython=True, cache=True, parallel=True)
def _compute_metrics_jit(
    equity_matrix   : np.ndarray,
    net_returns     : np.ndarray,
    periods_per_year: float,
    initial_cash    : float,
) -> tuple:
    n_combos = equity_matrix.shape[0]
    n_bars   = equity_matrix.shape[1]
    eps      = 1e-10
    years    = n_bars / periods_per_year

    total_return = np.zeros(n_combos)
    cagr         = np.zeros(n_combos)
    volatility   = np.zeros(n_combos)
    sharpe       = np.zeros(n_combos)
    sortino      = np.zeros(n_combos)
    max_drawdown = np.zeros(n_combos)
    calmar       = np.zeros(n_combos)

    for c in prange(n_combos):
        final_equity    = equity_matrix[c, n_bars - 1]
        total_return[c] = final_equity / initial_cash - 1.0

        ratio = final_equity / initial_cash
        if ratio < eps:
            ratio = eps
        if years > 0:
            cagr[c] = ratio ** (1.0 / years) - 1.0

        mean_r = 0.0
        for b in range(n_bars):
            mean_r += net_returns[c, b]
        mean_r /= n_bars

        var_r = 0.0
        for b in range(n_bars):
            diff = net_returns[c, b] - mean_r
            var_r += diff * diff
        if n_bars > 1:
            var_r /= (n_bars - 1)

        std_r         = var_r ** 0.5
        volatility[c] = std_r * (periods_per_year ** 0.5)

        if std_r > eps:
            sharpe[c] = (mean_r / std_r) * (periods_per_year ** 0.5)

        down_mean = 0.0
        for b in range(n_bars):
            if net_returns[c, b] < 0.0:
                down_mean += net_returns[c, b]
        down_mean /= n_bars

        down_var = 0.0
        for b in range(n_bars):
            r    = net_returns[c, b] if net_returns[c, b] < 0.0 else 0.0
            diff = r - down_mean
            down_var += diff * diff
        if n_bars > 1:
            down_var /= (n_bars - 1)

        down_std = down_var ** 0.5
        if down_std > eps:
            sortino[c] = (mean_r / down_std) * (periods_per_year ** 0.5)

        peak   = equity_matrix[c, 0]
        min_dd = 0.0
        for b in range(n_bars):
            if equity_matrix[c, b] > peak:
                peak = equity_matrix[c, b]
            dd = (equity_matrix[c, b] - peak) / (peak + eps)
            if dd < min_dd:
                min_dd = dd
        max_drawdown[c] = min_dd

        abs_dd = -min_dd if min_dd < 0.0 else min_dd
        if abs_dd > eps:
            calmar[c] = cagr[c] / abs_dd

    return total_return, cagr, volatility, sharpe, sortino, max_drawdown, calmar

def _compute_equity_curves_vectorized(
    signal_matrix   : np.ndarray,
    close_prices    : np.ndarray,
    initial_cash    : float,
    commission_rate : float,
    slippage_rate   : float,
) -> tuple:
    return _compute_equity_curves_jit(
        signal_matrix.astype(np.float64),
        close_prices.astype(np.float64),
        float(initial_cash),
        float(commission_rate),
        float(slippage_rate),
    )


def _compute_metrics_vectorized(
    equity_matrix   : np.ndarray,
    net_returns     : np.ndarray,
    periods_per_year: float,
    initial_cash    : float,
) -> Dict[str, np.ndarray]:
    total_return, cagr, volatility, sharpe, sortino, max_drawdown, calmar = \
        _compute_metrics_jit(
            equity_matrix.astype(np.float64),
            net_returns.astype(np.float64),
            float(periods_per_year),
            float(initial_cash),
        )

    return {
        "total_return" : total_return,
        "cagr"         : cagr,
        "volatility"   : volatility,
        "sharpe_ratio" : sharpe,
        "sortino_ratio": sortino,
        "max_drawdown" : max_drawdown,
        "calmar_ratio" : calmar,
    }


def run_numpy_optimization(
    close_prices      : np.ndarray,
    volume            : Optional[np.ndarray],
    ticker            : str,
    strategy_class,
    param_combinations: List[Dict[str, Any]],
    initial_cash      : float,
    commission_rate   : float,
    slippage_rate     : float,
    interval          : str,
    batch_size        : int = 5000,
    progress          = None,
    task              = None,
) -> List[Dict[str, Any]]:
    from engine.metrics import get_periods_per_year

    logger.info(
        "%s — NumPy optimization start | %d combinations | batch_size: %d",
        ticker,
        len(param_combinations),
        batch_size,
    )

    periods_per_year = get_periods_per_year(interval)
    strategy_name    = strategy_class.__name__
    n_combos         = len(param_combinations)
    n_batches        = (n_combos + batch_size - 1) // batch_size
    all_results      = []

    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end   = min(batch_start + batch_size, n_combos)
        batch       = param_combinations[batch_start:batch_end]

        logger.debug(
            "%s — batch %d/%d | combos %d→%d",
            ticker,
            batch_idx + 1,
            n_batches,
            batch_start,
            batch_end,
        )

        signal_matrix = strategy_class.generate_signals_numpy(
            close_prices = close_prices,
            volume       = volume,
            params_list  = batch,
        )

        equity_matrix, net_returns = _compute_equity_curves_vectorized(
            signal_matrix   = signal_matrix,
            close_prices    = close_prices,
            initial_cash    = initial_cash,
            commission_rate = commission_rate,
            slippage_rate   = slippage_rate,
        )

        metrics = _compute_metrics_vectorized(
            equity_matrix    = equity_matrix,
            net_returns      = net_returns,
            periods_per_year = periods_per_year,
            initial_cash     = initial_cash,
        )

        for i, params in enumerate(batch):
            result = {
                "ticker"  : ticker,
                "strategy": strategy_name,
            }
            for k, v in params.items():
                result[k] = v.item() if hasattr(v, "item") else v
            for metric_name, metric_arr in metrics.items():
                result[metric_name] = float(metric_arr[i])
            result["equity_curve_series"] = equity_matrix[i].tolist()
            all_results.append(result)

        if progress is not None and task is not None:
            progress.advance(task, advance=len(batch))

    logger.info(
        "%s — NumPy optimization complete | %d results",
        ticker,
        len(all_results),
    )

    return all_results