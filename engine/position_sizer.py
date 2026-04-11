import logging
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SizingMethod(Enum):
    FULL_PORTFOLIO     = "full_portfolio"      
    FIXED_FRACTIONAL   = "fixed_fractional"    
    VOLATILITY_TARGET  = "volatility_target"   
    KELLY              = "kelly"               

def size_full_portfolio(
    portfolio_value: float,
    price: float,
) -> float:
    """
    Returns the number of shares that uses the entire portfolio value.
    """
    if price <= 0:
        return 0.0
    shares = portfolio_value / price
    logger.debug(
        "size_full_portfolio — value: %.2f | price: %.2f | shares: %.4f",
        portfolio_value, price, shares,
    )
    return shares

def size_fixed_fractional(
    portfolio_value : float,
    price           : float,
    risk_pct        : float = 0.02,
    stop_loss_pct   : Optional[float] = 0.05,
) -> float:
    """
    Returns number of shares based on fixed fractional position sizing.

    Args:
        portfolio_value : current total portfolio value
        price           : current asset price
        risk_pct        : fraction of portfolio to risk per trade (default 2%)
        stop_loss_pct   : stop loss as fraction of price (default 5%)
                          if None, sizes as risk_pct * portfolio / price
    """
    if price <= 0:
        return 0.0

    risk_amount = portfolio_value * risk_pct

    if stop_loss_pct and stop_loss_pct > 0:
        stop_distance = price * stop_loss_pct
        shares = risk_amount / stop_distance
    else:
        shares = risk_amount / price

    # Cap at what we can actually afford
    max_shares = portfolio_value / price
    shares     = min(shares, max_shares)

    logger.debug(
        "size_fixed_fractional — value: %.2f | price: %.2f | risk_pct: %.3f | shares: %.4f",
        portfolio_value, price, risk_pct, shares,
    )

    return shares

def size_volatility_target(
    portfolio_value  : float,
    price            : float,
    returns          : pd.Series,
    target_vol_pct   : float = 0.15,
    lookback_window  : int   = 20,
    periods_per_year : int   = 252,
) -> float:
    """
    Returns number of shares based on volatility targeting.

    Args:
        portfolio_value  : current total portfolio value
        price            : current asset price
        returns          : series of historical returns up to current bar
        target_vol_pct   : annualized volatility target (default 15%)
        lookback_window  : bars to use for volatility estimate (default 20)
        periods_per_year : annualization factor (default 252 for daily)
    """
    if price <= 0:
        return 0.0

    recent_returns = returns.dropna().tail(lookback_window)

    if len(recent_returns) < 2:
        logger.warning(
            "size_volatility_target — insufficient return history, "
            "falling back to full portfolio"
        )
        return size_full_portfolio(portfolio_value, price)

    current_vol = recent_returns.std(ddof=1) * np.sqrt(periods_per_year)

    if current_vol <= 0:
        logger.warning(
            "size_volatility_target — zero volatility detected, "
            "falling back to full portfolio"
        )
        return size_full_portfolio(portfolio_value, price)

    shares     = (portfolio_value * target_vol_pct) / (price * current_vol)
    max_shares = portfolio_value / price
    shares     = min(shares, max_shares)

    logger.debug(
        "size_volatility_target — value: %.2f | price: %.2f | "
        "current_vol: %.4f | target_vol: %.4f | shares: %.4f",
        portfolio_value, price, current_vol, target_vol_pct, shares,
    )

    return shares

def size_kelly(
    portfolio_value : float,
    price           : float,
    trade_returns   : pd.Series,
    half_kelly      : bool  = True,
    min_trades      : int   = 20,
    max_fraction    : float = 0.25,
) -> float:
    """
    Returns number of shares based on Kelly criterion.

    Args:
        portfolio_value : current total portfolio value
        price           : current asset price
        trade_returns   : series of individual trade returns (not bar returns)
        half_kelly      : use half Kelly for safety (default True)
        min_trades      : minimum trades required to use Kelly (default 20)
        max_fraction    : cap Kelly fraction at this value (default 25%)
    """
    if price <= 0:
        return 0.0

    if len(trade_returns) < min_trades:
        logger.warning(
            "size_kelly — insufficient trade history (%d trades, need %d), "
            "falling back to fixed fractional",
            len(trade_returns),
            min_trades,
        )
        return size_fixed_fractional(portfolio_value, price)

    wins   = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]

    if len(wins) == 0 or len(losses) == 0:
        logger.warning(
            "size_kelly — no wins or no losses in history, "
            "falling back to fixed fractional"
        )
        return size_fixed_fractional(portfolio_value, price)

    win_rate  = len(wins) / len(trade_returns)
    avg_win   = wins.mean()
    avg_loss  = abs(losses.mean())

    if avg_loss == 0:
        return size_fixed_fractional(portfolio_value, price)

    kelly_fraction = (win_rate / avg_loss) - ((1 - win_rate) / avg_win)
    kelly_fraction = max(0.0, kelly_fraction)

    if half_kelly:
        kelly_fraction *= 0.5

    kelly_fraction = min(kelly_fraction, max_fraction)

    shares     = (portfolio_value * kelly_fraction) / price
    max_shares = portfolio_value / price
    shares     = min(shares, max_shares)

    logger.debug(
        "size_kelly — win_rate: %.3f | avg_win: %.4f | avg_loss: %.4f | "
        "kelly: %.4f | shares: %.4f",
        win_rate, avg_win, avg_loss, kelly_fraction, shares,
    )

    return shares

def calculate_position_size(
    method          : SizingMethod,
    portfolio_value : float,
    price           : float,
    returns         : Optional[pd.Series] = None,
    trade_returns   : Optional[pd.Series] = None,
    risk_pct        : float = 0.02,
    stop_loss_pct   : float = 0.05,
    target_vol_pct  : float = 0.15,
    lookback_window : int   = 20,
) -> float:
    """
    Central dispatcher — routes to the correct sizing function based on method.
    Always returns a non-negative number of shares.
    """
    if method == SizingMethod.FULL_PORTFOLIO:
        return size_full_portfolio(portfolio_value, price)

    elif method == SizingMethod.FIXED_FRACTIONAL:
        return size_fixed_fractional(
            portfolio_value, price, risk_pct, stop_loss_pct
        )

    elif method == SizingMethod.VOLATILITY_TARGET:
        if returns is None or returns.empty:
            logger.warning(
                "calculate_position_size — volatility target requested "
                "but no returns provided, falling back to full portfolio"
            )
            return size_full_portfolio(portfolio_value, price)
        return size_volatility_target(
            portfolio_value, price, returns,
            target_vol_pct, lookback_window,
        )

    elif method == SizingMethod.KELLY:
        if trade_returns is None or trade_returns.empty:
            logger.warning(
                "calculate_position_size — Kelly requested "
                "but no trade history provided, falling back to fixed fractional"
            )
            return size_fixed_fractional(portfolio_value, price)
        return size_kelly(portfolio_value, price, trade_returns)

    else:
        logger.error("calculate_position_size — unknown method: %s", method)
        raise ValueError(f"Unknown sizing method: {method}")