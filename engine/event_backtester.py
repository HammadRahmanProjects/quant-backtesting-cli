import logging
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from engine.events import (
    EventType,
    FillEvent,
    MarketEvent,
    OrderDirection,
    OrderEvent,
    OrderType,
    SignalDirection,
    SignalEvent,
)
from engine.execution_handler import SimulatedExecutionHandler
from engine.metrics import compute_all_metrics
from engine.position_sizer import SizingMethod, calculate_position_size

logger = logging.getLogger(__name__)

class Position:
    """Tracks an open position for a single ticker."""

    def __init__(self, ticker: str):
        self.ticker      = ticker
        self.quantity    = 0.0    # positive = long, negative = short
        self.avg_price   = 0.0    # average entry price
        self.realised_pnl   = 0.0
        self.unrealised_pnl = 0.0

    @property
    def is_long(self):
        return self.quantity > 0

    @property
    def is_short(self):
        return self.quantity < 0

    @property
    def is_flat(self):
        return self.quantity == 0

    def update_from_fill(self, fill: FillEvent):
        """Update position state after a fill."""
        if fill.direction == OrderDirection.BUY:
            # Opening or adding to long / closing short
            if self.quantity < 0:
                # Closing short — realise PnL
                close_qty = min(abs(self.quantity), fill.quantity)
                self.realised_pnl += (self.avg_price - fill.fill_price) * close_qty
                self.quantity     += fill.quantity
            else:
                # Opening or adding to long
                total_cost      = self.avg_price * self.quantity + fill.fill_price * fill.quantity
                self.quantity  += fill.quantity
                self.avg_price  = total_cost / self.quantity if self.quantity > 0 else 0.0

        else:  # SELL
            # Opening or adding to short / closing long
            if self.quantity > 0:
                # Closing long — realise PnL
                close_qty = min(self.quantity, fill.quantity)
                self.realised_pnl += (fill.fill_price - self.avg_price) * close_qty
                self.quantity     -= fill.quantity
            else:
                # Opening or adding to short
                total_cost      = self.avg_price * abs(self.quantity) + fill.fill_price * fill.quantity
                self.quantity  -= fill.quantity
                self.avg_price  = total_cost / abs(self.quantity) if self.quantity != 0 else 0.0

    def update_unrealised_pnl(self, current_price: float):
        if self.is_long:
            self.unrealised_pnl = (current_price - self.avg_price) * self.quantity
        elif self.is_short:
            self.unrealised_pnl = (self.avg_price - current_price) * abs(self.quantity)
        else:
            self.unrealised_pnl = 0.0

class EventBacktester:
    """
    Event-driven backtester.

    Processes one bar at a time in chronological order. At each bar:

        1. MarketEvent  — new bar data made available
        2. SignalEvent  — strategy generates direction based on data so far
        3. OrderEvent   — position sizer determines trade quantity
        4. FillEvent    — execution handler fills the order at next bar open

    This architecture makes look-ahead bias structurally impossible —
    the strategy can only see data up to and including the current bar.

    Args:
        df               : processed OHLCV DataFrame with signal column
        initial_cash     : starting portfolio value
        commission_rate  : commission as fraction of trade value
        slippage_rate    : slippage as fraction of fill price
        sizing_method    : position sizing method (SizingMethod enum)
        sizing_params    : dict of kwargs passed to the position sizer
    """

    def __init__(
        self,
        df              : pd.DataFrame,
        initial_cash    : float      = 10_000,
        commission_rate : float      = 0.001,
        slippage_rate   : float      = 0.001,
        sizing_method   : SizingMethod = SizingMethod.FULL_PORTFOLIO,
        sizing_params   : Optional[Dict] = None,
    ):
        self.df              = df.reset_index(drop=True).copy()
        self.initial_cash    = initial_cash
        self.cash            = initial_cash
        self.sizing_method   = sizing_method
        self.sizing_params   = sizing_params or {}

        self.execution_handler = SimulatedExecutionHandler(
            commission_rate=commission_rate,
            slippage_rate=slippage_rate,
        )

        # Portfolio state
        self.positions   : Dict[str, Position] = {}
        self.equity_curve: List[float]         = []
        self.trade_log   : List[Dict]          = []
        self.bar_log     : List[Dict]          = []

        # Pending order queue — filled at next bar open
        self.pending_orders: deque = deque()

        # Return history for volatility targeting / Kelly
        self.bar_returns   : List[float] = []
        self.trade_returns : List[float] = []

        logger.info(
            "EventBacktester initialized | cash: %.2f | sizing: %s | rows: %d",
            initial_cash,
            sizing_method.value,
            len(self.df),
        )

    def _get_or_create_position(self, ticker: str) -> Position:
        if ticker not in self.positions:
            self.positions[ticker] = Position(ticker)
        return self.positions[ticker]

    def _portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Current cash + market value of all open positions."""
        holdings_value = sum(
            pos.quantity * current_prices.get(pos.ticker, 0.0)
            for pos in self.positions.values()
        )
        return self.cash + holdings_value

    def _process_pending_orders(self, bar: pd.Series, bar_idx: int):
        """
        Fill all pending orders at this bar's open price.
        Orders placed on bar N fill at bar N+1 open — realistic execution.
        """
        if not self.pending_orders:
            return

        while self.pending_orders:
            order = self.pending_orders.popleft()

            fill = self.execution_handler.execute_order(
                order      = order,
                fill_price = bar["open"],
                dt         = bar["datetime"],
            )

            position = self._get_or_create_position(fill.ticker)
            prev_qty = position.quantity

            position.update_from_fill(fill)

            # Update cash
            if fill.direction == OrderDirection.BUY:
                self.cash -= fill.total_cost
            else:
                self.cash += (fill.fill_price * fill.quantity) - fill.commission

            # Record trade
            self.trade_log.append({
                "datetime"  : fill.datetime,
                "ticker"    : fill.ticker,
                "direction" : fill.direction.value,
                "quantity"  : fill.quantity,
                "fill_price": fill.fill_price,
                "commission": fill.commission,
                "slippage"  : fill.slippage,
                "total_cost": fill.total_cost,
                "realised_pnl": position.realised_pnl,
            })

            logger.debug(
                "%s — fill processed | %s | qty: %.4f→%.4f | cash: %.2f",
                fill.ticker,
                fill.direction.value,
                prev_qty,
                position.quantity,
                self.cash,
            )

    def _generate_order(
        self,
        signal    : SignalEvent,
        bar       : pd.Series,
        portfolio_value: float,
    ) -> Optional[OrderEvent]:
        """
        Convert a signal into an order with position sizing applied.
        Returns None if no trade is needed.
        """
        ticker   = signal.ticker
        position = self._get_or_create_position(ticker)
        price    = bar["close"]

        if signal.direction == SignalDirection.LONG:
            target_qty = calculate_position_size(
                method          = self.sizing_method,
                portfolio_value = portfolio_value,
                price           = price,
                returns         = pd.Series(self.bar_returns),
                trade_returns   = pd.Series(self.trade_returns),
                **self.sizing_params,
            )
            target_qty = round(target_qty, 6)

            if position.is_long:
                return None  # already long, no action needed

            # Close short first if needed, then go long
            if position.is_short:
                return OrderEvent(
                    ticker    = ticker,
                    datetime  = bar["datetime"],
                    direction = OrderDirection.BUY,
                    quantity  = abs(position.quantity) + target_qty,
                    order_type= OrderType.MARKET,
                )

            return OrderEvent(
                ticker    = ticker,
                datetime  = bar["datetime"],
                direction = OrderDirection.BUY,
                quantity  = target_qty,
                order_type= OrderType.MARKET,
            )

        elif signal.direction == SignalDirection.SHORT:
            target_qty = calculate_position_size(
                method          = self.sizing_method,
                portfolio_value = portfolio_value,
                price           = price,
                returns         = pd.Series(self.bar_returns),
                trade_returns   = pd.Series(self.trade_returns),
                **self.sizing_params,
            )
            target_qty = round(target_qty, 6)

            if position.is_short:
                return None  # already short, no action needed

            # Close long first if needed, then go short
            if position.is_long:
                return OrderEvent(
                    ticker    = ticker,
                    datetime  = bar["datetime"],
                    direction = OrderDirection.SELL,
                    quantity  = position.quantity + target_qty,
                    order_type= OrderType.MARKET,
                )

            return OrderEvent(
                ticker    = ticker,
                datetime  = bar["datetime"],
                direction = OrderDirection.SELL,
                quantity  = target_qty,
                order_type= OrderType.MARKET,
            )

        elif signal.direction == SignalDirection.FLAT:
            # Close any open position
            if position.is_long:
                return OrderEvent(
                    ticker    = ticker,
                    datetime  = bar["datetime"],
                    direction = OrderDirection.SELL,
                    quantity  = position.quantity,
                    order_type= OrderType.MARKET,
                )
            elif position.is_short:
                return OrderEvent(
                    ticker    = ticker,
                    datetime  = bar["datetime"],
                    direction = OrderDirection.BUY,
                    quantity  = abs(position.quantity),
                    order_type= OrderType.MARKET,
                )

        return None

    def _signal_to_direction(self, raw_signal: int) -> SignalDirection:
        if raw_signal == 1:
            return SignalDirection.LONG
        elif raw_signal == -1:
            return SignalDirection.SHORT
        else:
            return SignalDirection.FLAT

    def run(self) -> pd.DataFrame:
        """
        Run the event-driven backtest bar by bar.
        Returns a DataFrame matching the vectorized backtester output format
        so all downstream metrics and visualization code works unchanged.
        """
        logger.info(
            "EventBacktester.run — start | rows: %d | sizing: %s",
            len(self.df),
            self.sizing_method.value,
        )

        ticker = "asset"  # single-asset mode

        for bar_idx, bar in self.df.iterrows():

            if bar_idx > 0:
                self._process_pending_orders(bar, bar_idx)

            current_price = bar["close"]
            position      = self._get_or_create_position(ticker)
            position.update_unrealised_pnl(current_price)

            portfolio_value = self._portfolio_value({ticker: current_price})

            raw_signal = bar.get("signal", 0)
            direction  = self._signal_to_direction(int(raw_signal))

            signal = SignalEvent(
                ticker    = ticker,
                datetime  = bar["datetime"],
                direction = direction,
            )

            order = self._generate_order(signal, bar, portfolio_value)

            if order is not None and order.quantity > 0:
                self.pending_orders.append(order)
                logger.debug(
                    "Bar %d — order queued | %s | qty: %.4f",
                    bar_idx,
                    order.direction.value,
                    order.quantity,
                )

            if bar_idx > 0:
                prev_close = self.df.iloc[bar_idx - 1]["close"]
                bar_return = (current_price / prev_close - 1) if prev_close > 0 else 0.0
                self.bar_returns.append(bar_return)

            self.equity_curve.append(portfolio_value)

            self.bar_log.append({
                "datetime"       : bar["datetime"],
                "close"          : current_price,
                "signal"         : raw_signal,
                "position"       : position.quantity,
                "portfolio_value": portfolio_value,
                "cash"           : self.cash,
                "unrealised_pnl" : position.unrealised_pnl,
                "realised_pnl"   : position.realised_pnl,
            })

        results_df = pd.DataFrame(self.bar_log)
        results_df["equity_curve"]         = self.equity_curve
        results_df["returns"]              = results_df["close"].pct_change().fillna(0)
        results_df["strategy_returns"]     = results_df["equity_curve"].pct_change().fillna(0)
        results_df["net_strategy_returns"] = results_df["strategy_returns"]
        results_df["transaction_cost"]     = 0.0  # already baked into fills

        final_equity = self.equity_curve[-1] if self.equity_curve else self.initial_cash
        total_return = (final_equity / self.initial_cash - 1) * 100
        n_trades     = len(self.trade_log)

        logger.info(
            "EventBacktester.run complete | final equity: %.2f | "
            "return: %.2f%% | trades: %d",
            final_equity,
            total_return,
            n_trades,
        )

        return results_df

    def get_trade_log(self) -> pd.DataFrame:
        """Returns the full trade log as a DataFrame."""
        return pd.DataFrame(self.trade_log)