import logging
from datetime import datetime

from engine.events import FillEvent, OrderDirection, OrderEvent

logger = logging.getLogger(__name__)

class SimulatedExecutionHandler:
    """
    Simulates order execution against historical bar data.

    Key difference from vectorized backtester:
        - Slippage is applied TO the fill price itself, not subtracted
          as a separate cost. This means a BUY fills slightly above the
          bar open, a SELL fills slightly below. This is more realistic.

        - Orders fill at the NEXT bar's open price, not the current
          bar's close. You can't trade on a close price you just observed.

        - Commission is charged as a flat rate on the total trade value.
    """

    def __init__(
        self,
        commission_rate : float = 0.001,
        slippage_rate   : float = 0.001,
    ):
        self.commission_rate = commission_rate
        self.slippage_rate   = slippage_rate

        logger.debug(
            "SimulatedExecutionHandler initialized | "
            "commission: %.4f | slippage: %.4f",
            commission_rate,
            slippage_rate,
        )

    def execute_order(
        self,
        order      : OrderEvent,
        fill_price : float,
        dt         : datetime,
    ) -> FillEvent:
        """
        Execute an order at the given fill price (typically next bar open).

        Slippage is applied directionally:
            BUY  → fill_price * (1 + slippage_rate)  [pays more]
            SELL → fill_price * (1 - slippage_rate)  [receives less]

        Args:
            order      : the OrderEvent to execute
            fill_price : base price to fill at (next bar open)
            dt         : timestamp of fill

        Returns:
            FillEvent with all execution details
        """
        if order.direction == OrderDirection.BUY:
            actual_fill = fill_price * (1 + self.slippage_rate)
        else:
            actual_fill = fill_price * (1 - self.slippage_rate)

        slippage_cost = abs(actual_fill - fill_price) * order.quantity
        commission    = actual_fill * order.quantity * self.commission_rate
        total_cost    = actual_fill * order.quantity + commission

        fill = FillEvent(
            ticker     = order.ticker,
            datetime   = dt,
            direction  = order.direction,
            quantity   = order.quantity,
            fill_price = actual_fill,
            commission = commission,
            slippage   = slippage_cost,
            total_cost = total_cost,
        )

        logger.debug(
            "%s — fill | %s | qty: %.4f | base: %.4f | "
            "fill: %.4f | commission: %.4f | slippage: %.4f | total: %.4f",
            order.ticker,
            order.direction.value,
            order.quantity,
            fill_price,
            actual_fill,
            commission,
            slippage_cost,
            total_cost,
        )

        return fill