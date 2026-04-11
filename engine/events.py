from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

class EventType(Enum):
    MARKET = "MARKET"
    SIGNAL = "SIGNAL"
    ORDER  = "ORDER"
    FILL   = "FILL"

class SignalDirection(Enum):
    LONG  = "LONG"
    SHORT = "SHORT"
    FLAT  = "FLAT"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT  = "LIMIT"

class OrderDirection(Enum):
    BUY  = "BUY"
    SELL = "SELL"

@dataclass
class MarketEvent:
    event_type : EventType = field(default=EventType.MARKET, init=False)
    ticker     : str       = ""
    datetime   : datetime  = None
    open       : float     = 0.0
    high       : float     = 0.0
    low        : float     = 0.0
    close      : float     = 0.0
    volume     : float     = 0.0

@dataclass
class SignalEvent:
    event_type : EventType       = field(default=EventType.SIGNAL, init=False)
    ticker     : str             = ""
    datetime   : datetime        = None
    direction  : SignalDirection = SignalDirection.FLAT
    strength   : float           = 1.0

@dataclass
class OrderEvent:
    event_type  : EventType      = field(default=EventType.ORDER, init=False)
    ticker      : str            = ""
    datetime    : datetime       = None
    direction   : OrderDirection = OrderDirection.BUY
    quantity    : float          = 0.0
    order_type  : OrderType      = OrderType.MARKET
    limit_price : Optional[float] = None

@dataclass
class FillEvent:
    event_type  : EventType      = field(default=EventType.FILL, init=False)
    ticker      : str            = ""
    datetime    : datetime       = None
    direction   : OrderDirection = OrderDirection.BUY
    quantity    : float          = 0.0
    fill_price  : float          = 0.0
    commission  : float          = 0.0
    slippage    : float          = 0.0
    total_cost  : float          = 0.0