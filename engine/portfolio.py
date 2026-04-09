class Portfolio:
    def __init__(
        self,
        name,
        initial_cash,
        commission_rate,
        slippage_rate,
        tickers,
        weights,
        strategy_map,
        start_date,
        end_date,
        interval
    ):
        self.name = name
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)

        self.commission_rate = float(commission_rate)
        self.slippage_rate = float(slippage_rate)

        self.tickers = tickers
        self.weights = weights
        self.strategy_map = strategy_map

        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval

        self.equity = float(initial_cash)

    def __repr__(self):
        return (
            f"Portfolio(name={self.name}, "
            f"initial_cash={self.initial_cash}, "
            f"commission_rate={self.commission_rate}, "
            f"slippage_rate={self.slippage_rate}, "
            f"tickers={self.tickers}, "
            f"weights={self.weights}, "
            f"strategy_map={self.strategy_map}, "
            f"start_date={self.start_date}, "
            f"end_date={self.end_date}, "
            f"interval={self.interval})"
        )