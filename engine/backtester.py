class Backtester:
    
    def __init__(
        self,
        df,
        initial_cash=10000,
        commission_rate=0.001,
        slippage_rate=0.001
    ):
        self.df = df.copy()
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate

        self.position = 0
        self.shares = 0
        self.entry_price = None

        self.equity = initial_cash
        self.equity_curve = []
        self.positions = []

        self.trade_log = []

    def run(self):

        df = self.df.copy()

        df["returns"] = df["close"].pct_change().fillna(0)
        df["position"] = df["signal"].shift(1).fillna(0)
        df["trade"] = df["position"].diff().abs().fillna(0)

        df["strategy_returns"] = df["position"] * df["returns"]
        df["transaction_cost"] = df["trade"] * (self.commission_rate + self.slippage_rate)
        df["net_strategy_returns"] = df["strategy_returns"] - df["transaction_cost"]
        df["equity_curve"] = (1 + df["net_strategy_returns"]).cumprod() * self.initial_cash

        return df