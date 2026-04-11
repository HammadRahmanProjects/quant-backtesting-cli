import logging

logger = logging.getLogger(__name__)

class Backtester:

    def __init__(
        self,
        df,
        initial_cash=10000,
        commission_rate=0.001,
        slippage_rate=0.001,
    ):
        self.df              = df.copy()
        self.initial_cash    = initial_cash
        self.cash            = initial_cash
        self.commission_rate = commission_rate
        self.slippage_rate   = slippage_rate

        self.position    = 0
        self.shares      = 0
        self.entry_price = None

        self.equity       = initial_cash
        self.equity_curve = []
        self.positions    = []
        self.trade_log    = []

    def run(self):
        logger.debug(
            "Backtester.run — rows: %d | cash: %.2f | commission: %.4f | slippage: %.4f",
            len(self.df),
            self.initial_cash,
            self.commission_rate,
            self.slippage_rate,
        )

        df = self.df.copy()

        df["returns"]             = df["close"].pct_change().fillna(0)
        df["position"]            = df["signal"].shift(1).fillna(0)
        df["trade"]               = df["position"].diff().abs().fillna(0)
        df["strategy_returns"]    = df["position"] * df["returns"]
        df["transaction_cost"]    = df["trade"] * (self.commission_rate + self.slippage_rate)
        df["net_strategy_returns"] = df["strategy_returns"] - df["transaction_cost"]
        df["equity_curve"]        = (1 + df["net_strategy_returns"]).cumprod() * self.initial_cash

        final_equity  = df["equity_curve"].iloc[-1]
        total_return  = (final_equity / self.initial_cash - 1) * 100
        n_trades      = int(df["trade"].sum())
        total_cost    = df["transaction_cost"].sum()

        logger.debug(
            "Backtester.run complete — final equity: %.2f | return: %.2f%% | trades: %d | total cost: %.4f",
            final_equity,
            total_return,
            n_trades,
            total_cost,
        )

        return df