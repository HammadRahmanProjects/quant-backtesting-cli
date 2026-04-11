from indicators.volatility import add_bollinger_bands
from indicators.moving_averages import add_price_vs_moving_average
from strategies.base_strategy import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    
    def __init__(
        self,
        data,
        window           : int   = 20,
        num_std          : float = 2.0,
        lower_threshold  : float = 0.2,
        upper_threshold  : float = 0.8,
        long_only        : bool  = True,
    ):
        super().__init__(data)
        self.window          = window
        self.num_std         = num_std
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
        self.long_only       = long_only

        if self.window <= 0:
            raise ValueError("window must be a positive integer.")
        if self.num_std <= 0:
            raise ValueError("num_std must be positive.")
        if not (0 < self.lower_threshold < self.upper_threshold < 1):
            raise ValueError(
                "Thresholds must satisfy: 0 < lower_threshold < upper_threshold < 1"
            )

    def generate_signals(self):
        df = add_bollinger_bands(
            self.df,
            window  = self.window,
            num_std = self.num_std,
        )

        df["signal"] = 0
        valid_mask   = df["bb_pct"].notna()

        df.loc[valid_mask & (df["bb_pct"] < self.lower_threshold), "signal"] = 1

        if not self.long_only:
            df.loc[valid_mask & (df["bb_pct"] > self.upper_threshold), "signal"] = -1

        return df

    @classmethod
    def get_param_names(cls):
        return ["window", "num_std", "lower_threshold", "upper_threshold", "long_only"]

    @classmethod
    def get_default_params(cls):
        return {
            "window"          : 20,
            "num_std"         : 2.0,
            "lower_threshold" : 0.2,
            "upper_threshold" : 0.8,
            "long_only"       : True,
        }

    @classmethod
    def get_optimization_grid(cls):
        return {
            "window"          : list(range(5, 105, 2)),
            "num_std"         : [x / 10 for x in range(10, 35)],
            "lower_threshold" : [x / 100 for x in range(5, 40, 5)],
            "upper_threshold" : [x / 100 for x in range(65, 100, 5)],
            "long_only"       : [True],
        }