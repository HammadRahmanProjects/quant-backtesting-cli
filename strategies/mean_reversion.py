from indicators.volatility import add_bollinger_bands
from indicators.moving_averages import add_price_vs_moving_average
from strategies.base_strategy import BaseStrategy


class MeanReversionStrategy(BaseStrategy):

    """
    Mean Reversion Strategy using Bollinger Bands + Z-Score confirmation.

    Core idea: prices tend to revert to their historical mean. When price
    drops significantly below its rolling mean (oversold), we go long.
    When price rises significantly above its rolling mean (overbought),
    we go short.

    Signal logic:
        BUY  (+1) : bb_pct < lower_threshold  (price near/below lower band)
        SELL (-1) : bb_pct > upper_threshold  (price near/above upper band)
        FLAT ( 0) : price within the bands

    bb_pct is where the price sits within the Bollinger Bands:
        0.0 = exactly at the lower band
        0.5 = exactly at the middle (mean)
        1.0 = exactly at the upper band

    So a lower_threshold of 0.2 means "price is in the bottom 20% of
    the band" → buy signal. An upper_threshold of 0.8 means "price is
    in the top 20% of the band" → sell signal.
    """

    def __init__(
        self,
        data,
        window: int = 20,
        num_std: float = 2.0,
        lower_threshold: float = 0.2,
        upper_threshold: float = 0.8,
    ):
        super().__init__(data)
        self.window           = window
        self.num_std          = num_std
        self.lower_threshold  = lower_threshold
        self.upper_threshold  = upper_threshold

        if self.window <= 0:
            raise ValueError("window must be a positive integer.")
        if self.num_std <= 0:
            raise ValueError("num_std must be positive.")
        if not (0 < self.lower_threshold < self.upper_threshold < 1):
            raise ValueError(
                "Thresholds must satisfy: 0 < lower_threshold < upper_threshold < 1"
            )

    def generate_signals(self) -> "pd.DataFrame":
        df = add_bollinger_bands(
            self.df,
            window=self.window,
            num_std=self.num_std,
        )

        df["signal"] = 0

        valid_mask = df["bb_pct"].notna()

        df.loc[valid_mask & (df["bb_pct"] < self.lower_threshold), "signal"] = 1
        df.loc[valid_mask & (df["bb_pct"] > self.upper_threshold), "signal"] = -1

        return df

    @classmethod
    def get_param_names(cls):
        return ["window", "num_std", "lower_threshold", "upper_threshold"]

    @classmethod
    def get_default_params(cls):
        return {
            "window":          20,
            "num_std":         2.0,
            "lower_threshold": 0.2,
            "upper_threshold": 0.8,
        }

    @classmethod
    def get_optimization_grid(cls):
        return {
            "window":          list(range(5, 105, 2)),          # 50 values
            "num_std":         [x / 10 for x in range(10, 35)], # 25 values  1.0 → 3.4
            "lower_threshold": [x / 100 for x in range(5, 40, 5)],  # 7 values  0.05 → 0.35
            "upper_threshold": [x / 100 for x in range(65, 100, 5)], # 7 values  0.65 → 0.95
        }