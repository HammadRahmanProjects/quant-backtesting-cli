from indicators.moving_averages import add_moving_averages
from strategies.base_strategy import BaseStrategy

class MovingAverageCrossStrategy(BaseStrategy):

    def __init__(self, data, short_window=20, long_window=50):
        super().__init__(data)
        self.short_window = short_window
        self.long_window = long_window

        if self.short_window <= 0 or self.long_window <= 0:
            raise ValueError("Moving average windows must be positive integers.")

        if self.short_window >= self.long_window:
            raise ValueError("short_window must be less than long_window.")

    def generate_signals(self):
        df = add_moving_averages(
            self.df,
            short_window=self.short_window,
            long_window=self.long_window
        )

        df["signal"] = 0

        valid_mask = df["short_ma"].notna() & df["long_ma"].notna()

        df.loc[valid_mask & (df["short_ma"] > df["long_ma"]), "signal"] = 1
        df.loc[valid_mask & (df["short_ma"] < df["long_ma"]), "signal"] = -1

        return df

    @classmethod
    def get_optimization_grid(cls):
        return {
            "short_window": list(range(5, 155)),
            "long_window": list(range(155, 355)),
        }

    @classmethod
    def get_param_names(cls):
        return ["short_window", "long_window"]

    @classmethod
    def get_default_params(cls):
        return {
            "short_window": 20,
            "long_window": 50,
        }