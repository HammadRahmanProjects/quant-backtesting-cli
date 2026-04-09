from strategies.base_strategy import BaseStrategy

class MeanReversionStrategy(BaseStrategy):

    def __init__(self, data, window=20, threshold=2.0):
        super().__init__(data)
        self.window = window
        self.threshold = threshold

        if self.window <= 0:
            raise ValueError("window must be positive.")

        if self.threshold <= 0:
            raise ValueError("threshold must be positive.")

    def generate_signals(self):
        df = self.df.copy()

        if "close" not in df.columns:
            raise ValueError("Input DataFrame must contain a 'close' column.")

        df["rolling_mean"] = df["close"].rolling(window=self.window).mean()
        df["rolling_std"] = df["close"].rolling(window=self.window).std()

        df["z_score"] = (df["close"] - df["rolling_mean"]) / df["rolling_std"]

        df["signal"] = 0

        valid_mask = df["z_score"].notna()

        df.loc[valid_mask & (df["z_score"] < -self.threshold), "signal"] = 1
        df.loc[valid_mask & (df["z_score"] > self.threshold), "signal"] = -1

        return df

    @classmethod
    def get_optimization_grid(cls):
        return {
            "window": list(range(5, 105)),
            "threshold": [x / 100 for x in range(50, 150)],
        }

    @classmethod
    def get_param_names(cls):
        return ["window", "threshold"]