from strategies.base_strategy import BaseStrategy

class VolumeSpikeMomentumStrategy(BaseStrategy):

    def __init__(self, data, volume_window=20, spike_threshold=2.0):
        super().__init__(data)
        self.volume_window    = volume_window
        self.spike_threshold  = spike_threshold

        if self.volume_window <= 0:
            raise ValueError("volume_window must be positive.")
        if self.spike_threshold <= 0:
            raise ValueError("spike_threshold must be positive.")

    def generate_signals(self):
        df = self.df.copy()

        required_cols = {"close", "volume"}
        if not required_cols.issubset(df.columns):
            raise ValueError("DataFrame must contain 'close' and 'volume' columns.")

        df["returns"]      = df["close"].pct_change()
        df["avg_volume"]   = df["volume"].rolling(window=self.volume_window).mean()
        df["volume_ratio"] = df["volume"] / df["avg_volume"]

        df["signal"] = 0

        valid_mask = df["volume_ratio"].notna() & df["returns"].notna()

        df.loc[
            valid_mask &
            (df["volume_ratio"] > self.spike_threshold) &
            (df["returns"] > 0),
            "signal"
        ] = 1

        df.loc[
            valid_mask &
            (df["volume_ratio"] > self.spike_threshold) &
            (df["returns"] < 0),
            "signal"
        ] = -1

        return df

    @classmethod
    def get_optimization_grid(cls):
        return {
            "volume_window":   list(range(5, 55)),
            "spike_threshold": [x / 10 for x in range(10, 41)],
        }

    @classmethod
    def get_param_names(cls):
        return ["volume_window", "spike_threshold"]

    @classmethod
    def get_default_params(cls):
        return {
            "volume_window":  20,
            "spike_threshold": 2.0,
        }