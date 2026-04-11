import logging
from typing import Any, Dict, List

import numpy as np

from indicators.moving_averages import add_moving_averages
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class MovingAverageCrossStrategy(BaseStrategy):
    """
    Moving Average Crossover Strategy.

    Buys when short MA crosses above long MA (golden cross).
    Sells when short MA crosses below long MA (death cross).
    """

    def __init__(self, data, short_window=20, long_window=50):
        super().__init__(data)
        self.short_window = short_window
        self.long_window  = long_window

        if self.short_window <= 0 or self.long_window <= 0:
            raise ValueError("Moving average windows must be positive integers.")
        if self.short_window >= self.long_window:
            raise ValueError("short_window must be less than long_window.")

    def generate_signals(self):
        df = add_moving_averages(
            self.df,
            short_window=self.short_window,
            long_window=self.long_window,
        )

        df["signal"]  = 0
        valid_mask    = df["short_ma"].notna() & df["long_ma"].notna()

        df.loc[valid_mask & (df["short_ma"] > df["long_ma"]), "signal"] = 1
        df.loc[valid_mask & (df["short_ma"] < df["long_ma"]), "signal"] = -1

        return df

    @classmethod
    def generate_signals_numpy(
        cls,
        close_prices : np.ndarray,
        params_list  : List[Dict[str, Any]],
        volume       : np.ndarray = None,
    ) -> np.ndarray:
        """
        Vectorized MA cross signal generation for all combinations.

        Computes rolling means for all unique short and long windows
        simultaneously, reusing computations across shared windows.

        Args:
            close_prices : (n_bars,) float64
            params_list  : list of param dicts
            volume       : unused — accepted for API consistency

        Returns:
            signal_matrix : (n_combos, n_bars) float64
        """
        from engine.numpy_backtester import _rolling_mean_vectorized

        n_bars   = len(close_prices)
        n_combos = len(params_list)

        short_windows  = np.array([p["short_window"] for p in params_list], dtype=np.int64)
        long_windows   = np.array([p["long_window"]  for p in params_list], dtype=np.int64)
        unique_windows = np.unique(np.concatenate([short_windows, long_windows]))

        # Compute all rolling means in one vectorized pass
        all_means     = _rolling_mean_vectorized(close_prices, unique_windows)
        window_to_idx = {int(w): i for i, w in enumerate(unique_windows)}

        signal_matrix = np.zeros((n_combos, n_bars), dtype=np.float64)

        for i, params in enumerate(params_list):
            short_w   = int(params["short_window"])
            long_w    = int(params["long_window"])
            long_only = bool(params.get("long_only", False))

            short_ma = all_means[window_to_idx[short_w]]
            long_ma  = all_means[window_to_idx[long_w]]

            valid = ~(np.isnan(short_ma) | np.isnan(long_ma))

            signal_matrix[i, valid & (short_ma > long_ma)] = 1.0

            if not long_only:
                signal_matrix[i, valid & (short_ma < long_ma)] = -1.0

        return signal_matrix

    @classmethod
    def get_optimization_grid(cls):
        return {
            "short_window": list(range(5, 155)),
            "long_window" : list(range(155, 355)),
        }

    @classmethod
    def get_param_names(cls):
        return ["short_window", "long_window"]

    @classmethod
    def get_default_params(cls):
        return {
            "short_window": 20,
            "long_window" : 50,
        }