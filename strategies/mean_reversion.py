import logging
from typing import Any, Dict, List

import numpy as np

from indicators.volatility import add_bollinger_bands
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy using Bollinger Bands.

    Buys when price drops to the lower band (oversold).
    Optionally shorts when price rises to the upper band (overbought).

    Signal logic:
        BUY  (+1) : bb_pct < lower_threshold
        SELL (-1) : bb_pct > upper_threshold (if not long_only)
        FLAT ( 0) : price within band
    """

    def __init__(
        self,
        data,
        window          : int   = 20,
        num_std         : float = 2.0,
        lower_threshold : float = 0.2,
        upper_threshold : float = 0.8,
        long_only       : bool  = True,
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
        df         = add_bollinger_bands(self.df, window=self.window, num_std=self.num_std)
        df["signal"] = 0
        valid_mask   = df["bb_pct"].notna()

        df.loc[valid_mask & (df["bb_pct"] < self.lower_threshold), "signal"] = 1
        if not self.long_only:
            df.loc[valid_mask & (df["bb_pct"] > self.upper_threshold), "signal"] = -1

        return df

    @classmethod
    def generate_signals_numpy(
        cls,
        close_prices : np.ndarray,
        params_list  : List[Dict[str, Any]],
        volume       : np.ndarray = None,
    ) -> np.ndarray:
        """
        Vectorized Bollinger Band signal generation for all combinations.

        Computes rolling means and stds for all unique windows at once,
        then generates signals for all combinations in a single pass.
        Reuses rolling computations across combinations sharing the same window.

        Args:
            close_prices : (n_bars,) float64
            params_list  : list of param dicts
            volume       : unused — accepted for API consistency

        Returns:
            signal_matrix : (n_combos, n_bars) float64
        """
        from engine.numpy_backtester import (
            _rolling_mean_vectorized,
            _rolling_std_vectorized,
        )

        n_bars   = len(close_prices)
        n_combos = len(params_list)

        # Unique windows — compute rolling stats once, reuse across combos
        all_windows    = np.array([p["window"] for p in params_list], dtype=np.int64)
        unique_windows = np.unique(all_windows)

        means = _rolling_mean_vectorized(close_prices, unique_windows)
        stds  = _rolling_std_vectorized(close_prices, unique_windows)

        window_to_idx = {int(w): i for i, w in enumerate(unique_windows)}
        signal_matrix = np.zeros((n_combos, n_bars), dtype=np.float64)

        for i, params in enumerate(params_list):
            window          = int(params["window"])
            num_std         = float(params["num_std"])
            lower_threshold = float(params["lower_threshold"])
            upper_threshold = float(params["upper_threshold"])
            long_only       = bool(params.get("long_only", True))

            w_idx    = window_to_idx[window]
            mean     = means[w_idx]
            std      = stds[w_idx]

            bb_upper = mean + num_std * std
            bb_lower = mean - num_std * std
            bb_range = bb_upper - bb_lower

            # bb_pct: 0 = at lower band, 1 = at upper band
            with np.errstate(invalid="ignore", divide="ignore"):
                bb_pct = np.where(
                    bb_range > 0,
                    (close_prices - bb_lower) / bb_range,
                    np.nan,
                )

            valid = ~np.isnan(bb_pct)

            signal_matrix[i, valid & (bb_pct < lower_threshold)] = 1.0

            if not long_only:
                signal_matrix[i, valid & (bb_pct > upper_threshold)] = -1.0

        return signal_matrix

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