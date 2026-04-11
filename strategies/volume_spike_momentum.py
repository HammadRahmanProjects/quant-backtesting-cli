import logging
from typing import Any, Dict, List

import numpy as np

from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class VolumeSpikeMomentumStrategy(BaseStrategy):
    """
    Volume Spike Momentum Strategy.

    Identifies bars where volume spikes significantly above its
    rolling average and enters in the direction of price movement.

    Signal logic:
        BUY  (+1) : volume_ratio > spike_threshold AND returns > 0
        SELL (-1) : volume_ratio > spike_threshold AND returns < 0
        FLAT ( 0) : no spike or insufficient data
    """

    def __init__(self, data, volume_window=20, spike_threshold=2.0):
        super().__init__(data)
        self.volume_window   = volume_window
        self.spike_threshold = spike_threshold

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
        df["signal"]       = 0

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
    def generate_signals_numpy(
        cls,
        close_prices : np.ndarray,
        params_list  : List[Dict[str, Any]],
        volume       : np.ndarray = None,
    ) -> np.ndarray:
        """
        Vectorized volume spike signal generation for all combinations.

        Requires volume data. Computes rolling average volume for all
        unique windows simultaneously, then generates signals.

        Args:
            close_prices : (n_bars,) float64
            params_list  : list of param dicts
            volume       : (n_bars,) float64 — required for this strategy

        Returns:
            signal_matrix : (n_combos, n_bars) float64
        """
        from engine.numpy_backtester import _rolling_mean_single

        if volume is None:
            raise ValueError(
                "VolumeSpikeMomentumStrategy requires volume data. "
                "Pass volume array to generate_signals_numpy()."
            )

        n_bars   = len(close_prices)
        n_combos = len(params_list)

        # Bar returns — shape (n_bars,)
        bar_returns        = np.zeros(n_bars)
        bar_returns[1:]    = np.diff(close_prices) / np.maximum(close_prices[:-1], 1e-10)

        # Unique volume windows — compute once, reuse
        all_vol_windows    = np.array(
            [p["volume_window"] for p in params_list], dtype=np.int64
        )
        unique_vol_windows = np.unique(all_vol_windows)

        # Precompute rolling average volume for each unique window
        # shape: (n_unique_windows, n_bars)
        avg_volumes    = np.full((len(unique_vol_windows), n_bars), np.nan)
        window_to_idx  = {}

        for i, w in enumerate(unique_vol_windows):
            avg_volumes[i] = _rolling_mean_single(volume, int(w))
            window_to_idx[int(w)] = i

        signal_matrix = np.zeros((n_combos, n_bars), dtype=np.float64)

        for i, params in enumerate(params_list):
            vol_window      = int(params["volume_window"])
            spike_threshold = float(params["spike_threshold"])

            avg_vol      = avg_volumes[window_to_idx[vol_window]]
            volume_ratio = np.where(
                avg_vol > 0,
                volume / avg_vol,
                np.nan,
            )

            valid = (
                ~np.isnan(volume_ratio) &
                ~np.isnan(bar_returns)
            )

            spike = valid & (volume_ratio > spike_threshold)

            signal_matrix[i, spike & (bar_returns > 0)] =  1.0
            signal_matrix[i, spike & (bar_returns < 0)] = -1.0

        return signal_matrix

    @classmethod
    def get_optimization_grid(cls):
        return {
            "volume_window"  : list(range(5, 55)),
            "spike_threshold": [x / 10 for x in range(10, 41)],
        }

    @classmethod
    def get_param_names(cls):
        return ["volume_window", "spike_threshold"]

    @classmethod
    def get_default_params(cls):
        return {
            "volume_window"  : 20,
            "spike_threshold": 2.0,
        }