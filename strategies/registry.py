from strategies.moving_average_cross import MovingAverageCrossStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.volume_spike_momentum import VolumeSpikeMomentumStrategy

AVAILABLE_STRATEGIES = {
    "moving_average_cross": MovingAverageCrossStrategy,
    "mean_reversion": MeanReversionStrategy,
    "volume_spike_momentum": VolumeSpikeMomentumStrategy,
}