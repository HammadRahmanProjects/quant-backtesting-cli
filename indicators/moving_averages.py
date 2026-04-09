def simple_moving_average(series, window):
    return series.rolling(window).mean()
