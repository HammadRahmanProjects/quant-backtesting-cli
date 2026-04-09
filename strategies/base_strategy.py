class BaseStrategy:
    def __init__(self, df):
        self.df = df.copy()

    def generate_signals(self):
        raise NotImplementedError("Must implement generate_signals()")
