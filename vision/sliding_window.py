class SlidingWindow:
    def __init__(self, window_size=12, step=4):
        self.window_size = window_size
        self.step = step

    def generate(self, sequence):
        windows = []
        for i in range(0, len(sequence) - self.window_size + 1, self.step):
            windows.append((i, sequence[i : i + self.window_size]))
        return windows
