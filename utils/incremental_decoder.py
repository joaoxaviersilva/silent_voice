class IncrementalDecoder:
    def __init__(self, stability_threshold=3):
        self.current = None
        self.count = 0
        self.threshold = stability_threshold

    def update(self, word):
        if word == self.current:
            self.count += 1
        else:
            self.current = word
            self.count = 1

        if self.count >= self.threshold:
            confirmed = self.current
            self.current = None
            self.count = 0
            return confirmed

        return None
