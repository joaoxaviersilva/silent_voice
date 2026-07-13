import numpy as np


class MultiModalMatcher:
    def __init__(self, encoder):
        self.encoder = encoder

    def match(self, frames, words):
        if not words:
            return {}

        v_emb = self.encoder.encode_images(frames)
        t_emb = self.encoder.encode_text(words)

        scores = {}
        for w, e in zip(words, t_emb):
            scores[w] = float(np.dot(v_emb, e))

        total = sum(scores.values()) + 1e-6
        return {k: v / total for k, v in scores.items()}
