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
import numpy as np


class MultiModalMatcher:
    def __init__(self, encoder, temperature=0.07):
        if temperature <= 0:
            raise ValueError("A temperatura deve ser maior que zero.")

        self.encoder = encoder
        self.temperature = temperature

    def match(self, frames, words):
        if not frames or not words:
            return {}

        visual_embedding = self.encoder.encode_images(frames)
        text_embeddings = self.encoder.encode_text(words)

        similarities = np.array(
            [
                np.dot(visual_embedding, text_embedding)
                for text_embedding in text_embeddings
            ],
            dtype=np.float32,
        )

        logits = similarities / self.temperature
        logits = logits - np.max(logits)

        exponential_scores = np.exp(logits)
        probabilities = exponential_scores / np.sum(exponential_scores)

        return {
            word: float(probability)
            for word, probability in zip(words, probabilities)
        }