import pickle
import os


class SelfLearning:
    def __init__(self, path="data/words.pkl"):
        self.path = path
        self.memory = {}

        if os.path.exists(path):
            with open(path, "rb") as f:
                self.memory = pickle.load(f)

    def add(self, word):
        self.memory[word] = self.memory.get(word, 0) + 1
        self.save()

    def boost(self, vocab):
        return sorted(vocab, key=lambda x: self.memory.get(x, 0), reverse=True)

    def save(self):
        os.makedirs("data", exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump(self.memory, f)
