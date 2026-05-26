import pickle
import os


class PhraseMemory:
    def __init__(self, path="data/phrases.pkl"):
        self.path = path
        self.memory = {}

        if os.path.exists(path):
            with open(path, "rb") as f:
                self.memory = pickle.load(f)

    def add(self, phrase):
        self.memory[phrase] = self.memory.get(phrase, 0) + 1
        self.save()

    def suggest(self, partial):
        ranked = sorted(self.memory.items(), key=lambda x: x[1], reverse=True)
        for phrase, _ in ranked:
            if phrase.startswith(partial):
                return phrase
        return None

    def save(self):
        os.makedirs("data", exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump(self.memory, f)
