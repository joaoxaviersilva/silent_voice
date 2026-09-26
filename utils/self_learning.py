import json
from pathlib import Path


class SelfLearning:
    def __init__(self, path="data/words.json"):
        self.path = Path(path)
        self.memory = {}

        self.load()

    def load(self):
        if not self.path.exists():
            return

        try:
            with self.path.open("r", encoding="utf-8") as file:
                data = json.load(file)

            if not isinstance(data, dict):
                raise ValueError(
                    "O arquivo da memória não contém um dicionário."
                )

            self.memory = {
                word: count
                for word, count in data.items()
                if (
                    isinstance(word, str)
                    and isinstance(count, int)
                    and count > 0
                )
            }

        except (OSError, json.JSONDecodeError, ValueError) as error:
            print(f"Aviso: falha ao carregar memória de palavras: {error}")
            self.memory = {}

    def add(self, word):
        normalized_word = str(word).strip().upper()

        if not normalized_word:
            return

        self.memory[normalized_word] = (
            self.memory.get(normalized_word, 0) + 1
        )

        self.save()

    def boost(self, vocab):
        normalized_vocab = []

        for word in vocab:
            normalized_word = str(word).strip().upper()

            if not normalized_word:
                continue

            if normalized_word in normalized_vocab:
                continue

            normalized_vocab.append(normalized_word)

        return sorted(
            normalized_vocab,
            key=lambda word: self.memory.get(word, 0),
            reverse=True,
        )

    def save(self):
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = self.path.with_suffix(
            self.path.suffix + ".tmp"
        )

        try:
            with temporary_path.open("w", encoding="utf-8") as file:
                json.dump(
                    self.memory,
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

            temporary_path.replace(self.path)

        except OSError as error:
            print(f"Aviso: falha ao salvar memória de palavras: {error}")