import json
from pathlib import Path


class PhraseMemory:
    def __init__(self, path="data/phrases.json"):
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
                phrase: count
                for phrase, count in data.items()
                if (
                    isinstance(phrase, str)
                    and isinstance(count, int)
                    and count > 0
                )
            }

        except (OSError, json.JSONDecodeError, ValueError) as error:
            print(f"Aviso: falha ao carregar memória de frases: {error}")
            self.memory = {}

    def add(self, phrase):
        phrase = str(phrase).strip()

        if not phrase:
            return

        self.memory[phrase] = self.memory.get(phrase, 0) + 1
        self.save()

    def suggest(self, partial):
        partial = str(partial).strip()

        if not partial:
            return None

        normalized_partial = partial.casefold()

        matching_phrases = [
            (phrase, count)
            for phrase, count in self.memory.items()
            if phrase.casefold().startswith(normalized_partial)
        ]

        if not matching_phrases:
            return None

        best_phrase, _ = max(
            matching_phrases,
            key=lambda item: item[1],
        )

        return best_phrase

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
            print(f"Aviso: falha ao salvar memória de frases: {error}")