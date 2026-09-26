from pathlib import Path
from tempfile import TemporaryDirectory

from utils.phrase_memory import PhraseMemory


def main():
    with TemporaryDirectory() as temporary_directory:
        memory_path = (
            Path(temporary_directory)
            / "phrases.json"
        )

        memory = PhraseMemory(memory_path)

        memory.add("Eu quero água.")
        memory.add("Eu quero água.")
        memory.add("Eu quero comer.")
        memory.add("Bom dia.")
        memory.add("")

        print(
            "Sugestão para 'Eu quero':",
            memory.suggest("Eu quero"),
        )

        print(
            "Sugestão com letras minúsculas:",
            memory.suggest("eu quero"),
        )

        print(
            "Sugestão inexistente:",
            memory.suggest("Boa noite"),
        )

        print(
            "Sugestão vazia:",
            memory.suggest(""),
        )

        reloaded_memory = PhraseMemory(memory_path)

        print(
            "Memória após recarregar:",
            reloaded_memory.memory,
        )

        print(
            "Sugestão após recarregar:",
            reloaded_memory.suggest("Eu quero"),
        )


if __name__ == "__main__":
    main()