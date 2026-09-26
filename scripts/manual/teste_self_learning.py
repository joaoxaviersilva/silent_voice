from pathlib import Path
from tempfile import TemporaryDirectory

from utils.self_learning import SelfLearning


def main():
    with TemporaryDirectory() as temporary_directory:
        memory_path = (
            Path(temporary_directory)
            / "words.json"
        )

        learning = SelfLearning(memory_path)

        learning.add("ÁGUA")
        learning.add("água")
        learning.add(" água ")
        learning.add("COMER")
        learning.add("")

        vocab = [
            "DORMIR",
            "COMER",
            "ÁGUA",
            "FALAR",
        ]

        boosted_vocab = learning.boost(vocab)

        print("Memória:", learning.memory)
        print("Vocabulário original:", vocab)
        print("Vocabulário priorizado:", boosted_vocab)

        reloaded_learning = SelfLearning(memory_path)

        print(
            "Memória após recarregar:",
            reloaded_learning.memory,
        )

        print(
            "Vocabulário após recarregar:",
            reloaded_learning.boost(vocab),
        )


if __name__ == "__main__":
    main()