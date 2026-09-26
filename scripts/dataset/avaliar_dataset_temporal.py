from pathlib import Path

import numpy as np

from vision.temporal_matcher import TemporalLipMatcher


DATASET_PATH = Path("data/lip_samples")


def main():
    matcher = TemporalLipMatcher(
        dataset_path=DATASET_PATH,
    )

    samples = []

    for word, word_samples in matcher.samples.items():
        for index, sample in enumerate(word_samples):
            samples.append(
                {
                    "word": word,
                    "index": index,
                    "sample": sample,
                }
            )

    total = len(samples)
    correct = 0
    results = []

    print("=" * 70)
    print("Avaliação cruzada do dataset temporal")
    print("=" * 70)
    print(f"Total de amostras: {total}\n")

    for number, item in enumerate(samples, start=1):
        true_word = item["word"]
        true_index = item["index"]
        test_sample = item["sample"]

        distances_by_word = {}

        for candidate_word, candidate_samples in matcher.samples.items():
            distances = []

            for candidate_index, candidate_sample in enumerate(
                candidate_samples
            ):
                same_sample = (
                    candidate_word == true_word
                    and candidate_index == true_index
                )

                if same_sample:
                    continue

                distance = matcher.dtw_distance(
                    test_sample,
                    candidate_sample,
                )

                distances.append(distance)

            if not distances:
                continue

            distances.sort()

            neighbors = min(
                matcher.neighbors,
                len(distances),
            )

            distances_by_word[candidate_word] = float(
                np.mean(distances[:neighbors])
            )

        ranking = sorted(
            distances_by_word.items(),
            key=lambda item: item[1],
        )

        predicted_word = ranking[0][0]
        first_distance = ranking[0][1]

        second_distance = (
            ranking[1][1]
            if len(ranking) > 1
            else float("inf")
        )

        margin = second_distance - first_distance
        is_correct = predicted_word == true_word

        if is_correct:
            correct += 1

        results.append(
            {
                "true_word": true_word,
                "predicted_word": predicted_word,
                "distance": first_distance,
                "margin": margin,
                "correct": is_correct,
            }
        )

        status = "OK" if is_correct else "ERRO"

        print(
            f"{number:02d}/{total} | "
            f"Real: {true_word:<7} | "
            f"Prevista: {predicted_word:<7} | "
            f"Distância: {first_distance:.6f} | "
            f"Margem: {margin:.6f} | "
            f"{status}"
        )

    accuracy = correct / total if total else 0.0

    print("\n" + "=" * 70)
    print("Resultado geral")
    print("=" * 70)
    print(f"Acertos: {correct}/{total}")
    print(f"Acurácia: {accuracy * 100:.2f}%")

    print("\nEstatísticas por palavra:")

    for word in sorted(matcher.samples):
        word_results = [
            result
            for result in results
            if result["true_word"] == word
        ]

        word_correct = sum(
            result["correct"]
            for result in word_results
        )

        distances = np.array(
            [
                result["distance"]
                for result in word_results
            ],
            dtype=np.float64,
        )

        margins = np.array(
            [
                result["margin"]
                for result in word_results
            ],
            dtype=np.float64,
        )

        print(f"\n{word}")
        print(
            f"  Acertos: "
            f"{word_correct}/{len(word_results)}"
        )
        print(
            f"  Distância média: "
            f"{distances.mean():.6f}"
        )
        print(
            f"  Maior distância correta: "
            f"{distances.max():.6f}"
        )
        print(
            f"  Margem média: "
            f"{margins.mean():.6f}"
        )
        print(
            f"  Menor margem: "
            f"{margins.min():.6f}"
        )


if __name__ == "__main__":
    main()