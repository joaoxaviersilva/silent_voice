from pathlib import Path

import numpy as np


class TemporalLipMatcher:
    def __init__(
        self,
        dataset_path="data/lip_samples",
        shape_weight=0.35,
        movement_weight=0.65,
        neighbors=3,
    ):
        if shape_weight < 0 or movement_weight < 0:
            raise ValueError("Os pesos não podem ser negativos.")

        if shape_weight + movement_weight == 0:
            raise ValueError("Pelo menos um peso deve ser maior que zero.")

        if neighbors < 1:
            raise ValueError("A quantidade de vizinhos deve ser positiva.")

        self.dataset_path = Path(dataset_path)
        self.shape_weight = shape_weight
        self.movement_weight = movement_weight
        self.neighbors = neighbors

        self.samples = {}

        self.load_samples()

    def load_samples(self):
        self.samples = {}

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset não encontrado: {self.dataset_path}"
            )

        for word_directory in sorted(self.dataset_path.iterdir()):
            if not word_directory.is_dir():
                continue

            word = word_directory.name.upper()
            word_samples = []

            for sample_path in sorted(
                word_directory.glob("sample_*.npy")
            ):
                try:
                    sample = np.load(sample_path).astype(np.float32)

                    if sample.ndim != 2:
                        print(
                            f"Aviso: amostra ignorada por formato inválido: "
                            f"{sample_path}"
                        )
                        continue

                    if sample.shape[1] != 160:
                        print(
                            f"Aviso: amostra ignorada. Esperado 160 features, "
                            f"recebido {sample.shape[1]}: {sample_path}"
                        )
                        continue

                    word_samples.append(sample)

                except (OSError, ValueError) as error:
                    print(
                        f"Aviso: não foi possível carregar "
                        f"{sample_path}: {error}"
                    )

            if word_samples:
                self.samples[word] = word_samples

        if not self.samples:
            raise ValueError(
                "Nenhuma amostra válida foi encontrada no dataset."
            )

    def _frame_distance(self, first_frame, second_frame):
        feature_count = len(first_frame)
        half = feature_count // 2

        first_shape = first_frame[:half]
        second_shape = second_frame[:half]

        first_movement = first_frame[half:]
        second_movement = second_frame[half:]

        shape_distance = np.mean(
            np.square(first_shape - second_shape)
        )

        movement_distance = np.mean(
            np.square(first_movement - second_movement)
        )

        weighted_distance = (
            self.shape_weight * shape_distance
            + self.movement_weight * movement_distance
        )

        return float(np.sqrt(weighted_distance))

    def dtw_distance(self, first_sequence, second_sequence):
        first_sequence = np.asarray(
            first_sequence,
            dtype=np.float32,
        )

        second_sequence = np.asarray(
            second_sequence,
            dtype=np.float32,
        )

        if first_sequence.ndim != 2 or second_sequence.ndim != 2:
            raise ValueError(
                "As sequências devem possuir o formato (frames, features)."
            )

        if first_sequence.shape[1] != second_sequence.shape[1]:
            raise ValueError(
                "As sequências possuem quantidades diferentes de features."
            )

        first_length = len(first_sequence)
        second_length = len(second_sequence)

        previous_row = np.full(
            second_length + 1,
            np.inf,
            dtype=np.float64,
        )

        previous_row[0] = 0.0

        for first_index in range(1, first_length + 1):
            current_row = np.full(
                second_length + 1,
                np.inf,
                dtype=np.float64,
            )

            for second_index in range(1, second_length + 1):
                cost = self._frame_distance(
                    first_sequence[first_index - 1],
                    second_sequence[second_index - 1],
                )

                current_row[second_index] = cost + min(
                    previous_row[second_index],
                    current_row[second_index - 1],
                    previous_row[second_index - 1],
                )

            previous_row = current_row

        normalization = first_length + second_length

        return float(
            previous_row[second_length] / normalization
        )

    def compare(self, sequence, allowed_words=None):
        sequence = np.asarray(
            sequence,
            dtype=np.float32,
        )

        if sequence.ndim != 2:
            raise ValueError(
                "A sequência deve possuir o formato (frames, features)."
            )

        if allowed_words:
            allowed_words = {
                str(word).strip().upper()
                for word in allowed_words
            }

        results = {}

        for word, samples in self.samples.items():
            if allowed_words and word not in allowed_words:
                continue

            distances = [
                self.dtw_distance(sequence, sample)
                for sample in samples
            ]

            distances.sort()

            neighbors = min(
                self.neighbors,
                len(distances),
            )

            results[word] = float(
                np.mean(distances[:neighbors])
            )

        return results

    def predict(self, sequence, allowed_words=None):
        distances = self.compare(
            sequence,
            allowed_words=allowed_words,
        )

        if not distances:
            return None, {}

        predicted_word = min(
            distances,
            key=distances.get,
        )

        return predicted_word, distances

    def dataset_summary(self):
        return {
            word: len(samples)
            for word, samples in self.samples.items()
        }