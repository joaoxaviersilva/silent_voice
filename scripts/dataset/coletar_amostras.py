import argparse
import re
from pathlib import Path

import cv2
import numpy as np

from vision.detector import FaceMeshDetector
from vision.features import extract_sequence


CAMERA_INDEX = 1
TOTAL_SAMPLES = 8
FRAMES_PER_SAMPLE = 40
MIN_VALID_FRAMES = 30

DATASET_DIRECTORY = Path("data/lip_samples")


def normalize_label(word):
    normalized_word = str(word).strip().upper()

    normalized_word = re.sub(
        r"[^A-ZÀ-Ú0-9_-]+",
        "_",
        normalized_word,
    )

    return normalized_word.strip("_")


def draw_centered_text(
    frame,
    text,
    vertical_position,
    font_scale=1.0,
    thickness=2,
):
    font = cv2.FONT_HERSHEY_SIMPLEX

    text_size, _ = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness,
    )

    text_width = text_size[0]

    horizontal_position = (
        frame.shape[1] - text_width
    ) // 2

    cv2.putText(
        frame,
        text,
        (horizontal_position, vertical_position),
        font,
        font_scale,
        (0, 255, 0),
        thickness,
        cv2.LINE_AA,
    )


def show_countdown(cap, word):
    for number in range(3, 0, -1):
        countdown_started = cv2.getTickCount()

        while True:
            ret, frame = cap.read()

            if not ret:
                return False

            draw_centered_text(
                frame,
                f"Prepare-se para: {word}",
                60,
                font_scale=0.8,
            )

            draw_centered_text(
                frame,
                str(number),
                frame.shape[0] // 2,
                font_scale=3.0,
                thickness=5,
            )

            cv2.imshow(
                "Coletor - Silent Voice",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                return False

            elapsed_seconds = (
                cv2.getTickCount()
                - countdown_started
            ) / cv2.getTickFrequency()

            if elapsed_seconds >= 1:
                break

    return True


def capture_sample(
    cap,
    detector,
    word,
    sample_number,
):
    landmarks_sequence = []

    for frame_number in range(
        FRAMES_PER_SAMPLE
    ):
        ret, frame = cap.read()

        if not ret:
            continue

        landmarks = detector.extract(frame)

        if landmarks is not None:
            landmarks_sequence.append(
                landmarks
            )

        progress = (
            frame_number + 1
        ) / FRAMES_PER_SAMPLE

        progress_width = int(
            progress * frame.shape[1]
        )

        cv2.rectangle(
            frame,
            (0, frame.shape[0] - 20),
            (progress_width, frame.shape[0]),
            (0, 255, 0),
            -1,
        )

        draw_centered_text(
            frame,
            f"ARTICULE: {word}",
            60,
            font_scale=1.0,
            thickness=2,
        )

        draw_centered_text(
            frame,
            (
                f"Amostra {sample_number}/"
                f"{TOTAL_SAMPLES}"
            ),
            100,
            font_scale=0.7,
        )

        cv2.imshow(
            "Coletor - Silent Voice",
            frame,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            return None, True

    if len(landmarks_sequence) < MIN_VALID_FRAMES:
        return None, False

    features = extract_sequence(
        landmarks_sequence
    )

    return features, False


def get_next_sample_number(
    word_directory,
):
    existing_files = list(
        word_directory.glob("sample_*.npy")
    )

    numbers = []

    for file_path in existing_files:
        try:
            number = int(
                file_path.stem.split("_")[-1]
            )
            numbers.append(number)
        except ValueError:
            continue

    if not numbers:
        return 1

    return max(numbers) + 1


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Coleta amostras de movimentos labiais."
        )
    )

    parser.add_argument(
        "word",
        help="Palavra que será articulada.",
    )

    arguments = parser.parse_args()

    word = normalize_label(
        arguments.word
    )

    if not word:
        print("Erro: informe uma palavra válida.")
        return

    word_directory = (
        DATASET_DIRECTORY / word
    )

    word_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    detector = FaceMeshDetector()

    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW,
    )

    if not cap.isOpened():
        print(
            "Erro: não foi possível abrir "
            "a Iriun Webcam."
        )
        return

    first_sample_number = (
        get_next_sample_number(
            word_directory
        )
    )

    saved_samples = 0
    exit_requested = False

    print("=" * 60)
    print("Coletor de amostras do Silent Voice")
    print("=" * 60)
    print(f"Palavra: {word}")
    print(f"Quantidade: {TOTAL_SAMPLES}")
    print(
        "Articule a palavra somente uma vez "
        "em cada amostra."
    )
    print(
        "Mantenha distância e iluminação "
        "parecidas."
    )
    print(
        "Use ESPAÇO para iniciar e ESC para sair."
    )

    try:
        while saved_samples < TOTAL_SAMPLES:
            ret, frame = cap.read()

            if not ret:
                print(
                    "Erro ao capturar imagem."
                )
                break

            draw_centered_text(
                frame,
                f"Palavra: {word}",
                60,
                font_scale=1.0,
            )

            draw_centered_text(
                frame,
                (
                    "Pressione ESPACO para "
                    "gravar a proxima amostra"
                ),
                110,
                font_scale=0.65,
            )

            draw_centered_text(
                frame,
                (
                    f"Concluidas: {saved_samples}/"
                    f"{TOTAL_SAMPLES}"
                ),
                150,
                font_scale=0.7,
            )

            cv2.imshow(
                "Coletor - Silent Voice",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                exit_requested = True
                break

            if key != 32:
                continue

            countdown_completed = (
                show_countdown(
                    cap,
                    word,
                )
            )

            if not countdown_completed:
                exit_requested = True
                break

            sample_number = (
                first_sample_number
                + saved_samples
            )

            features, capture_cancelled = (
                capture_sample(
                    cap,
                    detector,
                    word,
                    saved_samples + 1,
                )
            )

            if capture_cancelled:
                exit_requested = True
                break

            if features is None:
                print(
                    "Amostra descartada: poucos "
                    "frames válidos. Tente novamente."
                )
                continue

            sample_path = word_directory / (
                f"sample_{sample_number:03d}.npy"
            )

            np.save(
                sample_path,
                features,
            )

            saved_samples += 1

            print(
                f"Amostra salva: {sample_path}"
            )
            print(
                f"Shape: {features.shape}"
            )

            cv2.waitKey(500)

    except KeyboardInterrupt:
        exit_requested = True
        print(
            "\nColeta interrompida pelo usuário."
        )

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print(
        f"Amostras salvas nesta execução: "
        f"{saved_samples}/{TOTAL_SAMPLES}"
    )

    if exit_requested:
        print("Coleta encerrada antes do fim.")
    elif saved_samples == TOTAL_SAMPLES:
        print("Coleta concluída com sucesso.")

    print(
        f"Diretório: {word_directory}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()