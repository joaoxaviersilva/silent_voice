import cv2

from vision.detector import FaceMeshDetector
from vision.features import extract_sequence
from vision.temporal_matcher import TemporalLipMatcher


CAMERA_INDEX = 1
CAPTURE_FRAMES = 40


def draw_text(frame, text, y, scale=0.8):
    cv2.putText(
        frame,
        text,
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def countdown(cap):
    for number in range(3, 0, -1):
        start = cv2.getTickCount()

        while True:
            ret, frame = cap.read()

            if not ret:
                return False

            draw_text(
                frame,
                f"Prepare-se: {number}",
                60,
                scale=1.2,
            )

            cv2.imshow(
                "Teste temporal - Silent Voice",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                return False

            elapsed = (
                cv2.getTickCount() - start
            ) / cv2.getTickFrequency()

            if elapsed >= 1:
                break

    return True


def capture_sequence(cap, detector):
    landmarks_sequence = []

    while len(landmarks_sequence) < CAPTURE_FRAMES:
        ret, frame = cap.read()

        if not ret:
            continue

        landmarks = detector.extract(frame)

        if landmarks is not None:
            landmarks_sequence.append(landmarks)

        draw_text(
            frame,
            "ARTICULE A PALAVRA UMA VEZ",
            50,
        )

        draw_text(
            frame,
            (
                f"Frames: {len(landmarks_sequence)}/"
                f"{CAPTURE_FRAMES}"
            ),
            90,
        )

        progress = len(landmarks_sequence) / CAPTURE_FRAMES

        cv2.rectangle(
            frame,
            (0, frame.shape[0] - 20),
            (
                int(frame.shape[1] * progress),
                frame.shape[0],
            ),
            (0, 255, 0),
            -1,
        )

        cv2.imshow(
            "Teste temporal - Silent Voice",
            frame,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            return None

    return landmarks_sequence


def main():
    print("Carregando amostras...")

    matcher = TemporalLipMatcher()
    detector = FaceMeshDetector()

    print("\nDataset carregado:")

    for word, quantity in matcher.dataset_summary().items():
        print(f"- {word}: {quantity} amostras")

    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW,
    )

    if not cap.isOpened():
        print("Erro: não foi possível abrir a Iriun Webcam.")
        return

    print("\nPressione ESPAÇO para iniciar.")
    print("Articule PIZZA uma única vez.")
    print("Pressione ESC para sair.")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                continue

            draw_text(
                frame,
                "Pressione ESPACO para iniciar",
                50,
            )

            cv2.imshow(
                "Teste temporal - Silent Voice",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                return

            if key == 32:
                break

        if not countdown(cap):
            return

        landmarks_sequence = capture_sequence(
            cap,
            detector,
        )

        if landmarks_sequence is None:
            return

        features = extract_sequence(
            landmarks_sequence
        )

        print("\nShape capturado:", features.shape)
        print("Comparando com as amostras...")

        predicted_word, distances = matcher.predict(
            features
        )

        ranking = sorted(
            distances.items(),
            key=lambda item: item[1],
        )

        print("\nRanking temporal:")

        for position, (word, distance) in enumerate(
            ranking,
            start=1,
        ):
            print(
                f"{position}. {word}: "
                f"distância {distance:.6f}"
            )

        print("\nPalavra reconhecida:", predicted_word)

        if len(ranking) >= 2:
            first_distance = ranking[0][1]
            second_distance = ranking[1][1]

            difference = second_distance - first_distance

            print(
                f"Diferença para o segundo lugar: "
                f"{difference:.6f}"
            )

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()