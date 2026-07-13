import cv2

from vision.detector import FaceMeshDetector
from vision.features import extract_sequence


def main():
    detector = FaceMeshDetector()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro ao abrir a câmera.")
        return

    sequence = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        landmarks = detector.extract(frame)

        if landmarks is not None:

            sequence.append(landmarks)

            if len(sequence) > 10:
                sequence.pop(0)

            for x, y in landmarks:
                cv2.circle(
                    frame,
                    (int(x), int(y)),
                    2,
                    (0, 255, 0),
                    -1,
                )

            if len(sequence) >= 2:

                features = extract_sequence(sequence)

                print("-" * 60)
                print("Shape:", features.shape)
                print("Último vetor:")
                print(features[-1])

        cv2.imshow("Features", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
