import cv2

from vision.detector import FaceMeshDetector
from vision.roi import crop_mouth


def main():
    detector = FaceMeshDetector()

    # Se quiser usar a Iriun depois, basta trocar para 1
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro ao abrir a câmera.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        landmarks = detector.extract(frame)

        if landmarks is not None:
            roi = crop_mouth(frame, landmarks)

            if roi is not None:
                cv2.imshow("Boca", roi)

            # Desenha os landmarks da boca
            for x, y in landmarks:
                cv2.circle(
                    frame,
                    (int(x), int(y)),
                    2,
                    (0, 255, 0),
                    -1,
                )

        cv2.imshow("Camera", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
