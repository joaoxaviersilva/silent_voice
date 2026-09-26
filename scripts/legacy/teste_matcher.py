import cv2

from utils.multimodal_encoder import MultiModalEncoder
from vision.detector import FaceMeshDetector
from vision.multimodal_matcher import MultiModalMatcher
from vision.roi import crop_mouth


def main():
    encoder = MultiModalEncoder()
    matcher = MultiModalMatcher(encoder)
    detector = FaceMeshDetector()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro ao abrir a câmera.")
        return

    mouth_frames = []

    print("Pressione ESPAÇO para capturar 12 imagens da boca.")
    print("Pressione ESC para sair.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Erro ao capturar frame.")
            break

        landmarks = detector.extract(frame)

        if landmarks is not None:
            mouth = crop_mouth(frame, landmarks)

            if mouth is not None:
                cv2.imshow("Região da boca", mouth)

        cv2.imshow("Teste do matcher", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        if key == 32:
            mouth_frames = []

            while len(mouth_frames) < 12:
                ret, frame = cap.read()

                if not ret:
                    break

                landmarks = detector.extract(frame)

                if landmarks is None:
                    continue

                mouth = crop_mouth(frame, landmarks)

                if mouth is not None:
                    mouth_frames.append(mouth)

            break

    cap.release()
    cv2.destroyAllWindows()

    if len(mouth_frames) < 12:
        print("Não foi possível capturar imagens suficientes da boca.")
        return

    words = ["agua", "comer", "bom dia"]

    probabilities = matcher.match(mouth_frames, words)

    print("\nResultados:")

    for word, probability in sorted(
        probabilities.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"{word}: {probability:.4f}")

    print(f"\nSoma: {sum(probabilities.values()):.4f}")


if __name__ == "__main__":
    main()