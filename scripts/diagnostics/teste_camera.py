import cv2


def main():
    camera_index = 1

    cap = cv2.VideoCapture(
        camera_index,
        cv2.CAP_DSHOW,
    )

    if not cap.isOpened():
        print("Erro: não foi possível abrir a Iriun Webcam.")
        return

    print("Iriun Webcam aberta.")
    print("Pressione ESC para sair.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Erro ao capturar imagem da Iriun.")
            break

        cv2.imshow("Iriun Webcam", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()