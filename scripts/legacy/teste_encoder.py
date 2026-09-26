import cv2

from utils.multimodal_encoder import MultiModalEncoder


def main():
    encoder = MultiModalEncoder()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro ao abrir a câmera.")
        return

    print("Pressione ESPAÇO para capturar uma imagem.")
    print("Pressione ESC para sair.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Erro ao capturar frame.")
            break

        cv2.imshow("Teste do encoder", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        if key == 32:
            embedding_imagem = encoder.encode_images([frame])

            palavras = ["agua", "comer", "bom dia"]
            embeddings_texto = encoder.encode_text(palavras)

            print("Embedding da imagem:")
            print("Shape:", embedding_imagem.shape)
            print("Tipo:", embedding_imagem.dtype)

            print("\nEmbeddings de texto:")
            print("Shape:", embeddings_texto.shape)
            print("Tipo:", embeddings_texto.dtype)

            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()