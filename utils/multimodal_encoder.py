import cv2
import open_clip
import torch

from PIL import Image


class MultiModalEncoder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="openai",
        )

        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

    def encode_images(self, images):
        if not images:
            raise ValueError("A lista de imagens não pode estar vazia.")

        processed_images = []

        for image in images:
            if image is None or image.size == 0:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)

            processed_image = self.preprocess(image_pil)
            processed_images.append(processed_image.unsqueeze(0))

        if not processed_images:
            raise ValueError("Nenhuma imagem válida foi recebida pelo encoder.")

        image_tensor = torch.cat(processed_images).to(self.device)

        with torch.no_grad():
            embeddings = self.model.encode_image(image_tensor)

        embeddings = embeddings / embeddings.norm(
            dim=-1,
            keepdim=True,
        )

        return embeddings.mean(dim=0).cpu().numpy()

    def encode_text(self, words):
        if not words:
            raise ValueError("A lista de palavras não pode estar vazia.")

        tokens = self.tokenizer(words).to(self.device)

        with torch.no_grad():
            embeddings = self.model.encode_text(tokens)

        embeddings = embeddings / embeddings.norm(
            dim=-1,
            keepdim=True,
        )

        return embeddings.cpu().numpy()