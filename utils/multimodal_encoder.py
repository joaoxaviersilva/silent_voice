import torch
import open_clip


class MultiModalEncoder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )

        self.model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

    def encode_images(self, images):
        imgs = [self.preprocess(img).unsqueeze(0) for img in images]
        imgs = torch.cat(imgs).to(self.device)

        with torch.no_grad():
            emb = self.model.encode_image(imgs)

        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.mean(dim=0).cpu().numpy()

    def encode_text(self, words):
        tokens = self.tokenizer(words).to(self.device)

        with torch.no_grad():
            emb = self.model.encode_text(tokens)

        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()
