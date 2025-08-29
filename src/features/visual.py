# src/features/visual.py
from pathlib import Path
from typing import Literal
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np

try:
    import clip
except ImportError:
    clip = None  # CLIP нужно ставить отдельно, если используем

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_pretrained_model(model_name: Literal["resnet50", "efficientnet_b0", "clip"] = "resnet50"):
    """Load pretrained model and return model + transform"""
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model = nn.Sequential(*list(model.children())[:-1])  # remove final classifier
        model.eval().to(DEVICE)
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                    )
                ]
            )
        return model, transform

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model = nn.Sequential(*list(model.children())[:-1])
        model.eval().to(DEVICE)
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                    )
                ]
            )
        return model, transform

    elif model_name == "clip":
        if clip is None:
            raise ImportError("CLIP not installed. pip install git+https://github.com/openai/CLIP.git")
        model, preprocess = clip.load("ViT-B/32", device=DEVICE)
        return model, preprocess

    else:
        raise ValueError(f"Unknown model_name: {model_name}")


def extract_image_embeddings(
    df: pd.DataFrame, img_col: str = "image_path",
    model_name: Literal["resnet50", "efficientnet_b0", "clip"] = "resnet50",
    batch_size: int = 32
    ) -> pd.DataFrame:
    """
    Extract embeddings for each image in df[img_col].
    Returns df with embedding columns: emb_0, emb_1, ...
    """
    model, transform = get_pretrained_model(model_name)
    embeddings = []

    img_paths = df[img_col].tolist()
    for i in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[i:i + batch_size]
        batch_imgs = []

        for path in batch_paths:
            img = Image.open(path).convert("RGB")
            batch_imgs.append(transform(img) if model_name != "clip" else transform(img).unsqueeze(0))

        batch_tensor = torch.stack(batch_imgs).to(DEVICE) if model_name != "clip" else torch.cat(batch_imgs, dim=0)

        with torch.no_grad():
            if model_name == "clip":
                batch_emb = model.encode_image(batch_tensor)
            else:
                batch_emb = model(batch_tensor).squeeze(-1).squeeze(-1)  # remove HxW dims

            batch_emb = batch_emb.cpu().numpy()
            embeddings.append(batch_emb)

    embeddings = np.vstack(embeddings)
    emb_df = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(embeddings.shape[1])])
    return pd.concat([df.reset_index(drop=True), emb_df], axis=1)
