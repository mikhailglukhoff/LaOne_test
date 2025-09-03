import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
from tqdm import tqdm

from src.data.load import prepare_data
from src.data.preprocess import add_image_path
from src.settings import PROCESSED_DATA_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_embeddings(save_path: str, force: bool = False) -> None:
    """
    extract embeddings from an imageset, if embedding's file doesn't exist yet.
    :param save_path: path to save embedding's file
    :param force: overwrite existing embedding's file
    :return:
    """

    def _extract(
        df: pd.DataFrame,
        img_col: str = "image_path",
        batch_size: int = 32,
        ) -> pd.DataFrame:
        """
        Extract embeddings for each image in df[img_col].
        :param df: dataframe with id's and paths to images
        :param img_col: name of column in df to extract embeddings
        :param batch_size: batch size to extract embeddings
        :return: Returns df with embedding columns: emb_0, emb_1, ...
        """
        model, transform = _get_pretrained_model()
        embeddings = []

        img_paths = df[img_col].tolist()
        n_batches = (len(img_paths) + batch_size - 1) // batch_size

        for i in tqdm(range(0, len(img_paths), batch_size), total=n_batches, desc="Extracting embeddings"):
            batch_paths = img_paths[i:i + batch_size]
            batch_imgs = []

            for path in batch_paths:
                img = Image.open(path).convert("RGB")
                batch_imgs.append(transform(img))

            batch_tensor = torch.stack(batch_imgs).to(DEVICE)

            with torch.no_grad():
                batch_emb = model(batch_tensor)
                batch_emb = batch_emb.squeeze(-1).squeeze(-1)
                batch_emb = batch_emb.cpu().numpy()
                embeddings.append(batch_emb)

        embeddings = np.vstack(embeddings)
        emb_df = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(embeddings.shape[1])])
        return pd.concat([df.reset_index(drop=True), emb_df], axis=1)

    def _get_pretrained_model() -> tuple:
        """
        Load pretrained model and return model + transform
        """
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
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

    output_file = PROCESSED_DATA_DIR / save_path
    if output_file.exists() and not force:
        print("Embeddings already exists.")
        return None
    else:
        df = prepare_data()
        df = add_image_path(df)
        df = _extract(df)

        df.to_parquet(output_file, index=False)
        print("Embeddings extracted:", df.shape)
        return output_file
