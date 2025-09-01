import os
from pathlib import Path

import pandas as pd

from src.settings import IMAGES_RAW_DIR, STYLES_CSV_PATH


def extract_csv(path_to_csv: Path = STYLES_CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(
        path_to_csv,
        na_values=("NA", "Null", ""),
        on_bad_lines="skip"
        )
    return df


def check_images(df: pd.DataFrame, images_dir: Path = IMAGES_RAW_DIR) -> pd.DataFrame:
    images = [
        int(Path(f).stem)
        for f in os.listdir(images_dir)
        if f.endswith(".jpg")
        ]
    return df[df["id"].isin(images)]


def clear_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()
