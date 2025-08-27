import os
import pandas as pd


def extract_csv() -> pd.DataFrame:
    df = pd.read_csv(
        "data/raw/styles.csv",
        na_values=("NA", "Null", ""),
        on_bad_lines="skip"
        )
    return df


def check_images(df: pd.DataFrame, images_dir: str) -> pd.DataFrame:
    images = [int(os.path.splitext(f)[0]) for f in os.listdir(images_dir) if f.endswith(".jpg")]
    return df[df["id"].isin(images)]


def clear_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()
