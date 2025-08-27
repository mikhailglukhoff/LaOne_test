import os

import pandas as pd


def extract_csv() -> pd.DataFrame:
    df = pd.read_csv("data/raw/styles.csv", on_bad_lines="skip")
    # print(df.head())
    # print(df.info())
    # print(df.describe())
    return df


def check_images(df: pd.DataFrame, images_dir: str) -> pd.DataFrame:
    images = [int(os.path.splitext(f)[0]) for f in os.listdir(images_dir) if f.endswith(".jpg")]
    df_images = pd.DataFrame(images, columns=['id'])

    return df[df["id"].isin(images)].copy()
