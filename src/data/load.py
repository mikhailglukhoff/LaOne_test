import os
import shutil
from pathlib import Path

import pandas as pd

from src.data.preprocess import assign_rating, create_target
from src.settings import IMAGES_RAW_DIR, STYLES_CSV_PATH

target_col = ['target']

numeric_cols = [
    'year'
    ]
categorical_cols = [
    'gender',
    'masterCategory',
    'subCategory',
    'articleType',
    'baseColour',
    'season',
    'usage'
    ]

categories = {
    'masterCategory': ['Apparel', 'Accessories'],
    'season'        : ['Summer', 'Fall'],
    'usage'         : ['Casual', 'Ethnic'],
    'gender'        : ['Women']
    }


def extract_csv(path_to_csv: Path = STYLES_CSV_PATH) -> pd.DataFrame:
    """
    extract data from csv file to pandas dataframe
    :param path_to_csv: path to csv file
    :return: pandas dataframe
    """

    def _load_dataset_from_kaggle():
        import kagglehub
        dataset_path = Path(
            kagglehub.dataset_download("paramaggarwal/fashion-product-images-small", force_download=False)
            )

        csv_file = next(dataset_path.glob("*.csv"))
        STYLES_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        IMAGES_RAW_DIR.mkdir(parents=True, exist_ok=True)

        shutil.copy(csv_file, STYLES_CSV_PATH)

        images_folder = dataset_path / "images"
        for img_file in images_folder.glob("*"):
            shutil.copy(img_file, IMAGES_RAW_DIR / img_file.name)
        else:
            pass
            # print("Images folder not found in dataset!")

        cache_dir = Path.home() / ".cache/kagglehub/datasets/paramaggarwal/fashion-product-images-small/versions/1"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            # print("Specific Kagglehub dataset cache cleared.")

    if not os.path.exists(path_to_csv):
        _load_dataset_from_kaggle()

    df = pd.read_csv(
        path_to_csv,
        na_values=("NA", "Null", ""),
        on_bad_lines="skip"
        )
    return df


def check_images(df: pd.DataFrame, images_dir: Path = IMAGES_RAW_DIR) -> pd.DataFrame:
    """
    check images for existence
    :param df: pandas dataframe with id's
    :param images_dir: path to images folder
    :return: dataframe with existing images id's
    """
    images = [
        int(Path(f).stem)
        for f in os.listdir(images_dir)
        if f.endswith(".jpg")
        ]
    return df[df["id"].isin(images)]


def clear_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    clear data from pandas dataframe. drop unnecessary columns,Nan and Nulls
    :param df: pandas dataframe
    :return: pandas dataframe
    """
    columns = numeric_cols + categorical_cols + ["id"]
    df = df[columns]
    return df.dropna()


def prepare_data() -> pd.DataFrame:
    df = extract_csv()
    df = check_images(df)
    df = clear_data(df)

    df = assign_rating(
        df,
        categories
        )
    df = create_target(df, top_quantile=0.8)
    print('Dataset prepared.')
    return df
