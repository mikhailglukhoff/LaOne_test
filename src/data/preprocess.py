import numpy as np
import pandas as pd

from src.settings import IMAGES_RAW_DIR


def assign_rating(df: pd.DataFrame, categories: dict[str, list] | None = None) -> pd.DataFrame:
    n = df.shape[0]

    # базовый рейтинг с шумом вокруг 3.0
    df['rating'] = 3.0 + np.random.normal(0, 0.4, size=n)

    if categories:
        for col, vals in categories.items():
            idx = df[df[col].isin(vals)].index
            # повысим рейтинг для совпавших строк
            df.loc[idx, 'rating'] += np.random.uniform(0.5, 1.5, size=len(idx))

            # добавим шум для лидеров (чтобы не все были "идеальными")
            df.loc[idx, 'rating'] += np.random.normal(0, 0.3, size=len(idx))

    # глобальный шум для всего датасета (моделируем случайные факторы)
    df['rating'] += np.random.normal(0, 0.15, size=n)

    # ограничим диапазон от 1 до 5
    df['rating'] = df['rating'].clip(1, 4.9)

    return df


def create_target(df: pd.DataFrame, top_quantile: float) -> pd.DataFrame:
    threshold = df['rating'].quantile(top_quantile)
    df['target'] = (df['rating'] >= threshold).astype(int)
    df.drop(columns=['rating'], inplace=True)
    return df


def add_image_path(df: pd.DataFrame) -> pd.DataFrame:
    df["image_path"] = df["id"].astype(str).apply(
        lambda img_id: str(IMAGES_RAW_DIR / f"{img_id}.jpg")
        )
    return df
