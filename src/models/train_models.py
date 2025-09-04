import pandas as pd

from catboost import CatBoostClassifier

from src.data.load import categorical_cols, numeric_cols, target_col
from src.features.tabular import train_catboost
from src.settings import PROCESSED_DATA_DIR


def train_tabular(df: pd.DataFrame) -> CatBoostClassifier:
    df = df.drop(columns=["id"])
    X = df.drop(columns=target_col)
    y = df[target_col]

    print("Training tabular model:")
    model = train_catboost(
        X, y,
        categorical_cols=categorical_cols,
        save_path=PROCESSED_DATA_DIR / "catboost_tabular_model"
        )
    return model


def train_visual(df: pd.DataFrame):
    df_visual = pd.read_parquet(PROCESSED_DATA_DIR / "dataset.parquet")

    cols_to_add = [c for c in df_visual.columns if c not in df.columns and c != "id"]
    df = df.merge(df_visual[["id"] + cols_to_add], on="id", how="left")

    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    df = df.drop(columns=["id"])
    X = df[emb_cols]
    y = df[target_col]

    print("Training visual model:")
    model = train_catboost(X, y, save_path=PROCESSED_DATA_DIR / "catboost_visual_model.cbm")
    return model


def train_combined(df: pd.DataFrame):
    df_visual = pd.read_parquet(PROCESSED_DATA_DIR / "dataset.parquet")

    # 2. Добавляем эмбеддинги к табличным данным
    cols_to_add = [c for c in df_visual.columns if c not in df.columns]
    df = df.merge(df_visual[["id"] + cols_to_add], on="id", how="left")

    # 3. Формируем признаки
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    feature_cols = numeric_cols + categorical_cols + emb_cols

    df = df.drop(columns=["id"])
    X = df[feature_cols]
    y = df[target_col]

    print("Training combined model:")
    model = train_catboost(
        X,
        y,
        categorical_cols=categorical_cols,
        save_path=PROCESSED_DATA_DIR / "catboost_combined_model.cbm",
        )
    return model
