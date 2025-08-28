import pandas as pd
from sklearn.preprocessing import StandardScaler

from .settings import categorical_cols, numeric_cols, target_col


def select_columns(
    df: pd.DataFrame,
    num: list = numeric_cols,
    cat: list = categorical_cols,
    target: list = target_col
    ) -> pd.DataFrame:
    return df[num + cat + target]


def scale_numeric(
    df: pd.DataFrame,
    num: list = numeric_cols
    ) -> pd.DataFrame:
    if not num:
        return df
    scaler = StandardScaler()
    df[num] = scaler.fit_transform(df[num])
    return df


def encode_categorical(
    df: pd.DataFrame,
    cat: list = categorical_cols
    ) -> pd.DataFrame:
    if not cat:
        raise ValueError("List of categorical columns is empty. Need at least one column.")

    for col in cat:
        df[col] = df[col].astype(str)
    return df
