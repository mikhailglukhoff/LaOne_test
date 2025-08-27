import numpy as np
import pandas as pd


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    n = df.shape[0]
    df['rating'] = np.random.choice(np.arange(1, 5.1, 0.1), size=n)

    threshold = df["rating"].quantile(0.8)

    df['target'] = df["rating"] >= threshold
    return df
