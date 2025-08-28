import numpy as np
import pandas as pd


def assign_rating(df: pd.DataFrame, categories: dict[str, list] | None = None) -> pd.DataFrame:
    n = df.shape[0]
    # df['rating'] = np.nan

    if not categories:
        df['rating'] = np.random.choice(np.arange(1, 5.1, 0.1), size=n)

    else:
        for col, vals in categories.items():
            idx = df[df[col].isin(vals)].index
            df.loc[idx, 'rating'] = np.random.choice(np.arange(1, 5.1, 0.1), size=len(idx))

            # # Можно добавить назначение рейтинга для оставшихся строк, если нужно
            # remaining_idx = df[df['rating'].isna()].index
            # if len(remaining_idx) > 0:
            #     df.loc[remaining_idx, 'rating'] = np.random.choice(np.arange(1, 5.1, 0.1), size=len(remaining_idx))

    return df


def create_target(df: pd.DataFrame, top_quantile: float = 0.8) -> pd.DataFrame:
    threshold = df['rating'].quantile(top_quantile)
    df['target'] = (df['rating'] >= threshold)
    return df
