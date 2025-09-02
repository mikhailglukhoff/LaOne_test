import pandas as pd
from sklearn.cluster import KMeans

from data.load import categorical_cols, numeric_cols
from settings import PROCESSED_DATA_DIR


def cluster_embeddings(
    df: pd.DataFrame,
    n_clusters: int = 10,
    top_n: int = 5
    ) -> pd.DataFrame:
    def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
        df_visual = pd.read_parquet(PROCESSED_DATA_DIR / "dataset.parquet")

        # 2. Добавляем эмбеддинги к табличным данным
        cols_to_add = [c for c in df_visual.columns if c not in df.columns]
        df = df.merge(df_visual[["id"] + cols_to_add], on="id", how="left")

        # 3. Формируем признаки
        emb_cols = [c for c in df.columns if c.startswith("emb_")]
        feature_cols = numeric_cols + categorical_cols + emb_cols

        df = df.drop(columns=["id"])
        df = df.dropna()
        return df

    df = _prepare_df(df)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    X = df[emb_cols].values

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X)

    cluster_stats = (
        df.groupby("cluster")["target"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .rename(
            columns={
                0: "unsuccessful",
                1: "successful"
                }
            )
    )

    print("\nSuccess rate per cluster:")
    print(cluster_stats.sort_values("successful", ascending=False).head(top_n))

    return df
