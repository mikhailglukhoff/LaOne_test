from data.extract_embeddings import extract_embeddings
from data.load import prepare_data
from features.cluster import cluster_embeddings

from models.train_models import train_combined, train_tabular, train_visual

df = prepare_data()

extract_embeddings("dataset.parquet", force=False)

train_tabular(df)
train_visual(df)
train_combined(df)

cluster_embeddings(df, n_clusters=10, top_n=10)
