from data.load import check_images, clear_data, extract_csv
from data.preprocess import add_image_path
from features.visual import extract_image_embeddings
from settings import PROCESSED_DATA_DIR

df = extract_csv()
df = check_images(df)
df = clear_data(df)

print(df.columns)
df = add_image_path(df)
df = extract_image_embeddings(df)

output_file = PROCESSED_DATA_DIR / "dataset.parquet"
df.to_parquet(output_file, index=False)
print("✅ Embeddings extracted:", df.shape)
