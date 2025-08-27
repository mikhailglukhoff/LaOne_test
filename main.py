from data.load import check_images
from data.preprocess import create_target
from src.data.load import extract_csv

df = extract_csv()
print(df.shape)
df = check_images(df, 'data/raw/images/')
print(df.shape)
target_df = create_target(df)
print(df["target"].value_counts(normalize=True))
print(target_df['target'].describe())
