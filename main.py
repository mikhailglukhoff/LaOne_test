from data.load import check_images, clear_data, extract_csv
from data.preprocess import assign_rating, create_target
from features.tabular import encode_categorical, scale_numeric, select_columns
from models.train_model import train_catboost

df = extract_csv()
df = check_images(df, 'data/raw/images/')
df = clear_data(df)

df = assign_rating(
    df,
    {
        'masterCategory': ['Apparel', 'Footwear'],
        'articleType'   : ['Shirts', 'Tshirts', 'Tops', 'Bra'],
        'season'        : ['Summer', 'Fall'],
        'year'          : [2019, 2018, 2017, 2014],
        'usage'         : ['Casual', 'Sports']
        }
    )
df = create_target(df, top_quantile=0.8)
popularity = (
    df.groupby(['masterCategory', 'articleType', 'season', 'year', 'usage', 'gender'])
    .size()
    .sort_values(ascending=False)
    .head(20)
)
print("Top 20 popular combinations:")
print(popularity)

df = select_columns(df)
df = scale_numeric(df)
df = encode_categorical(df)
print(df.head(), df.shape)

model = train_catboost(df)
