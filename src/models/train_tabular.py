from data.load import check_images, clear_data, extract_csv
from data.preprocess import assign_rating, create_target
from features.tabular import encode_categorical, scale_numeric, select_columns, train_catboost

df = extract_csv()
df = check_images(df)
df = clear_data(df)

df = assign_rating(
    df,
    {
        'masterCategory': ['Apparel', 'Accessories'],
        'season'        : ['Summer', 'Fall'],
        'usage'         : ['Casual', 'Ethnic'],
        'gender'        : ['Women']
        }
    )
df = create_target(df, top_quantile=0.8)

# --- Target analysis ---
print("\n📊 Dataset size:", len(df))
print("✅ Target distribution:")
print(df['target'].value_counts())
print(df['target'].value_counts(normalize=True).map(lambda x: f"{x:.2%}"))

print("\n📈 Rating stats by target:")
print(df.groupby('target')['rating'].describe())

df = select_columns(df)
df = scale_numeric(df)
df = encode_categorical(df)
print(df.head(), df.shape)

model = train_catboost(df)
