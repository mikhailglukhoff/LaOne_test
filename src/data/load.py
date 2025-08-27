import pandas as pd


def load_csv():
    df = pd.read_csv("data/styles.csv", on_bad_lines="skip")
    print(df.head())
    return df
