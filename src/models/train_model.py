import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split

from features.settings import categorical_cols, target_col


def train_catboost(df: pd.DataFrame) -> CatBoostClassifier:
    X = df.drop(columns=target_col)
    y = df[target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
        )

    train_pool = Pool(X_train, y_train, cat_features=categorical_cols)
    val_pool = Pool(X_val, y_val, cat_features=categorical_cols)

    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        class_weights=[1, 10],
        verbose=True
        )

    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)
    return model
