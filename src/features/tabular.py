import pandas as pd

from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.load import categorical_cols, numeric_cols, target_col


def select_columns(
    df: pd.DataFrame,
    num: list = numeric_cols,
    cat: list = categorical_cols,
    target: list = target_col
    ) -> pd.DataFrame:
    return df[num + cat + target]


def scale_numeric(
    df: pd.DataFrame,
    num: list = numeric_cols
    ) -> pd.DataFrame:
    if not num:
        return df
    scaler = StandardScaler()
    df[num] = scaler.fit_transform(df[num])
    return df


def encode_categorical(
    df: pd.DataFrame,
    cat: list = categorical_cols
    ) -> pd.DataFrame:
    if not cat:
        raise ValueError("List of categorical columns is empty. Need at least one column.")

    for col in cat:
        df[col] = df[col].astype(str)
    return df


def train_catboost(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_cols: list[str] | None = None,
    save_path: str | None = None,
    ) -> CatBoostClassifier:

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
        )
    if categorical_cols:
        train_pool = Pool(X_train, y_train, cat_features=categorical_cols)
        val_pool = Pool(X_val, y_val, cat_features=categorical_cols)
    else:
        train_pool = Pool(X_train, y_train)
        val_pool = Pool(X_val, y_val)

    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        class_weights=[1, 10],
        verbose=True,
        )

    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)

    # eval
    preds = model.predict(X_val)
    probas = model.predict_proba(X_val)[:, 1]
    print(classification_report(y_val, preds))
    print("ROC-AUC:", roc_auc_score(y_val, probas))

    # feature importance (только если не чисто эмбеддинги)
    if categorical_cols:
        importances = model.get_feature_importance(train_pool, type="FeatureImportance")
        feat_imp_df = pd.DataFrame(
            {
                "feature"   : X_train.columns,
                "importance": importances
                }
            ).sort_values("importance", ascending=False)
        print("\nTop-10 important features:")
        print(feat_imp_df.head(10))

    # save
    if save_path:
        model.save_model(save_path)

    return model
