from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "src" / "models" / "catboost_model"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

STYLES_CSV_PATH = RAW_DATA_DIR / "styles.csv"
IMAGES_RAW_DIR = RAW_DATA_DIR / "images"

target_col = ['target']

numeric_cols = [
    'year'
    ]
categorical_cols = [
    'gender',
    'masterCategory',
    'subCategory',
    'articleType',
    'baseColour',
    'season',
    'usage'
    ]
