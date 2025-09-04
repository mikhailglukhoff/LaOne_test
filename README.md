# Fashion Product Images Test Assignment

## Описание

В задании используется датасет **Fashion Product Images (Small)**:  
[https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)

При первом запуске датасет автоматически скачивается и распаковывается в:

- `data/raw/images/`  
- `data/raw/styles.csv`

## Запуск

```bash
uv run main.py
```

Эмбеддинги сохраняются в
- `data/processed/dataset.parquet`

Для пересоздания эмбеддингов используйте:

_extract_embeddings("dataset.parquet", force=True)_
