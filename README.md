# Uni Project - German Used Car Price Prediction

Data Source:

- Scraped from AutoScout24 - <https://www.kaggle.com/datasets/wspirat/germany-used-cars-dataset-2023/data>

## Pipeline usage

Run the full cleaning, transformation, dataset export, and model-evaluation pipeline:

```bash
uv sync
uv run python scripts/car_price_pipeline.py
```

Generated outputs:

- `data/data_partial_clean.csv`
- `data/cleaned_vw_golf.csv`
- `data/cleaned_opel_astra.csv`
- `data/cleaned_opel_corsa.csv`
- `data/transformed_cleaned_vw_golf.csv`
- `data/transformed_cleaned_opel_astra.csv`
- `data/transformed_cleaned_opel_corsa.csv`
- `data/model_metrics.csv` (Linear Regression, XGBoost, LightGBM on train/validation/test; models train on `log1p(price_in_euro)` and are evaluated back in EUR)

Multivariate analysis notebooks:

- `multivariate_analysis_vw_golf.ipynb`
- `multivariate_analysis_opel_astra.ipynb`
- `multivariate_analysis_opel_corsa.ipynb`
