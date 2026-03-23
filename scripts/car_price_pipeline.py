from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "data.csv"
PARTIAL_CLEAN_PATH = DATA_DIR / "data_partial_clean.csv"
METRICS_PATH = DATA_DIR / "model_metrics.csv"

MODEL_IDENTIFIERS = {
    "Volkswagen Golf": "vw_golf",
    "Opel Astra": "opel_astra",
    "Opel Corsa": "opel_corsa",
}

RARE_COLORS = ["yellow", "green", "brown", "gold", "beige", "violet", "orange", "bronze"]
FEATURE_COLUMNS = [
    "T_model_age",
    "T_car_age",
    "T_power_kw",
    "T_mileage",
    "mileage_per_year",
    "fuel_consumption_l_100km",
    "transmission_type_Automatic",
    "transmission_type_Manual",
    "color_black",
    "color_blue",
    "color_grey",
    "color_other",
    "color_red",
    "color_silver",
    "color_white",
]


@dataclass(frozen=True)
class DatasetExport:
    model_name: str
    slug: str
    cleaned_path: Path
    transformed_path: Path


def count_csv_rows(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for _ in handle) - 1


def load_and_clean_raw_data(raw_path: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_path)

    if "Unnamed: 0" in df.columns and "index" not in df.columns:
        df = df.rename(columns={"Unnamed: 0": "index"})
    if "index" in df.columns:
        df = df.set_index("index")

    df = df.dropna().copy()

    invalid_fuel = (
        df["fuel_type"].str.contains(r"\d{2}/\d{4}", regex=True)
        | df["fuel_type"].str.contains(r"\d+\.\d{3}", regex=True)
        | df["fuel_type"].isin(["Automatic", "Manual"])
        | df["fuel_type"].str.contains(r"\d+", regex=True)
    )
    df = df.loc[~invalid_fuel].copy()

    df["power_kw"] = df["power_kw"].astype(int)
    df["power_ps"] = df["power_ps"].astype(int)
    df["price_in_euro"] = df["price_in_euro"].astype(int)
    df["registration_date"] = pd.to_datetime(
        df["registration_date"].map(lambda s: str(s).strip()),
        format="%m/%Y",
    )
    df["fuel_consumption_l_100km"] = (
        df["fuel_consumption_l_100km"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.extract(r"([\d.]+)", expand=False)
        .astype(float)
    )

    return df


def transform_model_dataset(model_df: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame:
    df = model_df.copy()
    df["registration_date"] = pd.to_datetime(df["registration_date"])
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["power_kw"] = pd.to_numeric(df["power_kw"], errors="coerce")
    df["mileage_in_km"] = pd.to_numeric(df["mileage_in_km"], errors="coerce")
    df["fuel_consumption_l_100km"] = pd.to_numeric(df["fuel_consumption_l_100km"], errors="coerce")
    df["price_in_euro"] = pd.to_numeric(df["price_in_euro"], errors="coerce")
    df = df.dropna(
        subset=["year", "power_kw", "mileage_in_km", "fuel_consumption_l_100km", "price_in_euro"]
    ).copy()

    years_since_registration = (
        (today.year - df["registration_date"].dt.year)
        + (today.month - df["registration_date"].dt.month) / 12
    )
    df["years_since_registration"] = years_since_registration

    df = df[df["power_kw"] <= 300].copy()
    df = df[df["mileage_in_km"] <= 0.5 * 1e6].copy()
    df = df[df["fuel_consumption_l_100km"] <= 25].copy()
    df = df[df["fuel_consumption_l_100km"] != 0.0].copy()

    df["color"] = df["color"].replace(RARE_COLORS, "other")
    df = df[df["transmission_type"].isin(["Automatic", "Manual"])].copy()

    dftf = df.copy()
    dftf["T_model_age"] = (2023 - dftf["year"]) ** (2 / 3)
    dftf["T_power_kw"] = np.log(dftf["power_kw"])
    dftf["T_mileage"] = dftf["mileage_in_km"] ** (7 / 10)
    dftf["T_car_age"] = dftf["years_since_registration"] ** (1 / 2)
    dftf["mileage_per_year"] = dftf["T_mileage"] / (dftf["T_car_age"] + 1)

    encoded = pd.get_dummies(
        dftf,
        columns=["color", "transmission_type"],
        drop_first=False,
    )

    for col in FEATURE_COLUMNS:
        if col not in encoded.columns:
            encoded[col] = 0

    return encoded


def run_dataset_quality_tests(df: pd.DataFrame, model_name: str) -> None:
    if df.empty:
        raise ValueError(f"{model_name}: transformed dataset is empty")
    if df["price_in_euro"].le(0).any():
        raise ValueError(f"{model_name}: non-positive prices found")
    if df[FEATURE_COLUMNS].isna().any().any():
        raise ValueError(f"{model_name}: NaN values found in feature set")
    if df.shape[0] < 200:
        raise ValueError(f"{model_name}: not enough rows for train/val/test split ({df.shape[0]})")


def evaluate_split(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_models(df: pd.DataFrame, model_name: str) -> list[dict[str, Any]]:
    X = df[FEATURE_COLUMNS].astype(float)
    y_price = df["price_in_euro"].astype(float)
    y_log = np.log1p(y_price)

    X_train_val, X_test, y_train_val_log, y_test_log = train_test_split(
        X,
        y_log,
        test_size=0.2,
        random_state=42,
    )
    X_train, X_val, y_train_log, y_val_log = train_test_split(
        X_train_val,
        y_train_val_log,
        test_size=0.25,
        random_state=42,
    )
    y_train = np.expm1(y_train_log)
    y_val = np.expm1(y_val_log)
    y_test = np.expm1(y_test_log)

    rows: list[dict[str, Any]] = []

    lr = LinearRegression()
    lr.fit(X_train, y_train_log)
    for split_name, y_true, y_pred in [
        ("train", y_train, np.expm1(lr.predict(X_train))),
        ("validation", y_val, np.expm1(lr.predict(X_val))),
        ("test", y_test, np.expm1(lr.predict(X_test))),
    ]:
        rows.append(
            {
                "car_model": model_name,
                "regressor": "linear_regression",
                "split": split_name,
                **evaluate_split(y_true, y_pred),
            }
        )

    xgb = XGBRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        early_stopping_rounds=50,
    )
    xgb.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)], verbose=False)
    for split_name, y_true, y_pred in [
        ("train", y_train, np.expm1(xgb.predict(X_train))),
        ("validation", y_val, np.expm1(xgb.predict(X_val))),
        ("test", y_test, np.expm1(xgb.predict(X_test))),
    ]:
        rows.append(
            {
                "car_model": model_name,
                "regressor": "xgboost",
                "split": split_name,
                **evaluate_split(y_true, y_pred),
            }
        )

    lgbm = LGBMRegressor(
        objective="regression",
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    )
    lgbm.fit(
        X_train,
        y_train_log,
        eval_set=[(X_val, y_val_log)],
        eval_metric="l2",
        callbacks=[early_stopping(stopping_rounds=50, verbose=False)],
    )
    for split_name, y_true, y_pred in [
        ("train", y_train, np.expm1(lgbm.predict(X_train))),
        ("validation", y_val, np.expm1(lgbm.predict(X_val))),
        ("test", y_test, np.expm1(lgbm.predict(X_test))),
    ]:
        rows.append(
            {
                "car_model": model_name,
                "regressor": "lightgbm",
                "split": split_name,
                **evaluate_split(y_true, y_pred),
            }
        )

    return rows


def export_model_datasets(clean_df: pd.DataFrame) -> tuple[list[DatasetExport], list[dict[str, Any]]]:
    today = pd.Timestamp.today()
    exports: list[DatasetExport] = []
    metrics: list[dict[str, Any]] = []

    for model_name, slug in MODEL_IDENTIFIERS.items():
        model_df = clean_df[clean_df["model"] == model_name].copy()
        cleaned_path = DATA_DIR / f"cleaned_{slug}.csv"
        model_df.to_csv(cleaned_path, index=True)

        transformed_df = transform_model_dataset(model_df, today=today)
        transformed_path = DATA_DIR / f"transformed_cleaned_{slug}.csv"
        transformed_df.to_csv(transformed_path, index=True)

        run_dataset_quality_tests(transformed_df, model_name=model_name)
        metrics.extend(evaluate_models(transformed_df, model_name=model_name))

        exports.append(
            DatasetExport(
                model_name=model_name,
                slug=slug,
                cleaned_path=cleaned_path,
                transformed_path=transformed_path,
            )
        )

    return exports, metrics


def main() -> None:
    clean_df = load_and_clean_raw_data(RAW_DATA_PATH)
    clean_df.to_csv(PARTIAL_CLEAN_PATH, index=True)

    exports, metrics = export_model_datasets(clean_df)

    metrics_df = pd.DataFrame(metrics).sort_values(["car_model", "regressor", "split"])
    metrics_df.to_csv(METRICS_PATH, index=False)

    print(f"Saved partial cleaned data: {PARTIAL_CLEAN_PATH.relative_to(ROOT)} ({len(clean_df):,} rows)")
    for export in exports:
        cleaned_rows = count_csv_rows(export.cleaned_path)
        transformed_rows = count_csv_rows(export.transformed_path)
        print(
            f"{export.model_name}: "
            f"{export.cleaned_path.relative_to(ROOT)} ({cleaned_rows:,} rows), "
            f"{export.transformed_path.relative_to(ROOT)} ({transformed_rows:,} rows)"
        )
    print(f"Saved model metrics: {METRICS_PATH.relative_to(ROOT)} ({len(metrics_df)} rows)")


if __name__ == "__main__":
    main()
