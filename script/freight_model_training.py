#!/usr/bin/env python3
"""Chronological model comparison and final training for freight-rate data.

This script consumes the prepared files produced by freight_data_pipeline.py.
It deliberately excludes target-derived columns such as rate_per_mile and
target_outlier_flag from the feature matrix. The target-outlier flag is used
only to test an optional training-row filter.

Default workflow:
    python freight_model_training.py \
        --data-dir prepared_freight \
        --output-dir freight_model_results

Outputs include:
  model_comparison.csv          rolling-fold results for every candidate
  fold_metrics.csv              row-level fold metrics
  untouched_final_month.json    October holdout result
  final_model.joblib            fitted model bundle
  validation_predictions.csv    predictions for prepared_validation.csv
  december_predictions.csv      predictions for prepared_december.csv
  final_model_metadata.json     human-readable model metadata
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


SEED = 42
TARGET_COLUMN = "posted_rate"
FINAL_MONTH_DEFAULT = "2025-10"


# These columns are either identifiers, raw fields replaced by clean fields,
# or calculated from the target. They must not enter X.
EXCLUDED_FEATURES = {
    "load_id",
    "date",
    "predicted_rate",
    TARGET_COLUMN,
    "rate_per_mile",
    "log_posted_rate",
    "target_low_rpm_flag",
    "target_high_rpm_flag",
    "target_outlier_flag",
    "target_outlier_type",
    "route_id",
    "route_equipment_id",
    # Use clean numeric versions and canonical geometry instead.
    "weight",
    "market_index",
    "quote_signal",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "source_haversine_miles",
    "raw_coordinate_available_flag",
    "coordinate_error_flag",
    "pickup_coordinate_error_miles",
    "delivery_coordinate_error_miles",
}


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    family: str
    target_mode: str
    outlier_policy: str
    params: dict[str, Any]


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y-%m-%d")
    if pd.isna(value):
        return None
    return value


def load_prepared(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = data_dir / "prepared_train.csv"
    validation_path = data_dir / "prepared_validation.csv"
    december_path = data_dir / "prepared_december.csv"
    missing = [str(path) for path in [train_path, validation_path, december_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing prepared files: {missing}")
    train = pd.read_csv(train_path)
    validation = pd.read_csv(validation_path)
    december = pd.read_csv(december_path)
    if TARGET_COLUMN not in train.columns:
        raise ValueError(f"{train_path} does not contain {TARGET_COLUMN!r}.")
    for frame in [train, validation, december]:
        if "date" not in frame.columns:
            raise ValueError("Every prepared file must contain date.")
        frame["_date_dt"] = pd.to_datetime(frame["date"], errors="coerce")
    return train, validation, december


def select_features(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    december: pd.DataFrame,
) -> tuple[list[str], list[str], list[str]]:
    """Select only columns present in every scoring frame."""

    common = set(train.columns) & set(validation.columns) & set(december.columns)
    common -= EXCLUDED_FEATURES
    common.discard("_date_dt")
    if not common:
        raise ValueError("No common model features remain after leakage exclusions.")

    feature_columns = sorted(common)
    categorical: list[str] = []
    numeric: list[str] = []
    for column in feature_columns:
        if (
            pd.api.types.is_object_dtype(train[column])
            or pd.api.types.is_string_dtype(train[column])
            or isinstance(train[column].dtype, pd.CategoricalDtype)
        ):
            categorical.append(column)
        else:
            numeric.append(column)
    return feature_columns, numeric, categorical


def make_one_hot_encoder() -> OneHotEncoder:
    """Create a dense unknown-safe encoder across sklearn versions."""

    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float32,
        )
    except TypeError:  # sklearn < 1.2 uses sparse instead of sparse_output.
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
            dtype=np.float32,
        )


def make_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", make_one_hot_encoder()),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_features))
    if not transformers:
        raise ValueError("No numeric or categorical features available.")
    return ColumnTransformer(transformers=transformers, remainder="drop")


def model_factory(family: str, quick: bool = False) -> Any:
    if family == "ridge":
        return Ridge()
    if family == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=60 if quick else 120,
            random_state=SEED,
            n_jobs=-1,
            bootstrap=False,
        )
    if family == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            early_stopping=False,
            random_state=SEED,
        )
    raise ValueError(f"Unknown model family: {family}")


def candidate_specs(quick: bool = False) -> list[CandidateSpec]:
    """Return a compact but meaningful hyperparameter search space."""

    if quick:
        ridge_params = [{"model__alpha": 1.0}, {"model__alpha": 20.0}]
        extra_params = [
            {"model__max_depth": None, "model__min_samples_leaf": 2, "model__max_features": 1.0},
            {"model__max_depth": 24, "model__min_samples_leaf": 4, "model__max_features": 0.7},
        ]
        hist_params = [
            {
                "model__max_iter": 120,
                "model__learning_rate": 0.06,
                "model__max_leaf_nodes": 31,
                "model__l2_regularization": 1.0,
            },
            {
                "model__max_iter": 180,
                "model__learning_rate": 0.05,
                "model__max_leaf_nodes": 63,
                "model__l2_regularization": 2.0,
            },
        ]
    else:
        ridge_params = [
            {"model__alpha": 0.1},
            {"model__alpha": 1.0},
            {"model__alpha": 20.0},
            {"model__alpha": 100.0},
        ]
        extra_params = [
            {"model__max_depth": None, "model__min_samples_leaf": 1, "model__max_features": 1.0},
            {"model__max_depth": None, "model__min_samples_leaf": 4, "model__max_features": 0.7},
            {"model__max_depth": 24, "model__min_samples_leaf": 2, "model__max_features": 1.0},
            {"model__max_depth": 32, "model__min_samples_leaf": 8, "model__max_features": 0.7},
        ]
        hist_params = [
            {
                "model__max_iter": 150,
                "model__learning_rate": 0.04,
                "model__max_leaf_nodes": 31,
                "model__l2_regularization": 0.0,
            },
            {
                "model__max_iter": 220,
                "model__learning_rate": 0.05,
                "model__max_leaf_nodes": 63,
                "model__l2_regularization": 2.0,
            },
            {
                "model__max_iter": 300,
                "model__learning_rate": 0.03,
                "model__max_leaf_nodes": 63,
                "model__l2_regularization": 5.0,
            },
            {
                "model__max_iter": 180,
                "model__learning_rate": 0.08,
                "model__max_leaf_nodes": 31,
                "model__l2_regularization": 10.0,
            },
        ]

    specs: list[CandidateSpec] = []
    for family, params_list, modes in [
        ("ridge", ridge_params, ("raw", "log")),
        ("extra_trees", extra_params, ("raw", "log")),
        ("hist_gradient_boosting", hist_params, ("raw", "log")),
    ]:
        for target_mode in modes:
            for index, params in enumerate(params_list, start=1):
                specs.append(
                    CandidateSpec(
                        candidate_id=f"{family}__{target_mode}__config_{index}",
                        family=family,
                        target_mode=target_mode,
                        outlier_policy="all_rows",
                        params=params,
                    )
                )

    # The target-outlier flag is never a feature. This candidate tests whether
    # fitting only normal labels improves chronological generalization.
    specs.append(
        CandidateSpec(
            candidate_id="hist_gradient_boosting__raw__normal_labels_only",
            family="hist_gradient_boosting",
            target_mode="raw",
            outlier_policy="normal_labels_only",
            params=hist_params[min(1, len(hist_params) - 1)],
        )
    )
    return specs


def make_pipeline(
    spec: CandidateSpec,
    numeric_features: list[str],
    categorical_features: list[str],
    quick: bool,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(numeric_features, categorical_features)),
            ("model", model_factory(spec.family, quick=quick)),
        ]
    ).set_params(**spec.params)


def month_folds(
    train: pd.DataFrame,
    final_month: str | None = None,
    tuning_month_count: int = 3,
) -> tuple[list[tuple[str, np.ndarray, np.ndarray]], str]:
    months = sorted(train["_date_dt"].dt.to_period("M").astype(str).dropna().unique())
    if len(months) < tuning_month_count + 2:
        raise ValueError("Not enough chronological months for tuning and untouched holdout.")
    untouched = final_month or months[-1]
    if untouched not in months:
        raise ValueError(f"final_month={untouched!r} is not in training data: {months}")
    tuning_candidates = [month for month in months if month < untouched]
    tuning_months = tuning_candidates[-tuning_month_count:]
    folds: list[tuple[str, np.ndarray, np.ndarray]] = []
    month_series = train["_date_dt"].dt.to_period("M").astype(str)
    for validation_month in tuning_months:
        train_mask = (month_series < validation_month).to_numpy()
        validation_mask = (month_series == validation_month).to_numpy()
        if train_mask.sum() == 0 or validation_mask.sum() == 0:
            continue
        folds.append((validation_month, train_mask, validation_mask))
    if not folds:
        raise ValueError("No valid chronological folds were created.")
    return folds, untouched


def make_target(y: pd.Series | np.ndarray, mode: str) -> np.ndarray:
    values = np.asarray(y, dtype=float)
    if mode == "raw":
        return values
    if mode == "log":
        return np.log1p(np.clip(values, a_min=0.0, a_max=None))
    raise ValueError(f"Unsupported target mode: {mode}")


def inverse_prediction(
    fitted: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_pred: pd.DataFrame,
    target_mode: str,
) -> tuple[np.ndarray, float]:
    pred_transformed = fitted.predict(X_pred)
    if target_mode == "raw":
        return np.maximum(np.asarray(pred_transformed, dtype=float), 0.0), 1.0

    # Smearing corrects the systematic underprediction caused by fitting in
    # log-space and evaluating in the original rate scale.
    train_transformed_pred = fitted.predict(X_train)
    residual_factor = float(np.mean(np.exp(y_train - train_transformed_pred)))
    pred = np.expm1(pred_transformed) * residual_factor
    return np.maximum(np.asarray(pred, dtype=float), 0.0), residual_factor


def metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, prediction))
    return {
        "mse": mse,
        "rmse": mse ** 0.5,
        "mae": float(mean_absolute_error(y_true, prediction)),
        "r2": float(r2_score(y_true, prediction)),
    }


def evaluate_candidate(
    spec: CandidateSpec,
    train: pd.DataFrame,
    feature_columns: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    folds: list[tuple[str, np.ndarray, np.ndarray]],
    quick: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fold_rows: list[dict[str, Any]] = []
    y_all = train[TARGET_COLUMN].to_numpy(dtype=float)
    for validation_month, train_mask, validation_mask in folds:
        fit_mask = train_mask.copy()
        if spec.outlier_policy == "normal_labels_only":
            fit_mask &= train["target_outlier_flag"].to_numpy(dtype=float) == 0

        X_fit = train.loc[fit_mask, feature_columns]
        X_validation = train.loc[validation_mask, feature_columns]
        y_fit_original = y_all[fit_mask]
        y_validation = y_all[validation_mask]
        y_fit = make_target(y_fit_original, spec.target_mode)

        fitted = make_pipeline(spec, numeric_features, categorical_features, quick=quick)
        start = time.perf_counter()
        fitted.fit(X_fit, y_fit)
        prediction, smearing = inverse_prediction(
            fitted,
            X_fit,
            y_fit,
            X_validation,
            spec.target_mode,
        )
        elapsed = time.perf_counter() - start
        row = {
            "candidate_id": spec.candidate_id,
            "family": spec.family,
            "target_mode": spec.target_mode,
            "outlier_policy": spec.outlier_policy,
            "validation_month": validation_month,
            "fit_rows": int(fit_mask.sum()),
            "validation_rows": int(validation_mask.sum()),
            "smearing_factor": smearing,
            "fit_seconds": elapsed,
            **metrics(y_validation, prediction),
        }
        fold_rows.append(row)

    summary: dict[str, Any] = {
        "candidate_id": spec.candidate_id,
        "family": spec.family,
        "target_mode": spec.target_mode,
        "outlier_policy": spec.outlier_policy,
        "params": json.dumps(spec.params, sort_keys=True, default=_json_value),
        "fold_count": len(fold_rows),
        "mean_mse": float(np.mean([row["mse"] for row in fold_rows])),
        "median_mse": float(np.median([row["mse"] for row in fold_rows])),
        "mean_rmse": float(np.mean([row["rmse"] for row in fold_rows])),
        "median_rmse": float(np.median([row["rmse"] for row in fold_rows])),
        "mean_mae": float(np.mean([row["mae"] for row in fold_rows])),
        "mean_r2": float(np.mean([row["r2"] for row in fold_rows])),
        "total_fit_seconds": float(sum(row["fit_seconds"] for row in fold_rows)),
    }
    return summary, fold_rows


def find_spec(specs: list[CandidateSpec], candidate_id: str) -> CandidateSpec:
    for spec in specs:
        if spec.candidate_id == candidate_id:
            return spec
    raise KeyError(candidate_id)


def fit_final_bundle(
    spec: CandidateSpec,
    train: pd.DataFrame,
    feature_columns: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    quick: bool,
) -> tuple[dict[str, Any], float, int]:
    fit_mask = np.ones(len(train), dtype=bool)
    if spec.outlier_policy == "normal_labels_only":
        fit_mask &= train["target_outlier_flag"].to_numpy(dtype=float) == 0
    X_fit = train.loc[fit_mask, feature_columns]
    y_original = train.loc[fit_mask, TARGET_COLUMN].to_numpy(dtype=float)
    y_fit = make_target(y_original, spec.target_mode)
    fitted = make_pipeline(spec, numeric_features, categorical_features, quick=quick)
    fitted.fit(X_fit, y_fit)
    _, smearing = inverse_prediction(fitted, X_fit, y_fit, X_fit, spec.target_mode)
    bundle = {
        "pipeline": fitted,
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target_column": TARGET_COLUMN,
        "target_mode": spec.target_mode,
        "outlier_policy": spec.outlier_policy,
        "smearing_factor": smearing,
        "selected_candidate_id": spec.candidate_id,
        "selected_family": spec.family,
        "selected_params": spec.params,
        "training_rows_used": int(fit_mask.sum()),
        "training_rows_total": int(len(train)),
        "created_with_seed": SEED,
    }
    return bundle, smearing, int(fit_mask.sum())


def predict_bundle(bundle: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    X = frame[bundle["feature_columns"]]
    transformed = bundle["pipeline"].predict(X)
    if bundle["target_mode"] == "raw":
        return np.maximum(np.asarray(transformed, dtype=float), 0.0)
    return np.maximum(
        np.expm1(np.asarray(transformed, dtype=float)) * float(bundle["smearing_factor"]),
        0.0,
    )


def save_prediction_from_template(
    template_path: Path,
    prediction: np.ndarray,
    output_path: Path,
    strict_columns: bool = False
) -> None:
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")
    
    output = pd.read_csv(template_path)
    output["predicted_rate"] = np.round(prediction, 2)
    
    # Enforce exactly two columns for the validation submission
    if strict_columns:
        output = output[["load_id", "predicted_rate"]]
        
    output.to_csv(output_path, index=False)


def run_training(
    data_dir: Path,
    output_dir: Path,
    raw_data_dir: Path,
    quick: bool = False,
    final_month: str | None = None,
    tuning_month_count: int = 3,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train, validation, december = load_prepared(data_dir)
    feature_columns, numeric_features, categorical_features = select_features(
        train,
        validation,
        december,
    )
    folds, untouched_month = month_folds(
        train,
        final_month=final_month,
        tuning_month_count=tuning_month_count,
    )
    specs = candidate_specs(quick=quick)

    summaries: list[dict[str, Any]] = []
    all_fold_rows: list[dict[str, Any]] = []
    print(f"Using {len(feature_columns)} features: {len(numeric_features)} numeric, {len(categorical_features)} categorical")
    print(f"Tuning months: {[fold[0] for fold in folds]}; untouched month: {untouched_month}")
    print(f"Candidates: {len(specs)}")
    for index, spec in enumerate(specs, start=1):
        print(f"[{index}/{len(specs)}] {spec.candidate_id}")
        summary, fold_rows = evaluate_candidate(
            spec,
            train,
            feature_columns,
            numeric_features,
            categorical_features,
            folds,
            quick=quick,
        )
        summaries.append(summary)
        all_fold_rows.extend(fold_rows)
        print(f"    mean MSE={summary['mean_mse']:.3f} mean RMSE={summary['mean_rmse']:.3f} mean MAE={summary['mean_mae']:.3f}")

    comparison = pd.DataFrame(summaries).sort_values(
        ["mean_rmse", "median_rmse", "mean_mae"],
        ascending=[True, True, True],
    )
    comparison_path = output_dir / "model_comparison.csv"
    fold_metrics_path = output_dir / "fold_metrics.csv"
    comparison.to_csv(comparison_path, index=False)
    pd.DataFrame(all_fold_rows).to_csv(fold_metrics_path, index=False)

    selected_id = str(comparison.iloc[0]["candidate_id"])
    selected_spec = find_spec(specs, selected_id)

    # Evaluate the chosen configuration on the latest month without using it
    # for tuning. This is the honest internal generalization estimate.
    month_series = train["_date_dt"].dt.to_period("M").astype(str)
    untouched_train_mask = (month_series < untouched_month).to_numpy()
    untouched_validation_mask = (month_series == untouched_month).to_numpy()
    untouched_fit_mask = untouched_train_mask.copy()
    if selected_spec.outlier_policy == "normal_labels_only":
        untouched_fit_mask &= train["target_outlier_flag"].to_numpy(dtype=float) == 0
    X_untouched_fit = train.loc[untouched_fit_mask, feature_columns]
    X_untouched_validation = train.loc[untouched_validation_mask, feature_columns]
    y_untouched_fit = train.loc[untouched_fit_mask, TARGET_COLUMN].to_numpy(dtype=float)
    y_untouched_validation = train.loc[untouched_validation_mask, TARGET_COLUMN].to_numpy(dtype=float)
    transformed_fit_target = make_target(y_untouched_fit, selected_spec.target_mode)
    untouched_model = make_pipeline(
        selected_spec,
        numeric_features,
        categorical_features,
        quick=quick,
    )
    untouched_model.fit(X_untouched_fit, transformed_fit_target)
    untouched_prediction, untouched_smearing = inverse_prediction(
        untouched_model,
        X_untouched_fit,
        transformed_fit_target,
        X_untouched_validation,
        selected_spec.target_mode,
    )
    untouched_result = {
        "selected_candidate_id": selected_id,
        "selected_family": selected_spec.family,
        "target_mode": selected_spec.target_mode,
        "outlier_policy": selected_spec.outlier_policy,
        "params": selected_spec.params,
        "training_period": f"before {untouched_month}",
        "untouched_month": untouched_month,
        "fit_rows": int(untouched_fit_mask.sum()),
        "validation_rows": int(untouched_validation_mask.sum()),
        "smearing_factor": untouched_smearing,
        **metrics(y_untouched_validation, untouched_prediction),
    }
    untouched_path = output_dir / "untouched_final_month.json"
    untouched_path.write_text(json.dumps(untouched_result, indent=2, default=_json_value), encoding="utf-8")

    bundle, final_smearing, rows_used = fit_final_bundle(
        selected_spec,
        train,
        feature_columns,
        numeric_features,
        categorical_features,
        quick=quick,
    )
    model_path = output_dir / "final_model.joblib"
    joblib.dump(bundle, model_path, compress=3)

    validation_prediction = predict_bundle(bundle, validation)
    december_prediction = predict_bundle(bundle, december)

    # Define paths to original raw templates
    template_val_path = raw_data_dir / "validation-predictions-template.csv"
    template_dec_path = raw_data_dir / "december-chart-inputs.csv"

    # Define paths for the final scored outputs
    validation_predictions_path = output_dir / "validation_predictions.csv"
    december_predictions_path = output_dir / "december_chart_inputs.csv"

    # Inject predictions into the original templates
    save_prediction_from_template(
        template_val_path, 
        validation_prediction, 
        validation_predictions_path,
        strict_columns=True  # Strips all columns except load_id and predicted_rate
    )
    save_prediction_from_template(
        template_dec_path, 
        december_prediction, 
        december_predictions_path,
        strict_columns=False # Keeps original columns for charting
    )

    metadata = {
        "selected_candidate_id": selected_id,
        "selected_family": selected_spec.family,
        "target_mode": selected_spec.target_mode,
        "outlier_policy": selected_spec.outlier_policy,
        "selected_params": selected_spec.params,
        "feature_count": len(feature_columns),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "tuning_months": [fold[0] for fold in folds],
        "untouched_final_month": untouched_month,
        "tuning_mean_mse": float(comparison.iloc[0]["mean_mse"]),
        "tuning_median_mse": float(comparison.iloc[0]["median_mse"]),
        "tuning_mean_rmse": float(comparison.iloc[0]["mean_rmse"]),
        "tuning_median_rmse": float(comparison.iloc[0]["median_rmse"]),
        "untouched_final_month_metrics": untouched_result,
        "final_smearing_factor": final_smearing,
        "final_training_rows_used": rows_used,
        "final_training_rows_total": int(len(train)),
        "random_seed": SEED,
        "notes": [
            "Prepared input columns are used directly; raw source files are not modified.",
            "Target-derived columns are excluded from X.",
            "Validation and December predictions use handle_unknown=ignore for unseen cities/categories.",
            "The saved model expects prepared CSV columns, not raw CSV columns.",
        ],
    }
    metadata_path = output_dir / "final_model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=_json_value), encoding="utf-8")

    print(f"Selected: {selected_id}")
    print(f"Tuning mean MSE: {comparison.iloc[0]['mean_mse']:.3f}")
    print(f"Tuning mean RMSE: {comparison.iloc[0]['mean_rmse']:.3f}")
    print(f"Untouched {untouched_month} MSE: {untouched_result['mse']:.3f}")
    print(f"Untouched {untouched_month} RMSE: {untouched_result['rmse']:.3f}")
    print(f"Saved model: {model_path.resolve()}")

    return {
        "comparison": comparison_path,
        "fold_metrics": fold_metrics_path,
        "untouched": untouched_path,
        "model": model_path,
        "metadata": metadata_path,
        "validation_predictions": validation_predictions_path,
        "december_predictions": december_predictions_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("prepared_freight"))
    parser.add_argument("--output-dir", type=Path, default=Path("freight_model_results"))
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing validation-predictions-template.csv and december-chart-inputs.csv.",
    )
    parser.add_argument("--quick", action="store_true", help="Use a smaller hyperparameter grid and fewer trees.")
    parser.add_argument("--final-month", default=None, help="Untouched month, e.g. 2025-10.")
    parser.add_argument(
        "--tuning-month-count",
        type=int,
        default=3,
        help="Number of rolling months used for tuning before the untouched final month.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_training(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        raw_data_dir=args.raw_data_dir,
        quick=args.quick,
        final_month=args.final_month,
        tuning_month_count=args.tuning_month_count,
    )
    for label, path in paths.items():
        print(f"{label}: {path.resolve()}")


if __name__ == "__main__":
    main()
