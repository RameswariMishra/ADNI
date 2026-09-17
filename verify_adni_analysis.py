"""Verify the reproducible ADNI analysis and export one Excel workbook.

Place this file in the repository root (beside README.md) and run:

    python verify_adni_analysis.py

Expected datasets:
    testHere/data/alzheimers_balanced_small.csv
    testHere/data/NP_Binary_SHAP_50_50.csv

Output:
    results/ADNI_verification_results.xlsx
    results/SHAP_xgboost_internal_test.png  (when SHAP is installed)

The script distinguishes reproduced results from values merely reported in the
capstone. It cannot recover the unavailable Databricks/MLflow model or verify
participant overlap when no participant identifier is retained.
"""

from __future__ import annotations

import json
import platform
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


RANDOM_STATE = 42
TARGET = "y_mmse_binary"
TEST_SIZE = 0.30
BOOTSTRAP_REPETITIONS = 1_000

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "testHere" / "data"
DEVELOPMENT_PATH = DATA_DIR / "alzheimers_balanced_small.csv"
NP_PATH = DATA_DIR / "NP_Binary_SHAP_50_50.csv"
OUTPUT_DIR = REPO_ROOT / "results"
WORKBOOK_PATH = OUTPUT_DIR / "ADNI_verification_results.xlsx"
SHAP_FIGURE_PATH = OUTPUT_DIR / "SHAP_xgboost_internal_test.png"

SAVED_FEATURES = [
    "WORD1DL",
    "WORD3DL",
    "WORD2DL",
    "MMDRAW",
    "MMW",
    "MMLTR5_W",
    "MMO",
    "MMREPEAT",
    "MMR",
    "MMHAND",
    "VSWEIGHT",
    "GDBORED",
    "VSHEIGHT",
    "VSTMPSRC",
    "VSTEMP",
    "NPIG",
    "VSPULSE",
    "VSBPSYS",
    "GDTOTAL",
]

IDENTIFIER_CANDIDATES = [
    "RID",
    "PTID",
    "SUBJECT_ID",
    "PARTICIPANT_ID",
    "PATIENT_ID",
]

REPORTED_RESULTS = pd.DataFrame(
    [
        {
            "evaluation": "Internal",
            "model": "SVM",
            "metric": "accuracy",
            "reported_value": 0.8961,
            "status": "Reported in capstone; compare with reproduced sheet",
        },
        {
            "evaluation": "Internal",
            "model": "SVM",
            "metric": "f1",
            "reported_value": 0.9003,
            "status": "Reported in capstone; compare with reproduced sheet",
        },
        {
            "evaluation": "Internal",
            "model": "SVM",
            "metric": "roc_auc",
            "reported_value": 0.956,
            "status": "Reported in capstone; compare with reproduced sheet",
        },
        {
            "evaluation": "NP",
            "model": "XGBoost",
            "metric": "accuracy",
            "reported_value": 0.8043,
            "status": "Reported in capstone; compare with reproduced sheet",
        },
        {
            "evaluation": "NP",
            "model": "XGBoost",
            "metric": "f1",
            "reported_value": 0.8235,
            "status": "Reported in capstone; compare with reproduced sheet",
        },
        {
            "evaluation": "NP",
            "model": "XGBoost",
            "metric": "roc_auc",
            "reported_value": 0.933,
            "status": "Reported in capstone; compare with reproduced sheet",
        },
        {
            "evaluation": "Internal",
            "model": "SVM",
            "metric": "auc_gap",
            "reported_value": 0.03,
            "status": "Reported gap; underlying original train AUC unavailable",
        },
        {
            "evaluation": "NP",
            "model": "XGBoost",
            "metric": "auc_gap",
            "reported_value": 0.044,
            "status": "Reported gap; comparison AUC definition requires confirmation",
        },
    ]
)


def require_files() -> None:
    """Stop with a clear error when the required reduced dataset is absent."""
    if not DEVELOPMENT_PATH.exists():
        raise FileNotFoundError(
            f"Required development dataset not found: {DEVELOPMENT_PATH}\n"
            "Keep this script in the repository root and do not rename the data file."
        )


def read_dataset(path: Path) -> pd.DataFrame:
    """Read a CSV and validate its binary target."""
    frame = pd.read_csv(path)
    if TARGET not in frame.columns:
        raise ValueError(f"{path} does not contain target column {TARGET!r}.")
    target = pd.to_numeric(frame[TARGET], errors="raise")
    if target.isna().any() or not set(target.unique()).issubset({0, 1}):
        raise ValueError(f"{path}: {TARGET} must contain only nonmissing 0/1 values.")
    if target.nunique() != 2:
        raise ValueError(f"{path}: {TARGET} must contain both classes.")
    frame[TARGET] = target.astype(int)
    return frame


def dataset_summary(name: str, path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    """Return high-level dataset checks."""
    identifier = find_identifier(frame)
    return {
        "dataset": name,
        "file": str(path.relative_to(REPO_ROOT)),
        "rows": len(frame),
        "columns": frame.shape[1],
        "predictors": frame.shape[1] - 1,
        "target_0": int((frame[TARGET] == 0).sum()),
        "target_1": int((frame[TARGET] == 1).sum()),
        "exact_duplicate_rows": int(frame.duplicated().sum()),
        "missing_cells": int(frame.isna().sum().sum()),
        "participant_identifier": identifier or "Not retained",
        "unique_participants": (
            int(frame[identifier].nunique(dropna=True)) if identifier else np.nan
        ),
    }


def find_identifier(frame: pd.DataFrame) -> str | None:
    """Return the first recognized participant identifier."""
    upper_to_original = {column.upper(): column for column in frame.columns}
    for candidate in IDENTIFIER_CANDIDATES:
        if candidate in upper_to_original:
            return upper_to_original[candidate]
    return None


def column_audit(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Describe types, missingness, and cardinality for each column."""
    rows = []
    for column in frame.columns:
        rows.append(
            {
                "dataset": name,
                "column": column,
                "dtype": str(frame[column].dtype),
                "missing_count": int(frame[column].isna().sum()),
                "missing_percent": float(frame[column].isna().mean()),
                "unique_values": int(frame[column].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def feature_audit(development: pd.DataFrame, np_frame: pd.DataFrame | None) -> pd.DataFrame:
    """Check the saved 19 names against both reduced datasets."""
    rows = []
    for rank, feature in enumerate(SAVED_FEATURES, start=1):
        rows.append(
            {
                "saved_rank": rank,
                "feature": feature,
                "in_development_csv": feature in development.columns,
                "in_np_csv": feature in np_frame.columns if np_frame is not None else np.nan,
                "used_by_reproduced_models": feature in development.columns,
            }
        )
    return pd.DataFrame(rows)


def participant_checks(
    development: pd.DataFrame, np_frame: pd.DataFrame | None
) -> pd.DataFrame:
    """Report participant counts and cross-cohort overlap when possible."""
    development_id = find_identifier(development)
    np_id = find_identifier(np_frame) if np_frame is not None else None
    rows = []

    if development_id:
        rows.append(
            {
                "check": "Development repeated observations",
                "value": len(development) - development[development_id].nunique(dropna=True),
                "status": "Calculated",
                "details": f"Identifier: {development_id}",
            }
        )
    else:
        rows.append(
            {
                "check": "Development repeated observations",
                "value": np.nan,
                "status": "Not verifiable",
                "details": "No recognized participant identifier retained",
            }
        )

    if np_frame is not None and np_id:
        rows.append(
            {
                "check": "NP repeated observations",
                "value": len(np_frame) - np_frame[np_id].nunique(dropna=True),
                "status": "Calculated",
                "details": f"Identifier: {np_id}",
            }
        )
    elif np_frame is not None:
        rows.append(
            {
                "check": "NP repeated observations",
                "value": np.nan,
                "status": "Not verifiable",
                "details": "No recognized participant identifier retained",
            }
        )

    if development_id and np_frame is not None and np_id:
        overlap = set(development[development_id].dropna()).intersection(
            set(np_frame[np_id].dropna())
        )
        rows.append(
            {
                "check": "Living/NP participant overlap",
                "value": len(overlap),
                "status": "Calculated",
                "details": ", ".join(map(str, sorted(overlap))) if overlap else "None",
            }
        )
    else:
        rows.append(
            {
                "check": "Living/NP participant overlap",
                "value": np.nan,
                "status": "Not verifiable",
                "details": "Participant identifier is absent from one or both files",
            }
        )

    return pd.DataFrame(rows)


def build_searches() -> dict[str, GridSearchCV]:
    """Create leakage-controlled model-specific grid searches."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    model_specs = {
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_split": [2, 5],
                "model__min_samples_leaf": [1, 2],
                "model__max_features": ["sqrt", "log2"],
            },
        ),
        "XGBoost": (
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=1,
                tree_method="hist",
            ),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [3, 5],
                "model__learning_rate": [0.05, 0.1],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0],
            },
        ),
        "SVM": (
            SVC(probability=True, random_state=RANDOM_STATE),
            {
                "model__kernel": ["linear", "rbf"],
                "model__C": [0.1, 1, 10],
                "model__gamma": ["scale", 0.01, 0.001],
            },
        ),
    }

    searches = {}
    for name, (estimator, parameter_grid) in model_specs.items():
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        )
        searches[name] = GridSearchCV(
            pipeline,
            parameter_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
            return_train_score=True,
        )
    return searches


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Calculate binary metrics with class 1 as positive."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def bootstrap_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    evaluation: str,
) -> pd.DataFrame:
    """Calculate percentile bootstrap 95% intervals for principal metrics."""
    rng = np.random.default_rng(RANDOM_STATE)
    values: dict[str, list[float]] = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "specificity": [],
        "f1": [],
        "roc_auc": [],
        "pr_auc": [],
    }
    successful = 0
    for _ in range(BOOTSTRAP_REPETITIONS):
        indices = rng.integers(0, len(y_true), len(y_true))
        sampled_true = y_true[indices]
        if np.unique(sampled_true).size < 2:
            continue
        sampled = binary_metrics(
            sampled_true,
            y_pred[indices],
            y_prob[indices],
        )
        for metric in values:
            values[metric].append(sampled[metric])
        successful += 1

    rows = []
    observed = binary_metrics(y_true, y_pred, y_prob)
    for metric, samples in values.items():
        rows.append(
            {
                "evaluation": evaluation,
                "model": model_name,
                "metric": metric,
                "estimate": observed[metric],
                "ci_95_lower": float(np.percentile(samples, 2.5)),
                "ci_95_upper": float(np.percentile(samples, 97.5)),
                "successful_bootstrap_samples": successful,
                "resampling_unit": "observation",
            }
        )
    return pd.DataFrame(rows)


def run_models(
    development: pd.DataFrame, np_frame: pd.DataFrame | None
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Pipeline],
    pd.DataFrame,
    pd.DataFrame,
]:
    """Tune on the internal training partition and evaluate locked models."""
    feature_columns = [column for column in development.columns if column != TARGET]
    X = development[feature_columns].apply(pd.to_numeric, errors="coerce")
    y = development[TARGET].to_numpy(dtype=int)

    indices = np.arange(len(development))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    result_rows = []
    prediction_rows = []
    parameter_rows = []
    interval_frames = []
    fitted_models: dict[str, Pipeline] = {}

    np_results = []
    np_predictions = []
    X_np = None
    y_np = None
    if np_frame is not None:
        missing_np_features = sorted(set(feature_columns) - set(np_frame.columns))
        if missing_np_features:
            raise ValueError(f"NP CSV is missing predictors: {missing_np_features}")
        X_np = np_frame[feature_columns].apply(pd.to_numeric, errors="coerce")
        y_np = np_frame[TARGET].to_numpy(dtype=int)

    for model_name, search in build_searches().items():
        print(f"Tuning {model_name} ...", flush=True)
        search.fit(X_train, y_train)
        fitted = clone(search.best_estimator_).fit(X_train, y_train)
        fitted_models[model_name] = fitted

        train_probability = fitted.predict_proba(X_train)[:, 1]
        test_probability = fitted.predict_proba(X_test)[:, 1]
        test_prediction = fitted.predict(X_test)
        train_auc = roc_auc_score(y_train, train_probability)
        test_metrics = binary_metrics(y_test, test_prediction, test_probability)

        result_rows.append(
            {
                "evaluation": "Internal held-out 30%",
                "model": model_name,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "cv_best_roc_auc": search.best_score_,
                "train_roc_auc": train_auc,
                **test_metrics,
                "auc_gap_train_minus_test": train_auc - test_metrics["roc_auc"],
            }
        )
        parameter_rows.append(
            {
                "model": model_name,
                "best_parameters": json.dumps(search.best_params_, default=str),
            }
        )
        interval_frames.append(
            bootstrap_intervals(
                y_test,
                test_prediction,
                test_probability,
                model_name,
                "Internal held-out 30%",
            )
        )
        for position, original_index in enumerate(test_indices):
            prediction_rows.append(
                {
                    "model": model_name,
                    "source_row_index": int(original_index),
                    "observed_target": int(y_test[position]),
                    "predicted_class": int(test_prediction[position]),
                    "predicted_probability": float(test_probability[position]),
                }
            )

        if X_np is not None and y_np is not None:
            external_probability = fitted.predict_proba(X_np)[:, 1]
            external_prediction = fitted.predict(X_np)
            external_metrics = binary_metrics(
                y_np, external_prediction, external_probability
            )
            np_results.append(
                {
                    "evaluation": "NP cohort; no refitting",
                    "model": model_name,
                    "n": len(X_np),
                    **external_metrics,
                    "internal_train_auc": train_auc,
                    "gap_train_minus_np_auc": train_auc
                    - external_metrics["roc_auc"],
                    "gap_internal_test_minus_np_auc": test_metrics["roc_auc"]
                    - external_metrics["roc_auc"],
                }
            )
            interval_frames.append(
                bootstrap_intervals(
                    y_np,
                    external_prediction,
                    external_probability,
                    model_name,
                    "NP cohort; no refitting",
                )
            )
            for row_number in range(len(X_np)):
                np_predictions.append(
                    {
                        "model": model_name,
                        "source_row_index": row_number,
                        "observed_target": int(y_np[row_number]),
                        "predicted_class": int(external_prediction[row_number]),
                        "predicted_probability": float(
                            external_probability[row_number]
                        ),
                    }
                )

    return (
        pd.DataFrame(result_rows),
        pd.DataFrame(np_results),
        pd.concat(interval_frames, ignore_index=True),
        pd.DataFrame(parameter_rows),
        fitted_models,
        pd.DataFrame(prediction_rows),
        pd.DataFrame(np_predictions),
    )


def scan_repository_code() -> pd.DataFrame:
    """Find evidence relevant to splitting, SHAP, targets, and preprocessing."""
    patterns = {
        "Target definition/reference": re.compile(r"y_mmse_binary", re.I),
        "Row-level split": re.compile(r"train_test_split", re.I),
        "Grouped split": re.compile(r"GroupShuffleSplit|StratifiedGroupKFold|GroupKFold", re.I),
        "SHAP calculation": re.compile(r"TreeExplainer|shap_values|summary_plot", re.I),
        "Test data reference": re.compile(r"\bX_test\b", re.I),
        "Feature selection/reference": re.compile(r"selected_features|top_features|feature_columns", re.I),
        "Pipeline": re.compile(r"\bPipeline\s*\(", re.I),
        "Imputation": re.compile(r"SimpleImputer|fillna", re.I),
        "Scaling": re.compile(r"StandardScaler|MinMaxScaler", re.I),
    }
    excluded_parts = {".git", ".venv", "venv", "site-packages", "node_modules"}
    rows = []
    for extension in ("*.py", "*.ipynb"):
        for path in REPO_ROOT.rglob(extension):
            if any(part in excluded_parts for part in path.parts):
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                continue
            for line_number, line in enumerate(lines, start=1):
                compact = " ".join(line.strip().split())
                if not compact:
                    continue
                for finding, pattern in patterns.items():
                    if pattern.search(compact):
                        rows.append(
                            {
                                "finding": finding,
                                "file": str(path.relative_to(REPO_ROOT)),
                                "line": line_number,
                                "extract": compact[:500],
                            }
                        )
    return pd.DataFrame(rows)


def create_shap_outputs(
    development: pd.DataFrame, fitted_models: dict[str, Pipeline]
) -> tuple[pd.DataFrame, str]:
    """Create a reproduced XGBoost SHAP ranking and figure when available."""
    try:
        import matplotlib.pyplot as plt
        import shap
    except ImportError:
        return pd.DataFrame(), "SHAP not installed; run: pip install shap matplotlib"

    feature_columns = [column for column in development.columns if column != TARGET]
    X = development[feature_columns].apply(pd.to_numeric, errors="coerce")
    y = development[TARGET].to_numpy(dtype=int)
    _, test_indices = train_test_split(
        np.arange(len(development)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_test = X.iloc[test_indices]

    pipeline = fitted_models["XGBoost"]
    transformed = pipeline[:-1].transform(X_test)
    xgboost_model = pipeline.named_steps["model"]
    explainer = shap.TreeExplainer(xgboost_model)
    raw_values = explainer.shap_values(transformed)
    if isinstance(raw_values, list):
        values = np.asarray(raw_values[-1])
    else:
        values = np.asarray(raw_values)
    if values.ndim == 3:
        values = values[:, :, -1]
    if values.shape != transformed.shape:
        raise ValueError(
            f"Unexpected SHAP shape {values.shape}; expected {transformed.shape}."
        )

    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "mean_absolute_shap": np.abs(values).mean(axis=0),
            "mean_signed_shap": values.mean(axis=0),
        }
    ).sort_values("mean_absolute_shap", ascending=False)
    importance.insert(0, "rank", range(1, len(importance) + 1))

    plot_data = importance.sort_values("mean_absolute_shap")
    figure, axis = plt.subplots(figsize=(8.5, max(5.5, 0.36 * len(plot_data))))
    axis.barh(
        plot_data["feature"],
        plot_data["mean_absolute_shap"],
        color="#276FBF",
    )
    axis.set_title("Reproduced XGBoost global SHAP importance")
    axis.set_xlabel("Mean absolute SHAP value")
    axis.set_ylabel("Predictor")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="x", color="#D9D9D9", linewidth=0.7)
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(SHAP_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return importance.reset_index(drop=True), f"Created {SHAP_FIGURE_PATH.name}"


def build_summary(
    development: pd.DataFrame,
    np_frame: pd.DataFrame | None,
    shap_status: str,
) -> pd.DataFrame:
    """Create a short, manuscript-focused audit summary."""
    actual_features = [
        feature for feature in SAVED_FEATURES if feature in development.columns
    ]
    participant_id = find_identifier(development)
    rows = [
        {
            "check": "Required development dataset",
            "finding": str(DEVELOPMENT_PATH.relative_to(REPO_ROOT)),
            "interpretation": "Used for the reproduced 70/30 internal evaluation",
        },
        {
            "check": "NP dataset",
            "finding": (
                str(NP_PATH.relative_to(REPO_ROOT))
                if np_frame is not None
                else "Not found"
            ),
            "interpretation": "Evaluated without model refitting when present",
        },
        {
            "check": "Saved feature names",
            "finding": len(SAVED_FEATURES),
            "interpretation": "The accessible saved list contains 19, not 20",
        },
        {
            "check": "Saved features present in development CSV",
            "finding": len(actual_features),
            "interpretation": "See Feature_Check for absent names",
        },
        {
            "check": "Participant-level verification",
            "finding": "Available" if participant_id else "Unavailable",
            "interpretation": (
                f"Identifier retained: {participant_id}"
                if participant_id
                else "Use 'observations' because no recognized participant ID is retained"
            ),
        },
        {
            "check": "Model rerun",
            "finding": "Training-only five-fold GridSearchCV",
            "interpretation": "Final internal test set is not used for tuning",
        },
        {
            "check": "Confidence intervals",
            "finding": f"{BOOTSTRAP_REPETITIONS} bootstrap repetitions",
            "interpretation": "Observation-level intervals; participant bootstrap requires IDs",
        },
        {
            "check": "SHAP output",
            "finding": shap_status,
            "interpretation": "Reproduced model explanation, not original MLflow model output",
        },
    ]
    return pd.DataFrame(rows)


def write_workbook(sheets: dict[str, pd.DataFrame]) -> None:
    """Write and lightly format the verification workbook."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(WORKBOOK_PATH, engine="openpyxl") as writer:
        for sheet_name, frame in sheets.items():
            safe_name = sheet_name[:31]
            output_frame = frame if not frame.empty else pd.DataFrame({"status": ["No data"]})
            output_frame.to_excel(writer, sheet_name=safe_name, index=False)

        workbook = writer.book
        for worksheet in workbook.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            worksheet.sheet_view.showGridLines = False
            for cell in worksheet[1]:
                cell.font = cell.font.copy(bold=True, color="FFFFFF")
                cell.fill = cell.fill.copy(fill_type="solid", fgColor="1F4E78")
            for column_cells in worksheet.columns:
                width = min(
                    60,
                    max(
                        10,
                        max(len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells)
                        + 2,
                    ),
                )
                worksheet.column_dimensions[column_cells[0].column_letter].width = width
            for row in worksheet.iter_rows(min_row=2):
                for cell in row:
                    if isinstance(cell.value, float):
                        cell.number_format = "0.0000"


def main() -> None:
    """Run all available checks and create the Excel workbook."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    require_files()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading development data:", DEVELOPMENT_PATH)
    development = read_dataset(DEVELOPMENT_PATH)
    np_frame = read_dataset(NP_PATH) if NP_PATH.exists() else None
    if np_frame is None:
        print("NP dataset not found; NP evaluation will be marked unavailable.")
    else:
        print("Loading NP data:", NP_PATH)

    dataset_rows = [dataset_summary("Development", DEVELOPMENT_PATH, development)]
    if np_frame is not None:
        dataset_rows.append(dataset_summary("NP", NP_PATH, np_frame))

    print("Running model verification. This can take several minutes ...")
    (
        internal_results,
        np_results,
        confidence_intervals,
        best_parameters,
        fitted_models,
        internal_predictions,
        np_predictions,
    ) = run_models(development, np_frame)

    try:
        shap_importance, shap_status = create_shap_outputs(development, fitted_models)
    except Exception as error:  # Preserve the rest of the verification if SHAP fails.
        shap_importance = pd.DataFrame()
        shap_status = f"SHAP failed: {type(error).__name__}: {error}"

    environment = pd.DataFrame(
        [
            {"item": "Python", "value": platform.python_version()},
            {"item": "Platform", "value": platform.platform()},
            {"item": "Repository root", "value": str(REPO_ROOT)},
            {"item": "Random state", "value": RANDOM_STATE},
            {"item": "Internal test size", "value": TEST_SIZE},
            {"item": "Bootstrap repetitions", "value": BOOTSTRAP_REPETITIONS},
            {"item": "Target", "value": TARGET},
        ]
    )

    column_frames = [column_audit("Development", development)]
    if np_frame is not None:
        column_frames.append(column_audit("NP", np_frame))

    sheets = {
        "Summary": build_summary(development, np_frame, shap_status),
        "Dataset_Summary": pd.DataFrame(dataset_rows),
        "Feature_Check": feature_audit(development, np_frame),
        "Participant_Check": participant_checks(development, np_frame),
        "Internal_Results": internal_results,
        "NP_Results": np_results,
        "Confidence_Intervals": confidence_intervals,
        "Reported_Results": REPORTED_RESULTS,
        "Best_Parameters": best_parameters,
        "SHAP_Importance": shap_importance,
        "Column_Audit": pd.concat(column_frames, ignore_index=True),
        "Code_Search": scan_repository_code(),
        "Internal_Predictions": internal_predictions,
        "NP_Predictions": np_predictions,
        "Environment": environment,
    }
    write_workbook(sheets)

    print("\nCompleted.")
    print("Excel workbook:", WORKBOOK_PATH)
    print("SHAP status:", shap_status)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
