"""Classify large claims at empirical severity thresholds."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from severity.ml_models import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES
from severity.traditional_models import RANDOM_STATE, split_claims_by_policy

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH: Final = PROJECT_ROOT / "data" / "processed" / "claim_severity.parquet"
DEFAULT_OUTPUT_DIR: Final = PROJECT_ROOT / "reports" / "large_loss_classification"
QUANTILES: Final = (0.90, 0.95, 0.99)
PRIMARY_QUANTILE: Final = 0.95


def prepare_native_features(claims: pd.DataFrame) -> pd.DataFrame:
    features = claims[FEATURES].copy()
    for column in CATEGORICAL_FEATURES:
        features[column] = features[column].astype("category")
    return features


def make_logistic_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(max_iter=1_000, C=1.0, random_state=RANDOM_STATE),
            ),
        ]
    )


def make_gradient_boosting() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=20,
        min_samples_leaf=50,
        l2_regularization=5.0,
        categorical_features="from_dtype",
        random_state=RANDOM_STATE,
    )


def select_f1_threshold(actual: pd.Series, probability: np.ndarray) -> float:
    """Select an operating threshold on development data only."""
    precision, recall, thresholds = precision_recall_curve(actual, probability)
    if thresholds.size == 0:
        return 0.5
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return float(thresholds[int(np.argmax(f1))])


def evaluate_classifier(
    actual: pd.Series, probability: np.ndarray, operating_threshold: float
) -> dict[str, float | int]:
    observed = actual.to_numpy(dtype=int)
    predicted = probability >= operating_threshold
    return {
        "events": int(observed.sum()),
        "event_rate": float(observed.mean()),
        "pr_auc": float(average_precision_score(observed, probability)),
        "roc_auc": float(roc_auc_score(observed, probability)),
        "recall": float(recall_score(observed, predicted, zero_division=0)),
        "precision": float(precision_score(observed, predicted, zero_division=0)),
        "brier_score": float(brier_score_loss(observed, probability)),
        "operating_threshold": operating_threshold,
        "predicted_positive_rate": float(predicted.mean()),
    }


def probability_calibration(
    actual: pd.Series, probability: np.ndarray, bins: int = 10
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {"Actual": actual.to_numpy(dtype=int), "Probability": probability}
    )
    frame["ProbabilityGroup"] = pd.qcut(
        frame["Probability"], q=bins, labels=False, duplicates="drop"
    )
    return (
        frame.groupby("ProbabilityGroup", observed=True)
        .agg(
            Claims=("Actual", "size"),
            LargeClaims=("Actual", "sum"),
            ObservedProbability=("Actual", "mean"),
            PredictedProbability=("Probability", "mean"),
        )
        .reset_index()
    )


def fit_classifiers(
    claims: pd.DataFrame, test_size: float = 0.2
) -> tuple[dict, dict[str, pd.DataFrame], dict[str, object]]:
    """Fit both classifiers at each development-sample quantile threshold."""
    train, test = split_claims_by_policy(claims, test_size)
    tables: dict[str, pd.DataFrame] = {}
    fitted: dict[str, object] = {}
    metrics: dict = {
        "split": {
            "test_size": test_size,
            "training_claims": len(train),
            "test_claims": len(test),
            "training_policies": int(train["IDpol"].nunique()),
            "test_policies": int(test["IDpol"].nunique()),
        },
        "thresholds": {},
    }
    for quantile in QUANTILES:
        label = f"q{int(quantile * 100)}"
        amount_threshold = float(train["ClaimAmount"].quantile(quantile))
        train_target = train["ClaimAmount"].gt(amount_threshold).astype(int)
        test_target = test["ClaimAmount"].gt(amount_threshold).astype(int)

        logistic = make_logistic_pipeline()
        boosting = make_gradient_boosting()
        logistic.fit(train[FEATURES], train_target)
        boosting.fit(prepare_native_features(train), train_target)
        train_probabilities = {
            "logistic": logistic.predict_proba(train[FEATURES])[:, 1],
            "gradient_boosting": boosting.predict_proba(
                prepare_native_features(train)
            )[:, 1],
        }
        test_probabilities = {
            "logistic": logistic.predict_proba(test[FEATURES])[:, 1],
            "gradient_boosting": boosting.predict_proba(
                prepare_native_features(test)
            )[:, 1],
        }
        prediction_table = test[
            ["SourceRow", "IDpol", "ClaimIndex", "ClaimAmount"]
        ].copy()
        prediction_table["LargeLoss"] = test_target
        threshold_metrics: dict = {
            "claim_amount_threshold": amount_threshold,
            "training_events": int(train_target.sum()),
            "test_events": int(test_target.sum()),
            "models": {},
        }
        for model_name in ("logistic", "gradient_boosting"):
            operating_threshold = select_f1_threshold(
                train_target, train_probabilities[model_name]
            )
            probability = test_probabilities[model_name]
            threshold_metrics["models"][model_name] = evaluate_classifier(
                test_target, probability, operating_threshold
            )
            prediction_table[f"{model_name.title().replace('_', '')}Probability"] = (
                probability
            )
            tables[f"{label}_{model_name}_calibration"] = probability_calibration(
                test_target, probability
            )
        tables[f"{label}_predictions"] = prediction_table
        fitted[f"{label}_logistic"] = logistic
        fitted[f"{label}_gradient_boosting"] = boosting
        metrics["thresholds"][label] = threshold_metrics
    return metrics, tables, fitted


def _render_report(metrics: dict) -> str:
    lines = [
        "# Large-loss classification",
        "",
        f"Training claims: {metrics['split']['training_claims']:,}",
        f"Held-out claims: {metrics['split']['test_claims']:,}",
        "",
        "| Threshold | Model | Events | PR-AUC | ROC-AUC | Recall | Precision | Brier |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, threshold in metrics["thresholds"].items():
        for model_name, item in threshold["models"].items():
            lines.append(
                f"| {label.upper()} (€{threshold['claim_amount_threshold']:,.2f}) | "
                f"{model_name.replace('_', ' ').title()} | {item['events']:,} | "
                f"{item['pr_auc']:.4f} | {item['roc_auc']:.4f} | "
                f"{item['recall']:.4f} | {item['precision']:.4f} | "
                f"{item['brier_score']:.5f} |"
            )
    lines.extend(
        [
            "",
            "Severity thresholds and F1 operating cutoffs were estimated on development data only.",
            "PR-AUC is interpreted relative to each threshold's held-out event prevalence; accuracy is intentionally omitted.",
            "",
        ]
    )
    return "\n".join(lines)


def run_models(data_path: Path, output_dir: Path, test_size: float = 0.2) -> dict:
    claims = pd.read_parquet(data_path)
    metrics, tables, fitted = fit_classifiers(claims, test_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        suffix = ".parquet" if name.endswith("predictions") else ".csv"
        destination = output_dir / f"{name}{suffix}"
        if suffix == ".parquet":
            table.to_parquet(destination, index=False)
        else:
            table.to_csv(destination, index=False)
    for name, model in fitted.items():
        joblib.dump(model, output_dir / f"{name}.joblib")
    result = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), **metrics}
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    report = _render_report(metrics)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_models(args.data_path.resolve(), args.output_dir.resolve(), args.test_size)
