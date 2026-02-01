"""Training entry point for UFC outcome models."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier

from src.config import ARTIFACTS_DIR, ModelConfig
from src.models.calibrate import compute_calibration
from src.models.evaluate import compute_metrics

logger = logging.getLogger(__name__)


def load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def train_model(data_path: Path, model_type: str, output_dir: Path = ARTIFACTS_DIR) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset(data_path)

    df = df.sort_values("fight_date")
    feature_cols = [c for c in df.columns if c.endswith("_diff")]
    X = df[feature_cols]
    y = df["label"].astype(int)

    config = ModelConfig()
    test_size = min(config.test_size_recent, len(df) // 5)
    train_df = df.iloc[:-test_size]
    test_df = df.iloc[-test_size:]

    X_train = train_df[feature_cols]
    y_train = train_df["label"].astype(int)
    X_test = test_df[feature_cols]
    y_test = test_df["label"].astype(int)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, feature_cols)], remainder="drop"
    )

    if model_type == "logistic":
        clf = LogisticRegression(max_iter=200, C=config.logistic_c)
    elif model_type == "hgb":
        clf = HistGradientBoostingClassifier(random_state=config.random_state)
    else:
        raise ValueError("model_type must be 'logistic' or 'hgb'")

    model = Pipeline(steps=[("pre", preprocessor), ("clf", clf)])
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test.to_numpy(), y_proba)
    calib = compute_calibration(y_test.to_numpy(), y_proba)

    model_path = output_dir / f"model_{model_type}.joblib"
    joblib.dump(model, model_path)

    importance = _model_importance(model, model_type, X_test, y_test, feature_cols)
    meta = {
        "model_type": model_type,
        "feature_cols": feature_cols,
        "metrics": metrics.__dict__,
        "importance": importance,
        "calibration": {
            "prob_true": calib.prob_true.tolist(),
            "prob_pred": calib.prob_pred.tolist(),
        },
    }
    meta_path = output_dir / f"model_{model_type}.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {"model": model_path, "meta": meta_path}


def _model_importance(model: Pipeline, model_type: str, X: pd.DataFrame, y: pd.Series, cols: list[str]) -> list[dict]:
    if model_type == "logistic":
        coefs = model.named_steps["clf"].coef_[0]
        pairs = sorted(zip(cols, coefs), key=lambda x: abs(x[1]), reverse=True)
        return [{"feature": f, "coef": float(c)} for f, c in pairs[:20]]
    result = permutation_importance(model, X, y, n_repeats=10, random_state=0)
    pairs = sorted(zip(cols, result.importances_mean), key=lambda x: abs(x[1]), reverse=True)
    return [{"feature": f, "importance": float(i)} for f, i in pairs[:20]]
