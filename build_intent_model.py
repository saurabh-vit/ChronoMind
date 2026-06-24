"""
build_intent_model.py — Train and evaluate the offline Intent Classifier.

Usage:
    python build_intent_model.py

Output:
    models/intent_model.pkl   — trained pipeline
    Prints accuracy, precision, recall, F1, and latency benchmark to console.
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_intent_model")

BASE_DIR   = Path(__file__).resolve().parent
DATA_PATH  = BASE_DIR / "data" / "intent_dataset.csv"
MODEL_DIR  = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "intent_model.pkl"


def load_data() -> tuple[list[str], list[str]]:
    logger.info("Loading dataset from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text", "intent"])
    df["text"]   = df["text"].str.strip()
    df["intent"] = df["intent"].str.strip().str.lower()
    logger.info("Loaded %d examples across %d classes: %s",
                len(df), df["intent"].nunique(), sorted(df["intent"].unique()))
    return df["text"].tolist(), df["intent"].tolist()


def build_pipeline() -> Pipeline:
    """TF-IDF (char + word n-grams) + Logistic Regression."""
    tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        max_features=8000,
        sublinear_tf=True,
        strip_accents="unicode",
        min_df=1,
    )
    clf = LogisticRegression(
        C=3.0,
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
    )
    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def evaluate(pipeline: Pipeline, X_test: list, y_test: list) -> dict:
    y_pred = pipeline.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    prec   = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1     = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_test, y_pred, zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "report": report}


def benchmark_latency(pipeline: Pipeline, X_test: list, n_runs: int = 200) -> float:
    times = []
    for text in X_test[:n_runs]:
        t0 = time.perf_counter()
        pipeline.predict([text])
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))


def main():
    logger.info("=" * 60)
    logger.info("ChronoMind v2 — Intent Classifier Training")
    logger.info("=" * 60)

    # 1. Load data
    texts, labels = load_data()

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.20, random_state=42, stratify=labels
    )
    logger.info("Train: %d | Test: %d", len(X_train), len(X_test))

    # 3. Build & train
    pipeline = build_pipeline()
    logger.info("Training TF-IDF + Logistic Regression pipeline...")
    pipeline.fit(X_train, y_train)

    # 4. Evaluate
    metrics = evaluate(pipeline, X_test, y_test)
    logger.info("--- Evaluation Metrics ---")
    logger.info("Accuracy  : %.4f", metrics["accuracy"])
    logger.info("Precision : %.4f", metrics["precision"])
    logger.info("Recall    : %.4f", metrics["recall"])
    logger.info("F1 Score  : %.4f", metrics["f1"])
    print("\n" + metrics["report"])

    # 5. Cross-validation (5-fold)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(build_pipeline(), texts, labels, cv=cv, scoring="f1_weighted")
    logger.info("5-Fold CV F1: %.4f ± %.4f", cv_scores.mean(), cv_scores.std())

    # 6. Latency benchmark
    avg_latency = benchmark_latency(pipeline, X_test)
    logger.info("Avg inference latency: %.2f ms (over %d runs)", avg_latency, min(200, len(X_test)))

    # 7. Save model
    MODEL_DIR.mkdir(exist_ok=True)
    bundle = {
        "pipeline":    pipeline,
        "classes":     list(pipeline.classes_),
        "metrics": {
            "accuracy":        round(metrics["accuracy"], 4),
            "precision":       round(metrics["precision"], 4),
            "recall":          round(metrics["recall"], 4),
            "f1":              round(metrics["f1"], 4),
            "cv_f1_mean":      round(float(cv_scores.mean()), 4),
            "cv_f1_std":       round(float(cv_scores.std()), 4),
            "latency_ms":      round(avg_latency, 2),
            "train_samples":   len(X_train),
            "test_samples":    len(X_test),
        },
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)

    # Also save metrics as JSON for the Streamlit tab
    metrics_path = MODEL_DIR / "intent_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(bundle["metrics"], f, indent=2)

    logger.info("Model saved → %s", MODEL_PATH)
    logger.info("Metrics saved → %s", metrics_path)
    logger.info("=" * 60)
    logger.info("Intent classifier ready.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
