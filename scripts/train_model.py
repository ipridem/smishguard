"""Train the smishing classifier: TF-IDF (word + char n-gram) + engineered
features -> Logistic Regression and Linear SVM baselines. Saves the served
pipeline (Logistic Regression, for predict_proba) plus metrics.json with
both models' scores on a clean test set and an adversarial (obfuscated)
test set.

Usage:
    .venv/Scripts/python scripts/train_model.py --seed 42
Requires labeled data in sms_messages (run scripts/generate_sms.py first).
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from app.config import Config
from app.extensions import db_session, init_db
from app.models.sms import SmsMessage
from app.smishing.features import SmishingFeatureExtractor, normalize_for_tfidf

# Robustness check against leetspeak/obfuscated variants (research
# contribution #2 in the brief): character substitution + light typo
# injection applied to the held-out test set only, never seen in training.
LEET_MAP = str.maketrans({"a": "4", "e": "3", "i": "1", "o": "0", "s": "5"})


def obfuscate(text: str, rng: random.Random) -> str:
    chars = list(text.translate(LEET_MAP))
    i = 0
    while i < len(chars):
        if chars[i].isalpha() and rng.random() < 0.05:
            chars.insert(i, chars[i])  # duplicate a letter, typo-like
            i += 1
        i += 1
    return "".join(chars)


def build_pipeline(estimator, min_df: int = 2) -> Pipeline:
    # note: a custom preprocessor disables TfidfVectorizer's built-in
    # lowercasing, so normalize_for_tfidf does it explicitly at the end
    features = FeatureUnion([
        ("word_tfidf", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=min_df, preprocessor=normalize_for_tfidf,
        )),
        ("char_tfidf", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=min_df, preprocessor=normalize_for_tfidf,
        )),
        ("engineered", SmishingFeatureExtractor()),
    ])
    return Pipeline([("features", features), ("clf", estimator)])


def load_corpus() -> tuple[list[str], list[str]]:
    rows = db_session.query(SmsMessage).filter(SmsMessage.label.isnot(None)).all()
    return [r.text for r in rows], [r.label.value for r in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--model-out", default=Config.SMISHING_MODEL_PATH)
    parser.add_argument("--metrics-out", default="ml/metrics.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    init_db(Config.DATABASE_URL)

    texts, labels = load_corpus()
    if len(texts) < 50:
        raise SystemExit("Not enough labeled data — run scripts/generate_sms.py first.")

    holdout_size = args.test_size + args.val_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=holdout_size, random_state=args.seed, stratify=labels
    )
    val_fraction = args.val_size / holdout_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - val_fraction, random_state=args.seed, stratify=y_temp
    )
    adversarial_X = [obfuscate(t, rng) for t in X_test]

    results = {}
    served_pipeline = None
    for name, estimator in [
        ("logistic_regression", LogisticRegression(max_iter=1000)),
        ("linear_svm", LinearSVC()),
    ]:
        pipeline = build_pipeline(estimator)
        pipeline.fit(X_train, y_train)

        val_report = classification_report(y_val, pipeline.predict(X_val), output_dict=True, zero_division=0)
        test_report = classification_report(y_test, pipeline.predict(X_test), output_dict=True, zero_division=0)
        adv_report = classification_report(y_test, pipeline.predict(adversarial_X), output_dict=True, zero_division=0)

        results[name] = {
            "val_macro_f1": val_report["macro avg"]["f1-score"],
            "test_macro_f1": test_report["macro avg"]["f1-score"],
            "adversarial_macro_f1": adv_report["macro avg"]["f1-score"],
            "test_report": test_report,
            "adversarial_report": adv_report,
        }
        if name == "logistic_regression":
            served_pipeline = pipeline

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(served_pipeline, args.model_out)

    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_out).write_text(json.dumps({
        "seed": args.seed,
        "train_size": len(X_train), "val_size": len(X_val), "test_size": len(X_test),
        "served_model": "logistic_regression",
        "models": results,
    }, indent=2))

    for name, r in results.items():
        print(f"{name}: test_macro_f1={r['test_macro_f1']:.3f} adversarial_macro_f1={r['adversarial_macro_f1']:.3f}")
    print(f"Saved served model -> {args.model_out}")
    print(f"Saved metrics -> {args.metrics_out}")


if __name__ == "__main__":
    main()
