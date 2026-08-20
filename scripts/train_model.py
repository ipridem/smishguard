"""Train the smishing classifier: TF-IDF (word + char n-gram) + engineered
features -> Logistic Regression and Linear SVM baselines. Saves the served
pipeline (best predict_proba-capable model) plus metrics.json with
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
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

from app.config import Config
from app.extensions import db_session, init_db
from app.models.sms import SmsMessage
from app.smishing.features import SmishingFeatureExtractor, normalize_for_tfidf

# Robustness check against obfuscated variants (research contribution #2 in
# the brief): several evasion families applied to the held-out test set (for
# the adversarial score) and to a fraction of training positives (so the
# char n-gram vectorizer sees some of this surface directly, not only via
# canonicalise() at inference time). Deliberately indiscriminate/global,
# unlike canonicalise()'s vocabulary-gated de-leeting -- this is the attacker
# simulation, not the defense, so it should NOT self-limit the same way.
LEET_MAP = str.maketrans({"a": "4", "e": "3", "i": "1", "o": "0", "s": "5"})
HOMOGLYPH_MAP = str.maketrans({
    "a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "y": "у", "x": "х", "i": "і",
})


def _leet_typo(text: str, rng: random.Random) -> str:
    chars = list(text.translate(LEET_MAP))
    i = 0
    while i < len(chars):
        if chars[i].isalpha() and rng.random() < 0.05:
            chars.insert(i, chars[i])  # duplicate a letter, typo-like
            i += 1
        i += 1
    return "".join(chars)


def _homoglyph(text: str, rng: random.Random) -> str:
    return text.translate(HOMOGLYPH_MAP)


def _fullwidth(text: str, rng: random.Random) -> str:
    out = []
    for ch in text:
        cp = ord(ch)
        if 0x21 <= cp <= 0x7E:
            out.append(chr(cp + 0xFEE0))
        elif ch == " ":
            out.append("　")
        else:
            out.append(ch)
    return "".join(out)


def _zero_width_insert(text: str, rng: random.Random, rate: float = 0.08) -> str:
    out = []
    for ch in text:
        out.append(ch)
        if ch.isalpha() and rng.random() < rate:
            out.append("​")
    return "".join(out)


def _defang_urls(text: str, rng: random.Random) -> str:
    def repl(m):
        return m.group(0).replace("https://", "hxxps://").replace("http://", "hxxp://").replace(".", "[.]")
    return re.sub(r"https?://\S+", repl, text)


def _space_out_word(text: str, rng: random.Random) -> str:
    words = re.findall(r"[A-Za-z]{4,}", text)
    if not words:
        return text
    target = rng.choice(words)
    spaced = " ".join(list(target))
    return re.sub(rf"\b{re.escape(target)}\b", spaced, text, count=1)


OBFUSCATION_TECHNIQUES = [_leet_typo, _homoglyph, _fullwidth, _zero_width_insert, _defang_urls, _space_out_word]


def obfuscate(text: str, rng: random.Random) -> str:
    """Apply one randomly chosen evasion technique -- an attacker uses one
    trick at a time, not all of them stacked, and this keeps the adversarial
    set from being uniformly-mangled in a way real evasions aren't."""
    technique = rng.choice(OBFUSCATION_TECHNIQUES)
    return technique(text, rng)


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


def load_corpus() -> tuple[list[str], list[str], list[str]]:
    """Groups: template_id when the row came from a generator template (so
    every paraphrase/parameter-fill of the same template lands in one group),
    else a per-row singleton group (imported/real rows have no duplication
    structure to protect against)."""
    rows = db_session.query(SmsMessage).filter(SmsMessage.label.isnot(None)).all()
    texts = [r.text for r in rows]
    labels = [r.label.value for r in rows]
    groups = [r.template_id or f"row_{r.id}" for r in rows]
    return texts, labels, groups


def _group_split(X: np.ndarray, y: np.ndarray, groups: np.ndarray, test_size: float, seed: int):
    """A plain train_test_split scatters paraphrases of one template across
    both sides, so the test set is largely memorised (this is the leakage
    the adversarial report's test_macro_f1=1.000 was measuring). Grouping on
    template_id keeps every paraphrase of a template on one side."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    return train_idx, test_idx


def _augment_with_obfuscated_positives(X_train, y_train, g_train, rng: random.Random, rate: float = 0.15):
    """Add obfuscated copies of a fraction of fraud-labeled training rows.
    canonicalise() only undoes vocabulary-gated leeting at inference time; it
    can't do anything for the char n-grams unless the vectorizer has also
    seen some obfuscated surface form during fitting. Copies keep their
    source row's group so a grouped re-split downstream still can't leak."""
    fraud_idx = [i for i, y in enumerate(y_train) if y != "legit"]
    n_augment = int(len(fraud_idx) * rate)
    chosen = rng.sample(fraud_idx, min(n_augment, len(fraud_idx)))
    aug_X = [obfuscate(X_train[i], rng) for i in chosen]
    aug_y = [y_train[i] for i in chosen]
    aug_g = [g_train[i] for i in chosen]
    return (
        np.concatenate([X_train, aug_X]),
        np.concatenate([y_train, aug_y]),
        np.concatenate([g_train, aug_g]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--model-out", default=Config.SMISHING_MODEL_PATH)
    parser.add_argument("--metrics-out", default="ml/metrics.json")
    # The corpus is class-balanced by construction (see data/sms_dataset_card.md),
    # but real-world SMS prevalence is nothing like it -- legitimate traffic
    # vastly outnumbers scams. class_weight="balanced" would re-balance an
    # ALREADY-balanced training set, which doesn't address that gap; it's the
    # threshold/product layer (app/smishing/model.py's risk >= 0.5, and the
    # warn/block bands in Task 7) that has to account for real prevalence.
    # Left off by default for that reason -- flag exists so this is a
    # deliberate choice, not an unconsidered default.
    parser.add_argument("--class-weight", default=None, choices=[None, "balanced"])
    args = parser.parse_args()

    rng = random.Random(args.seed)
    init_db(Config.DATABASE_URL)

    texts, labels, groups = load_corpus()
    if len(texts) < 50:
        raise SystemExit("Not enough labeled data — run scripts/generate_sms.py first.")

    texts = np.array(texts, dtype=object)
    labels = np.array(labels, dtype=object)
    groups = np.array(groups, dtype=object)

    holdout_size = args.test_size + args.val_size
    train_idx, temp_idx = _group_split(texts, labels, groups, holdout_size, args.seed)
    X_train, y_train, g_train = texts[train_idx], labels[train_idx], groups[train_idx]
    X_temp, y_temp, g_temp = texts[temp_idx], labels[temp_idx], groups[temp_idx]

    test_fraction_of_temp = args.test_size / holdout_size
    val_idx, test_idx = _group_split(X_temp, y_temp, g_temp, test_fraction_of_temp, args.seed)
    X_val, y_val = X_temp[val_idx], y_temp[val_idx]
    X_test, y_test = X_temp[test_idx], y_temp[test_idx]

    X_train, y_train, g_train = _augment_with_obfuscated_positives(X_train, y_train, g_train, rng)
    adversarial_X = [obfuscate(t, rng) for t in X_test]

    results = {}
    served_pipeline = served_name = None
    for name, estimator in [
        ("logistic_regression", LogisticRegression(max_iter=1000, class_weight=args.class_weight)),
        ("linear_svm", LinearSVC(class_weight=args.class_weight)),
        # LinearSVC scores better but has no predict_proba, which the risk
        # bands need; the calibration wrapper buys it back.
        ("calibrated_linear_svm", CalibratedClassifierCV(LinearSVC(class_weight=args.class_weight))),
    ]:
        pipeline = build_pipeline(estimator)
        pipeline.fit(X_train, y_train)

        val_report = classification_report(y_val, pipeline.predict(X_val), output_dict=True, zero_division=0)
        test_report = classification_report(y_test, pipeline.predict(X_test), output_dict=True, zero_division=0)
        adv_report = classification_report(y_test, pipeline.predict(adversarial_X), output_dict=True, zero_division=0)

        results[name] = {
            # headline: performance under evasion on a leakage-controlled
            # (grouped) split. test_macro_f1 is kept for comparison, but it
            # measures a much easier, non-adversarial question.
            "adversarial_macro_f1": adv_report["macro avg"]["f1-score"],
            "test_macro_f1": test_report["macro avg"]["f1-score"],
            "val_macro_f1": val_report["macro avg"]["f1-score"],
            "test_report": test_report,
            "adversarial_report": adv_report,
        }
        # serve the best model that can produce probabilities
        if hasattr(pipeline, "predict_proba") and (
            served_pipeline is None
            or adv_report["macro avg"]["f1-score"] > results[served_name]["adversarial_macro_f1"]
        ):
            served_pipeline, served_name = pipeline, name

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(served_pipeline, args.model_out)

    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_out).write_text(json.dumps({
        "headline_metric": "adversarial_macro_f1",
        "seed": args.seed,
        "class_weight": args.class_weight,
        "split": "grouped (template_id) via GroupShuffleSplit -- paraphrases of one template never straddle train/val/test",
        "train_size": len(X_train), "val_size": len(X_val), "test_size": len(X_test),
        "served_model": served_name,
        "models": results,
    }, indent=2))

    for name, r in results.items():
        print(f"{name}: adversarial_macro_f1={r['adversarial_macro_f1']:.3f}  test_macro_f1={r['test_macro_f1']:.3f}")
    print(f"Saved served model -> {args.model_out}")
    print(f"Saved metrics -> {args.metrics_out}")


if __name__ == "__main__":
    main()
