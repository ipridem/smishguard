"""Floor test for the 40-case adversarial suite against the served model.

Asserts a floor, not an exact score — see adversarial/SMISHGUARD_ADVERSARIAL_REPORT.md
for the measured baseline (17 TP / 4 FP) this floor is derived from.
"""
import joblib
import pytest

from app.smishing.model import classify
from tests.adversarial_corpus import CASES

THRESHOLD = 0.5


@pytest.mark.slow
def test_adversarial_suite_floor():
    pipe = joblib.load("ml/smishing_model.joblib")

    tp = fp = 0
    for c in CASES:
        risk = classify(pipe, c["text"])["risk"]
        flagged = risk >= THRESHOLD
        if c["label"] == "smish" and flagged:
            tp += 1
        elif c["label"] == "genuine" and flagged:
            fp += 1

    assert tp >= 17, f"true positives regressed: {tp} < 17"
    assert fp <= 4, f"false positives regressed: {fp} > 4"
