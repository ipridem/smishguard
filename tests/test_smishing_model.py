"""Explainability: risk_signals must report every engineered feature,
including ones that DIDN'T fire — the gap _top_tokens can't cover, since a
value*weight contribution is always 0 for an absent (0-valued) feature."""
from sklearn.linear_model import LogisticRegression

from app.smishing.features import FEATURE_NAMES
from app.smishing.model import classify
from scripts.train_model import build_pipeline

TEXTS = [
    "Confirmed. You have received $50 from John. Balance $200.",
    "Confirmed. You paid $20 to Jane for ZESA. Balance $80.",
    "URGENT verify your PIN now at http://fake-verify.tk/abc to avoid suspension.",
    "Security alert confirm your PIN at http://fake-verify.tk/xyz within 24 hours.",
]
LABELS = ["legit", "legit", "phishing_credential", "phishing_credential"]


def test_risk_signals_cover_every_engineered_feature():
    pipeline = build_pipeline(LogisticRegression(max_iter=1000), min_df=1)
    pipeline.fit(TEXTS, LABELS)

    result = classify(pipeline, "Confirmed. You have received $30 from Jane. Balance $90.")
    signals = result["risk_signals"]

    assert len(signals) == len(FEATURE_NAMES)
    # a legit message with no URL/PIN mention: those signals must still
    # appear in the report, just marked absent, not silently dropped
    absent_labels = {s["label"] for s in signals if not s["present"]}
    assert "Contains a URL" in absent_labels
    assert "Requests PIN/OTP/password" in absent_labels
    # sorted by influence, most decision-relevant first
    weights = [abs(s["weight"]) for s in signals]
    assert weights == sorted(weights, reverse=True)


def test_risk_signal_notes_credential_mention_without_request():
    pipeline = build_pipeline(LogisticRegression(max_iter=1000), min_df=1)
    pipeline.fit(TEXTS, LABELS)

    result = classify(
        pipeline,
        "Confirmed. Your bank will never ask for your password by SMS. Balance $90.",
    )
    credential_signal = next(
        s for s in result["risk_signals"] if s["label"] == "Requests PIN/OTP/password"
    )
    assert credential_signal["present"] is False
    assert "note" in credential_signal
