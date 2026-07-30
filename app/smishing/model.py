"""Inference wrapper around the trained smishing classifier pipeline."""
import numpy as np
import joblib

from app.smishing.features import MENTION_LEXICONS, keyword_mentioned

# human-readable labels for the security-meaningful (engineered) features,
# keyed by name as produced in app.smishing.features.FEATURE_NAMES
FEATURE_LABELS = {
    "has_shortcode": "Contains a short-code number",
    "has_full_number": "Contains a full phone number",
    "has_url": "Contains a URL",
    "has_currency_amount": "Mentions a currency amount",
    "urgency_word_count": "Urgency language detected",
    "requests_sensitive_credentials": "Requests PIN/OTP/password",
    "brand_spoof_indicator": "Brand name paired with a suspicious link/shortcode",
    "has_shortener_url": "Link uses a URL shortener",
    "has_unofficial_url": "Link goes to a non-official domain",
    "brand_lookalike_domain": "Brand name embedded in an unofficial domain",
    "requests_identity_verification": "Requests identity/account verification",
    "requests_personal_id": "Requests a national ID / ID number",
    "deceptive_subdomain": "Real domain hidden behind a lookalike subdomain",
    "passive_consent_device_change": "Account change proceeds by default unless you act",
    "authorization_via_inbound_call": "Told to approve a prompt or read a code during an inbound call",
    "screen_mismatch_coaching": "Pre-excuses the approval screen not matching the message's story",
    "ussd_advance_fee_offer": "USSD code paired with a promise of getting back more than you send",
}


def load_model(path: str):
    return joblib.load(path)


def _top_tokens(pipeline, text: str, top_n: int = 5) -> list[dict]:
    """Coefficient-inspection explainability: which TF-IDF/engineered
    features contributed most to the predicted class for this message."""
    features_step = pipeline.named_steps["features"]
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return []

    x = features_step.transform([text])
    x_dense = x.toarray()[0] if hasattr(x, "toarray") else np.asarray(x)[0]
    feature_names = features_step.get_feature_names_out()
    predicted_idx = list(clf.classes_).index(pipeline.predict([text])[0])
    contributions = x_dense * clf.coef_[predicted_idx]

    top_idx = np.argsort(np.abs(contributions))[::-1][:top_n]
    return [
        {"token": str(feature_names[i]), "contribution": float(contributions[i])}
        for i in top_idx if contributions[i] != 0
    ]


def _risk_signals(pipeline, text: str) -> list[dict]:
    """Report every engineered (security-meaningful) feature — whether it
    fired and the model's learned weight for it on the predicted class.

    Unlike _top_tokens (value * weight), this also surfaces features that
    DIDN'T fire: for an absent feature, value=0 forces contribution to 0
    regardless of weight, so a plain contribution ranking can never explain
    "why legit" in terms of risk signals that were checked and not found."""
    features_step = pipeline.named_steps["features"]
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return []

    feature_names = features_step.get_feature_names_out()
    prefix = "engineered__"
    engineered_idx = [i for i, n in enumerate(feature_names) if n.startswith(prefix)]

    x = features_step.transform([text])
    x_dense = x.toarray()[0] if hasattr(x, "toarray") else np.asarray(x)[0]
    predicted_idx = list(clf.classes_).index(pipeline.predict([text])[0])

    signals = []
    for i in engineered_idx:
        name = feature_names[i][len(prefix):]
        present = bool(x_dense[i])
        signal = {
            "label": FEATURE_LABELS.get(name, name),
            "present": present,
            "weight": float(clf.coef_[predicted_idx][i]),
        }
        # requests_* features need an actual request, not just the keyword (see
        # features.py) — "your password was changed" and "we will never ask for
        # your PIN" both mention the noun without asking for it. Surface that so
        # the explanation doesn't read as "no credential mention at all".
        if not present and name in MENTION_LEXICONS and keyword_mentioned(name, text):
            signal["note"] = "keyword mentioned, but not requested — stated as fact, or negated"
        signals.append(signal)
    return sorted(signals, key=lambda s: abs(s["weight"]), reverse=True)


LEGIT_LABEL = "legit"


def classify(pipeline, text: str) -> dict:
    """`confidence` is the 6-class argmax; `risk` is P(any fraud class).

    They answer different questions and diverge on hybrid messages: a scam
    that reads as part reversal, part credential theft splits its probability
    across several fraud classes, so argmax confidence looks low while the
    model is in fact near-certain the message is fraudulent. `risk` is the
    number that matches what a user is actually asking.
    """
    label = pipeline.predict([text])[0]
    confidence = risk = None

    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba([text])[0]
        classes = list(pipeline.classes_)
        confidence = float(max(probs))
        if LEGIT_LABEL in classes:
            risk = 1.0 - float(probs[classes.index(LEGIT_LABEL)])

    return {
        "label": label,
        "confidence": confidence,
        "risk": risk,
        "top_tokens": _top_tokens(pipeline, text),
        "risk_signals": _risk_signals(pipeline, text),
    }
