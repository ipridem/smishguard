"""JSON API surface: single classify, CSV batch, meta, and input validation."""
import io

from app import api


def test_classify_returns_label_and_explainability(client):
    resp = client.post("/api/classify", json={"text": "URGENT verify your PIN now"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] in {
        "legit", "phishing_credential", "phishing_reversal_scam",
        "fake_agent", "prize_scam", "account_takeover", "other_fraud",
    }
    assert 0.0 <= body["confidence"] <= 1.0
    assert 0.0 <= body["risk"] <= 1.0
    assert body["risk_signals"], "risk signals must always be reported"


def test_risk_survives_a_split_across_fraud_classes(client):
    """The 6-class argmax collapses when a scam reads as several fraud types at
    once. risk = P(any fraud) must stay decisive there — a security tool that
    sounds unsure about a confident detection is failing in the worst direction."""
    body = client.post("/api/classify", json={
        "text": "OneMoney: A cash-out of USD 60.00 is being processed. If this was not you, "
                "reply with the OTP you just received to block the transaction.",
    }).json()
    assert body["label"] != "legit"
    assert body["risk"] >= body["confidence"], "risk must not understate a split-class fraud"


def test_classify_rejects_empty_text(client):
    assert client.post("/api/classify", json={"text": ""}).status_code == 422


def test_batch_classifies_each_row(client):
    csv_bytes = b"Confirmed you paid $10 to Jane for ZESA\nSend your PIN to 12345 immediately\n"
    resp = client.post(
        "/api/classify/batch",
        files={"file": ("batch.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 2
    assert all(r["text"] and r["label"] for r in rows)


def test_batch_rejects_non_csv(client):
    resp = client.post(
        "/api/classify/batch",
        files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert resp.status_code == 400


def test_meta_exposes_threshold_and_labels(client, monkeypatch):
    monkeypatch.setattr(api.Config, "GROQ_API_KEY", None)   # isolate from a real .env
    body = client.get("/api/meta").json()
    assert 0 < body["low_confidence_threshold"] < 1
    assert "legit" in body["labels"]
    assert body["llm_review_available"] is False


def test_llm_opinion_absent_without_a_configured_key(client, monkeypatch):
    """No GROQ_API_KEY -> llm_opinion must stay null even when risk is
    inconclusive, and groq_second_opinion must not even be attempted."""
    monkeypatch.setattr(api.Config, "GROQ_API_KEY", None)   # isolate from a real .env
    monkeypatch.setattr(api, "classify", lambda pipeline, text: {
        "label": "legit", "confidence": 0.5, "risk": 0.5, "top_tokens": [], "risk_signals": [],
    })

    def fail_if_called(text):
        raise AssertionError("groq_second_opinion should not be called without an API key")

    monkeypatch.setattr(api, "groq_second_opinion", fail_if_called)
    body = client.post("/api/classify", json={"text": "anything"}).json()
    assert body["llm_opinion"] is None


def test_llm_opinion_only_attached_in_the_inconclusive_band(client, monkeypatch):
    monkeypatch.setattr(api.Config, "GROQ_API_KEY", "test-key")
    canned = {"verdict": "fraud", "confidence": 0.7, "reasoning": "coaches an out-of-band approval"}
    monkeypatch.setattr(api, "groq_second_opinion", lambda text: canned)

    # inconclusive risk -> attached
    monkeypatch.setattr(api, "classify", lambda pipeline, text: {
        "label": "legit", "confidence": 0.5, "risk": 0.5, "top_tokens": [], "risk_signals": [],
    })
    assert client.post("/api/classify", json={"text": "x"}).json()["llm_opinion"] == canned

    # decisive risk -> not attached, even though a key is configured
    monkeypatch.setattr(api, "classify", lambda pipeline, text: {
        "label": "phishing_credential", "confidence": 0.95, "risk": 0.95, "top_tokens": [], "risk_signals": [],
    })
    assert client.post("/api/classify", json={"text": "x"}).json()["llm_opinion"] is None


def test_llm_opinion_never_attached_on_batch(client, monkeypatch):
    """Cost/latency bound: the LLM leg is single-message only."""
    monkeypatch.setattr(api.Config, "GROQ_API_KEY", "test-key")
    monkeypatch.setattr(api, "groq_second_opinion", lambda text: {
        "verdict": "fraud", "confidence": 0.7, "reasoning": "x",
    })
    csv_bytes = b"Send your PIN to 12345 immediately\n"
    rows = client.post(
        "/api/classify/batch",
        files={"file": ("batch.csv", io.BytesIO(csv_bytes), "text/csv")},
    ).json()
    assert all(r.get("llm_opinion") is None for r in rows)
