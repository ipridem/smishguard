"""JSON API surface: single classify, CSV batch, meta, and input validation."""
import io


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


def test_meta_exposes_threshold_and_labels(client):
    body = client.get("/api/meta").json()
    assert 0 < body["low_confidence_threshold"] < 1
    assert "legit" in body["labels"]
