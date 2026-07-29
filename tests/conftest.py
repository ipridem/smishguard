import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from app import api
from scripts.train_model import build_pipeline

# A tiny model trained on the fly, so the API tests don't depend on the real
# artifact under ml/ (which is regenerated whenever features change).
TINY_TEXTS = [
    "Confirmed. You have received $50 from John. Balance $200.",
    "Confirmed. You paid $20 to Jane for ZESA. Balance $80.",
    "Your OTP is 123456. Do not share this code.",
    "URGENT verify your PIN now at http://fake-verify.tk/abc to avoid suspension.",
    "Security alert confirm your PIN at http://fake-verify.tk/xyz within 24 hours.",
    "Send your PIN to 12345 to unlock your wallet immediately.",
    "Sorry I sent $50 to you by mistake please reverse it to 4521 urgently.",
    "System error $30 was sent to your account in error refund now.",
    "Agent 4521 needs your PIN to reverse the failed transaction of $40.",
    "This is Agent Harare12 send your PIN to process cash-out of $60.",
    "Congratulations your number has won $500 claim at http://prize-win.xyz/win.",
    "You have been selected to win a car send your PIN to 3321 to claim.",
    "For help with your blocked account call customer care on 5544 now.",
    "EcoCash support desk 5544 call for urgent assistance with your issue.",
]
TINY_LABELS = [
    "legit", "legit", "legit",
    "phishing_credential", "phishing_credential", "phishing_credential",
    "phishing_reversal_scam", "phishing_reversal_scam",
    "fake_agent", "fake_agent",
    "prize_scam", "prize_scam",
    "other_fraud", "other_fraud",
]


@pytest.fixture
def client(tmp_path, monkeypatch):
    pipeline = build_pipeline(LogisticRegression(max_iter=1000), min_df=1)
    pipeline.fit(TINY_TEXTS, TINY_LABELS)
    model_path = tmp_path / "model.joblib"
    joblib.dump(pipeline, model_path)

    api.get_pipeline.cache_clear()
    monkeypatch.setattr(api.Config, "SMISHING_MODEL_PATH", str(model_path))
    from app.main import app

    yield TestClient(app)
    api.get_pipeline.cache_clear()
