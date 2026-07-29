# FinGuard-Smishing

Standalone extraction of FinGuard's smishing (SMS phishing) detection
engine — same code, same behavior, pulled out of the larger
[FinGuard](../FinGuard) capstone project. No login/RBAC (stripped for
standalone use); everything else — dataset generator, TF-IDF+engineered-
feature classifier, adversarial evaluation, paste/CSV-upload UI — is as it
was.

## Setup

```
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
cp .env.example .env        # edit SECRET_KEY for anything beyond local dev
.venv/Scripts/python -m alembic upgrade head
```

`.env`'s `DATABASE_URL` defaults to SQLite. For Postgres, set it to
`postgresql+psycopg2://user:password@localhost:5432/smishing` and uncomment
`psycopg2-binary` in `requirements.txt`.

## Generate the dataset and train the classifier

```
.venv/Scripts/python scripts/generate_sms.py --seed 42
.venv/Scripts/python scripts/train_model.py --seed 42
```

The generator produces a labeled corpus (legit + 5 fraud categories —
`phishing_credential`, `phishing_reversal_scam`, `fake_agent`, `prize_scam`,
`other_fraud` — with Shona/English code-switched variants) into
`sms_messages` and writes a dataset card to `data/sms_dataset_card.md`. The
trainer builds TF-IDF (word + char n-gram) + 7 engineered features
(shortcode/full-number, URL, currency amount, urgency lexicon, PIN/OTP
keywords, brand-spoof heuristic) → Logistic Regression + Linear SVM
baselines, evaluates on a held-out test set *and* an adversarial
(leetspeak-obfuscated) test set, saves the served model (Logistic
Regression, for `predict_proba`) to `ml/smishing_model.joblib`, and writes
full metrics to `ml/metrics.json`.

Note: clean test F1 is inflated by template-generation artifacts
(near-duplicate structure across samples) — the adversarial score is the
more meaningful robustness signal. On seed 42: clean F1 = 1.00, adversarial
F1 ≈ 0.51-0.57 depending on model.

## Run

```
.venv/Scripts/python wsgi.py
```

Visit http://127.0.0.1:5000 — the classifier page directly, no login. Paste
a message or upload a CSV (one SMS per row, first column) for batch
classification. Results include confidence and the top contributing
TF-IDF/engineered features (coefficient-inspection explainability).

## Tests

```
.venv/Scripts/python -m pytest
```

## What's different from the FinGuard version

- No auth/RBAC (Flask-Login, User model, roles) — the classifier page is
  open, no accounts. CSRF protection on the forms is still active.
- No dashboard, cases, transaction analytics, or APK pipeline — this is
  only the smishing engine.
- `scripts/generate_sms.py` inlines its own name list instead of importing
  from a transaction-simulator module that doesn't exist here.
- Everything else (feature engineering, training, inference, templates,
  the `UTCDateTime` column type) is unchanged.
