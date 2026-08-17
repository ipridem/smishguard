# SmishGuard

[![GitHub](https://img.shields.io/badge/GitHub-ipridem%2Fsmishguard-181717?logo=github)](https://github.com/ipridem/smishguard)

A Zimbabwe-context SMS smishing (SMS phishing) classifier: TF-IDF +
engineered social-engineering features over a logistic-regression pipeline,
with a FastAPI JSON API, a static single-page console, per-feature
explainability, and an optional LLM second opinion for messages the local
model can't confidently classify.

## What it detects

Labels: `legit` + 6 fraud categories — `phishing_credential`,
`phishing_reversal_scam`, `fake_agent`, `prize_scam`, `account_takeover`,
`other_fraud`.

The engineered features (17 of them) target attack *structure*, not just
brand keywords or a fixed phrase list — deceptive/lookalike subdomains,
passive-consent SIM-swap and device-linking scams, vishing code-readout,
MFA-push/consent-phishing (approve-during-inbound-call), screen-mismatch
coaching (a message pre-excusing why the real confirmation prompt won't
match its own story), USSD advance-fee ("send airtime, get double back")
offers, and negation/conditional-aware credential and identity-verification
requests. These sit alongside TF-IDF word/char n-grams (with numeric
specifics normalised to placeholder tokens so incidental digits never
become vocabulary) in a `FeatureUnion` feeding a Logistic Regression /
Linear SVM baseline comparison.

The API reports **risk** (P of any fraud class) as the headline metric,
kept distinct from the 6-class argmax **confidence** — a scam that reads as
part reversal-scam, part credential-phishing splits its probability mass
across classes, so confidence alone can look falsely unsure about a message
the model is actually confident is fraudulent.

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
cp .env.example .env        # optional: add GROQ_API_KEY for the LLM second opinion
.venv/Scripts/python -m alembic upgrade head
```

`.env`'s `DATABASE_URL` defaults to SQLite. For Postgres, set it to
`postgresql+psycopg2://user:password@localhost:5432/smishing` and uncomment
`psycopg2-binary` in `requirements.txt`.

## Generate the dataset and train the classifier

```bash
.venv/Scripts/python scripts/generate_sms.py --seed 42
.venv/Scripts/python scripts/train_model.py --seed 42
```

The generator produces a labeled corpus into `sms_messages` — legit + 6
fraud categories, with Shona/English and Ndebele/English code-switched
variants — and writes a dataset card to `data/sms_dataset_card.md`. A
configurable fraction of rows (`--paraphrase-rate`, default 35%) are
additionally rewritten through `PARAPHRASE_GROUPS`: interchangeable
phrasings (word-order swaps, vocabulary substitutions) drawn from real
evasions found during adversarial testing, so training coverage per attack
idea isn't limited to one fixed wording.

`scripts/import_hf_sms.py` can additionally import a labeled pan-African
smishing dataset from HuggingFace, mapped onto this project's label
taxonomy, if you want a larger and more lexically varied training set than
the generator alone produces.

The trainer builds TF-IDF (word + char n-gram) + the engineered features →
Logistic Regression + Linear SVM baselines, evaluates on a held-out test
set *and* an adversarial (multi-technique obfuscated: leet, homoglyph,
fullwidth, zero-width, defanged URLs, spaced letters) test set, saves the
served model (Logistic Regression, for `predict_proba`) to
`ml/smishing_model.joblib`, and writes full metrics to `ml/metrics.json`.

**`adversarial_macro_f1` is the headline metric, not `test_macro_f1`.**
Earlier versions of this trainer used a plain `train_test_split`, which
scatters paraphrases of the same template across train and test — the
clean score was measuring memorisation, not generalisation, and read a
suspicious ~1.00 for both models. The split is now grouped on
`template_id` (`SmsMessage.template_id`, set by `generate_sms.py`) via
`GroupShuffleSplit`, so every paraphrase of one template lands on one side
only, and a fraction of fraud-labeled training rows are additionally
augmented with obfuscated variants so the char n-grams see some of that
surface directly. On seed 42 with a 6,000-row generated corpus this reads
`test_macro_f1` ≈ 0.73–0.82 and `adversarial_macro_f1` ≈ 0.71–0.78
depending on model — both well below 1.0, which is the point: this is the
first honest number this project has produced. `metrics.json` records
`headline_metric: "adversarial_macro_f1"` explicitly. Real reported SMS
(not synthetic templates) remain the highest-value missing ingredient for
closing the clean/adversarial gap further.

## Run

```bash
.venv/Scripts/python -m uvicorn app.main:app --reload --port 5000
```

Visit http://127.0.0.1:5000 for the classifier console (paste a message or
upload a CSV — one SMS per row, first column — for batch classification),
or http://127.0.0.1:5000/docs for the OpenAPI docs. No login — this is a
standalone tool, not a multi-tenant app.

Each result includes the risk score, the closest-match label, every
engineered risk signal the model checked (fired *and* not-found, with the
weight it learned for each), and the top contributing TF-IDF/engineered
features by value × weight.

### Optional: LLM second opinion

Set `GROQ_API_KEY` in `.env` (free key at
[console.groq.com/keys](https://console.groq.com/keys)) to enable a Groq
call for messages whose local risk score lands in the 40–60% inconclusive
band. It's a genuine second opinion, not a replacement — the local
engineered features match known attack *shapes*; the LLM instead reasons
about the coached *action*, which generalizes past paraphrasing no lexicon
has seen yet. Best-effort only: no key, a network failure, or a malformed
response all fall back to `null` and the local classification stands on
its own. Never called on the batch path (cost/latency bound).

### Temporary public URL (Cloudflare quick tunnel)

`scripts/tunnel.ps1` starts the dev server plus a `cloudflared` quick
tunnel and prints the public URL, tearing both down together on Ctrl+C:

```powershell
powershell -File scripts/tunnel.ps1
# or, for a different port/command:
powershell -File scripts/tunnel.ps1 -Command "..." -Port 8080
```

Requires `cloudflared.exe` at `%USERPROFILE%\cloudflared.exe` (no account
needed — [cloudflare.com/products/tunnel](https://www.cloudflare.com/products/tunnel/)).

## Tests

```bash
.venv/Scripts/python -m pytest
```

58 tests covering the engineered features (including explicit false-positive
guards for every fix made during adversarial testing), the API surface, the
LLM integration (mocked — no real network calls in tests), and the dataset
generator.

## License

[MIT](LICENSE)
