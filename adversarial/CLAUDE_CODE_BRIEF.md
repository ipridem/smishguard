# Implementation brief: SmishGuard adversarial hardening

Hand this file to Claude Code from the repo root
(`C:\Users\mazwi\Documents\Claude\Projects\FinGuard-Smishing`), along with
`normalize.py`, `corpus.py` and `run_eval.py`.

## Context you need before starting

An adversarial evaluation of the currently served model
(`ml/smishing_model.joblib`) scored **17/20 true positives, 4/20 false
positives** on a 40-case suite (recall 0.850, precision 0.810). Investigation
found four defects. Every claim below was measured against the served pipeline
in-process — do not re-derive them, but do re-verify after each change.

Read `SMISHGUARD_ADVERSARIAL_REPORT.md` first. Do not skip it; several tasks
below will look wrong without the coefficient data in §4.

**Ground rules**

- Do not hand-edit model coefficients. Every weight problem here is a data
  problem; fix the data and retrain.
- Do not "fix" a failing case by adding its literal wording to a lexicon. The
  defect in tasks 4 and 5 *is* that the detectors match wording instead of
  structure. Adding more wording makes it worse.
- After each task, run `python run_eval.py` from the repo root and paste the
  confusion matrix into your commit message. Any change that raises false
  positives above 4/20 or drops true positives below 17/20 must be justified
  or reverted.
- Run `pytest` after each task. `tests/test_smishing_features.py` is 23 KB of
  existing feature assertions — expect task 1 and task 2 to break some of them
  legitimately (inputs are now canonicalised). Update those tests to assert the
  new intended behaviour; do not delete assertions to make them pass.

---

## Task 0 — Set up the evaluation harness (do this first)

1. Copy `corpus.py` to `tests/adversarial_corpus.py` and `run_eval.py` to
   `scripts/eval_adversarial.py`. Fix the import in the latter accordingly.
2. Add a pytest regression test, `tests/test_adversarial_suite.py`, that loads
   the served model and asserts a **floor**, not an exact score:
   `true_positives >= 17 and false_positives <= 4` at `risk >= 0.5`.
   Mark it `@pytest.mark.slow` if model load time bothers you.
3. Commit this before changing any behaviour, so the baseline is in history.

This is the safety net for everything below. Do not proceed without it.

---

## Task 1 — Canonicalise input before featurisation  [P0]

**Defect.** Case S11 in the suite produces a *completely empty feature vector*
— zero engineered signals, zero TF-IDF contributions — because every lexicon and
regex in `app/smishing/features.py` is Latin-alphabet and the message uses
fullwidth codepoints (`ｈｔｔｐ：／／`, `ＥｃｏＣａｓｈ`). It scored 0.215 and was
labelled legit. Leetspeak (S12) fails the same way.

**Change.**

1. Add the provided `normalize.py` as `app/smishing/normalize.py`.
2. In `app/smishing/features.py`, call `canonicalise(text)` as the **first
   statement** of both:
   - `extract_features()` (line ~416) — reassign `text` before `lower` is
     derived from it, so every downstream regex sees canonical text.
   - `normalize_for_tfidf()` (line ~397) — before `URL_RE.sub(...)`, so
     defanged and fullwidth URLs collapse to `URL_TOKEN` like any other.
3. Import it as `from app.smishing.normalize import canonicalise`. Watch for a
   circular import — `normalize.py` must not import from `features.py`.

**Expected result (already measured, no retraining):** recall 0.850 → 0.900,
false positives unchanged at 4/20, zero regressions. S11 goes 0.215 → 0.974.

**Do not** make de-leeting global. It is vocabulary-gated in `normalize.py` on
purpose: global de-leeting turns `$45.00` into `s45.oo` and destroys
`has_currency_amount` on genuine transaction receipts. If you widen
`DELEET_VOCAB`, re-run the suite and confirm the genuine cases G01, G03, G13,
G15, G20 have not moved upward.

---

## Task 2 — Match scheme-less domains in `URL_RE`  [P0]

**Defect.** `URL_RE = re.compile(r"https?://\S+|www\.\S+")` at line 10. A bare
domain has no scheme and no `www.`, so `_url_domains()` returns `[]` and **five
features silently go to zero at once**: `has_url`, `has_unofficial_url`,
`brand_lookalike_domain`, `deceptive_subdomain`, `has_shortener_url`.

Verified failing inputs:

```
ecocash-help.net/unlock              -> [] domains
zimra-refunds.claim-portal.co        -> [] domains
```

**Change.** Add a scheme-less alternative to `URL_RE`, anchored on a real public
suffix list. Requirements:

- Require a known public suffix, not merely a dot — otherwise "arrive 15 min
  early.Please" matches and you manufacture URLs out of missing spaces.
- Replace `SUFFIX_LABELS` (line ~122) with a proper PSL. The current set misses
  `.help`, `.top`, `.online`, `.gle`, `.me`, `.ly` — all of which appear in the
  suite. Prefer a vendored PSL snapshot over a network-fetching dependency;
  this runs at inference time.
- `_url_domains()` already prepends `http://` for scheme-less matches
  (line ~303), so it should need no change. Verify with a test.

**Acceptance.** Both strings above yield their correct hostname; the existing
`_is_official`, `_is_deceptive_host` and punycode behaviour is unchanged. Note
the `@`-trick case already works correctly
(`https://www.ecocash.co.zw@secure-login-zw.ru/auth` → `secure-login-zw.ru`) —
do not regress it.

---

## Task 3 — Split `brand_spoof_indicator`  [P1]

**Defect.** This feature has coefficient **+0.367 toward the `legit` class** —
the opposite of its intent. It is defined as
`brand_mentioned and (has_url or has_shortcode)` (line ~433), which fires on
almost all legitimate telco traffic ("Econet: dial \*143#"), so the model
learned it as a *legitimacy* marker. It is currently making the S08 and S13
attacks harder to detect.

`has_full_number` has the same problem at **+0.740**, though that one is
defensible — genuine transaction receipts do contain phone numbers.

**Change.** Replace the single boolean with two features whose signs cannot
cancel:

- `brand_with_trusted_channel` — brand mentioned alongside an *official* domain
  (`_is_official`) or a first-party short code. Expected to learn a negative
  (pro-legit) weight.
- `brand_with_untrusted_channel` — brand mentioned alongside an *unofficial*
  domain or an unknown full number. Expected to learn a positive (fraud) weight.

Update `FEATURE_NAMES` (line ~289), `extract_features()`, and `FEATURE_LABELS`
in `app/smishing/model.py` (line ~9) together — `_risk_signals()` keys off
`FEATURE_LABELS`, so a missing entry degrades the UI explanation to a raw
feature name.

**This task requires retraining to take effect** (task 6). A new feature with
no learned weight contributes exactly zero.

---

## Task 4 — Widen `ussd_advance_fee_offer`, add `ussd_embeds_msisdn`  [P2]

**Defect.** `_is_ussd_advance_fee_offer()` (line ~330) requires an explicit
amplification promise from `AMPLIFICATION_WORDS` or a "receive … back" + send-verb
pair. Suite case S14 — `"dial *151*2*2*0784556301*250# and enter your PIN when
prompted"` — is a textbook instance and returns **False**. The feature fired
0 times across all 40 cases despite carrying a +2.073 fraud weight.

**Change.** Add `ussd_embeds_msisdn` from `normalize.py` as a new engineered
feature. A USSD string with a full MSISDN inside it is an *addressed transfer to
a third party*; no legitimate telco template does this, because `*171#` and
`*151*2#` are menu paths. This is wording-independent.

Verified: fires on S14; does not fire on G05 (`Dial *171# to buy`) or G13
(biller code `011977`).

Keep the existing `ussd_advance_fee_offer` — it targets a different thing. Add
training examples for it (task 6) rather than loosening its logic.

---

## Task 5 — Add `has_transaction_reference`  [P1]

**Defect.** Suite case G10 is a genuine family money request
(*"The funeral contribution is $10 per household. Please send to Aunt Rudo on
0772 884 110 before Friday"*) and scored **0.882 as a reversal scam** — the
worst false positive in the suite. Only two signals fired, and the top
contribution was `has_currency_amount` at **+2.724**. That one boolean carries
the decision, with nothing opposing it. Any genuine "please send money to this
number" message — the most common legitimate SMS in this market — is
structurally identical to the real scam S04 (0.941) as far as this model can see.

**Change.** Add `has_transaction_reference` from `normalize.py` as a
**pro-legit** engineered feature. Genuine automated financial SMS carry a
machine-minted reference (`Ref MP250817.1432.K84210`, `REV-88214`, meter and
policy numbers) because the sender has a back end; fraud SMS rarely do.

Note the implementation strips URLs before matching — without that, a scam
shortlink like `bit.ly/ecc-claim9` matches the reference shape and hands the
attacker a free legitimacy signal. Do not remove that step.

Verified separation on the suite: **5/20 genuine, 0/20 smish.**

Add a matching `FEATURE_LABELS` entry. Because this is the model's first genuine
pro-legit signal, `_risk_signals()` sorts by `abs(weight)` and will now surface
it prominently in the UI checklist — check that the frontend copy in
`static/app.js` reads sensibly for a signal whose *presence* is reassuring
rather than alarming.

---

## Task 6 — Fix the corpus and the evaluation protocol  [P1 — the real fix]

Tasks 1–5 are inference-time patches. This is where the actual gains are.

**Defect.** `ml/metrics.json` reports `test_macro_f1: 1.000` — perfect scores
across all seven classes — and `adversarial_macro_f1: 0.198`. The 1.000 is a
warning, not an achievement. Two causes:

1. **Leakage.** `scripts/generate_sms.py` builds messages from hand-written
   templates with a 35% paraphrase rate, and `train_model.py` (line ~86) uses a
   plain `train_test_split`. Paraphrases of one template land on both sides of
   the split, so the test set is largely memorised.
2. **Trivial separability.** The engineered features alone linearly separate the
   corpus, so the model never had to learn language. Measured consequence:
   mean `|coef|` is **0.762** across the 17 engineered features vs **0.037**
   across the 18,120 TF-IDF features — a 20× per-feature gap. This is a
   17-signal rule engine with a decorative embedding layer, which is exactly why
   recall collapses under perturbation.

**Changes, in order:**

1. **Group-split.** Group all paraphrases of one template into the same fold.
   Cheapest route without a migration: derive a grouping key by running the text
   through `normalize_for_tfidf()` (which already replaces amounts, URLs, phone
   numbers and IDs with placeholder tokens), stripping the remaining
   randomised names, and hashing the result; then use
   `sklearn.model_selection.StratifiedGroupKFold` or `GroupShuffleSplit`.
   Cleaner route: add a `template_id` column to `SmsMessage`
   (`app/models/sms.py`) plus an Alembic migration, and have
   `generate_sms.py` record it. Prefer this if you are regenerating the corpus
   anyway. **Expect test F1 to drop well below 1.0. That is the point** — it is
   the first honest number this project will have produced.

2. **Generate the conflict cases.** The corpus currently emits scam-only and
   legit-only feature combinations. Deliberately add:
   - *legit* messages carrying urgency + currency amount + short code
     (the G04, G10, G13 shapes: school fees, family contributions, insurance
     premiums);
   - *fraud* messages carrying none of them (the S04, S18 shapes: polite agent
     reversal, advance-fee job offers — no link, no urgency, no brand);
   - *legit* messages using shorteners and third-party domains (G19, G06);
   - anti-fraud education messages, which are dense with scam vocabulary but
     legitimate (G12).

   Target: a linear model can no longer reach 1.0 on a grouped split. Until
   then it will not learn language.

3. **Train on canonicalised text**, and augment positives with obfuscated
   variants. Note the current `obfuscate()` in `train_model.py` (line ~37) is
   leet + duplicated-letter typos only. Widen it to the families in the suite:
   homoglyph substitution, fullwidth, zero-width insertion, defanged URLs,
   spaced letters. Also note that `canonicalise()` is vocabulary-gated, so it
   will only partially undo `obfuscate()`'s indiscriminate leeting — the
   adversarial number should improve substantially but not reach the clean
   score, and that gap is informative.

4. **Add examples for the five dead detectors.** `requests_personal_id`,
   `passive_consent_device_change`, `authorization_via_inbound_call`,
   `screen_mismatch_coaching`, `ussd_advance_fee_offer` fired **0 times** across
   all 40 cases while carrying fraud weights up to +2.53. They are matching the
   generator's phrasing, not the attack's structure. Add positives with varied
   phrasing — e.g. for the inbound-call family, S07's *"Our agent will now call
   you to confirm this PIN"* matches neither `CALL_PRIMING_LEX` nor
   `APPROVAL_LEX`.

5. **Promote the honest metric.** Make `adversarial_macro_f1` on a grouped split
   the headline in `metrics.json` and the README, not a footnote under a 1.000.
   Also compare `linear_svm` seriously: it currently scores 0.303 adversarial vs
   logistic regression's 0.198. If you keep logistic regression for
   `predict_proba`, say so explicitly as a calibration trade-off.

6. **Consider `class_weight="balanced"`.** The corpus is balanced by
   construction, but the dataset card notes real prevalence is nothing like it.
   Decide deliberately rather than by default.

7. **Regenerate `data/sms_dataset_card.md`.** It currently lists ~3,000 rows
   while `metrics.json` reports 11,453 (8017 + 1145 + 2291). One of them is
   stale.

---

## Task 7 — Product-level, not model-level  [P2]

1. **`risk` vs `confidence`.** Suite cases G04 (0.703) and G11 (0.515) are
   argmax-`legit` but cross the 0.5 risk threshold, so the UI will warn on a
   message the model calls legitimate. `classify()` in `app/smishing/model.py`
   returns both; decide explicitly which gates the verdict and make
   `static/app.js` consistent with that choice.

2. **Two bands, not one verdict.** The risk distributions overlap badly — smish
   spans 0.215–1.000, genuine spans 0.020–0.882 — so **no single threshold
   separates these classes.** Keep 0.5 as the F1 optimum (measured: F1 0.829 at
   0.5, vs 0.791 at 0.4 and 0.811 at 0.6), but consider warn ≥ 0.5 / block ≥ 0.8.

3. **Route the middle band to `llm_review.py`.** The 0.4–0.7 band is precisely
   the population a linear model cannot resolve — G04, G11, G16, S16, S18 all
   live there. That is the highest-value use of the review path you already have.

---

## Verification checklist

- [ ] `pytest` green, with updated (not deleted) feature assertions
- [ ] `python scripts/eval_adversarial.py` ≥ 18 TP, ≤ 4 FP at 0.5
- [ ] No genuine case from {G01, G03, G13, G15, G20} rises above 0.3
- [ ] S11 > 0.9, S12 > 0.5, S13 > 0.5 after tasks 1–2
- [ ] `ecocash-help.net/unlock` and `zimra-refunds.claim-portal.co` resolve to
      hostnames after task 2
- [ ] `https://www.ecocash.co.zw@secure-login-zw.ru/auth` still resolves to
      `secure-login-zw.ru` (regression guard)
- [ ] After retraining: no engineered feature has a positive `coef_` toward
      `legit` that you cannot justify in a comment
- [ ] After retraining: `test_macro_f1 < 1.0` on a grouped split
- [ ] `brand_lookalike_domain` |coef| is no longer among the weakest three
      (currently −0.308, below `has_shortener_url`, which is indefensible for
      what is nearly a proof of phishing)

## Caveats to keep in mind

- 40 cases is small — each case moves F1 by ~2.5 points. Treat the failure
  *families* as the finding, not the decimals. Grow the suite with real reported
  scams as you collect them.
- G10 (family money request) and S04 (agent reversal) differ only by intent, not
  by form. If product policy is to warn on both, G10 is not a false positive and
  precision is really 0.850. Decide that policy question before optimising
  against it.
