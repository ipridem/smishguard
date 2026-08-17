# SmishGuard — adversarial evaluation & training feedback

**Date:** 17 August 2026
**Model under test:** `ml/smishing_model.joblib` — `LogisticRegression` over a
`FeatureUnion` of word TF-IDF (1–2 grams), char_wb TF-IDF (3–5 grams), and 17
engineered signals.
**Suite:** 40 hand-written cases — 20 evasion-hardened smishing, 20 genuine
messages built specifically as false-positive traps (Zimbabwe mobile-money
context: EcoCash, OneMoney, ZIPIT, ZESA, POTRAZ, ZIMRA).

I ran these through the actual pipeline in-process (not through the tunnel), so
these are the served model's real scores. Threshold = `risk >= 0.5`, where
`risk = 1 - P(legit)`, matching `app/smishing/model.py`.

---

## 1. Headline results

| | |
|---|---|
| True positives | 17 / 20 |
| False negatives | **3 / 20** |
| False positives | **4 / 20** |
| Recall | 0.850 |
| Precision | 0.810 |
| F1 | 0.829 |

That is a respectable score for a linear model on genuinely hard cases. But the
number that matters more is already in your own `ml/metrics.json`:

```
test_macro_f1         1.000
adversarial_macro_f1  0.198
```

**A held-out test F1 of exactly 1.000 across all seven classes is not a good
result — it is a warning.** It means your synthetic corpus is linearly separable
by the engineered features alone, so the model never had to learn language. Your
own adversarial split confirms it: F1 collapses to 0.198, and `legit` recall
goes to 0.999 with precision 0.424 — under adversarial pressure the model
answers "legit" to almost everything. My suite lands between the two (0.829)
because it is smaller and half of it is deliberately clean genuine traffic.

Everything below is downstream of that one fact.

---

## 2. The three missed attacks

| ID | Family | Risk | Predicted |
|---|---|---|---|
| S11 | fullwidth-unicode | **0.215** | legit |
| S13 | whatsapp-redirect | 0.330 | legit |
| S12 | leetspeak | 0.393 | legit |

### S11 — fullwidth codepoints (the worst failure)

```
ＥｃｏＣａｓｈ： Ｙｏｕｒ　ａｃｃｏｕｎｔ　ｗｉｌｌ　ｂｅ　ｃｌｏｓｅｄ．
Ｖｅｒｉｆｙ： ｈｔｔｐ：／／ｅｃｃ－ｖｅｒｉｆｙ．ｎｅｔ
```

**Zero engineered signals fired. Zero TF-IDF tokens contributed.** The feature
vector is empty. `URL_RE` is `https?://\S+` and the message contains `ｈｔｔｐ：／／`
— different codepoints, no match. `BRAND_LEX` misses `ＥｃｏＣａｓｈ`. Every
lexicon in `features.py` is a Latin-alphabet regex, and this message contains no
Latin alphabet at all. The 0.215 is just the class prior.

This is not a weight problem. **The classifier cannot see the message.**

### S12 — leetspeak

```
3c0Cash Alert: y0ur acc0unt is on h0ld. Re-activate at ecocash-he1p.net/unl0ck ...
```

Also **zero engineered signals**. Note the domain `ecocash-he1p.net` — no
scheme, no `www.`, so `URL_RE` misses it entirely, which means
`has_unofficial_url`, `brand_lookalike_domain` and `deceptive_subdomain` all stay
at zero even though the string "ecocash" is sitting right there in an
attacker-controlled hostname.

I verified this separately: **bare domains with no scheme are invisible to every
URL-derived feature.** `zimra-refunds.claim-portal.co` (S17) and
`ecocash-help.net` both parse to `[]` domains.

### S13 — WhatsApp redirect

```
EcoCash Customer Care: ... Chat with your assigned officer on WhatsApp:
https://wa.me/263779112044
```

Here four signals *did* fire, and the message still scored 0.330. The
contribution breakdown shows why — see §4.

---

## 3. The four false alarms

| ID | Family | Risk | Predicted |
|---|---|---|---|
| G10 | community-money | **0.882** | phishing_reversal_scam |
| G04 | school-fees | 0.703 | legit* |
| G16 | university | 0.584 | phishing_credential |
| G11 | tax-legit | 0.515 | legit* |

\* G04 and G11 illustrate the `confidence` / `risk` divergence your docstring
describes: argmax is `legit`, but probability mass spread across fraud classes
pushes `risk` over 0.5. Worth deciding deliberately which number gates the UI —
right now a message the model calls `legit` is still shown as high-risk.

**G10 is the one to fix.** A real family message:

> "The funeral contribution is $10 per household. Please send to Aunt Rudo on
> 0772 884 110 before Friday..."

scored **0.882** as a reversal scam. Only two signals fired, and the top
contribution was `has_currency_amount` at **+2.724**. That single boolean is
carrying the entire decision. Any genuine "please send money to this number"
message — the single most common legitimate SMS in this market — is
structurally indistinguishable from S04 (the real agent-reversal scam, 0.941) to
your current model. The only difference is context, and the model has no
representation of context.

---

## 4. Root cause: the coefficients

I inspected `clf.coef_` directly. Two things fall out.

### 4a. TF-IDF is decorative

| | mean \|coef\| | feature count |
|---|---|---|
| engineered | **0.762** | 17 |
| TF-IDF | **0.037** | 18,120 |

A 20× per-feature gap. In every failure I traced, engineered contributions were
±0.4–2.7 and the best TF-IDF token was ±0.1–0.2. **You have effectively built a
17-signal rule engine with an 18,000-feature decorative layer.** That is why
recall dies the moment an attacker perturbs the surface form: there is no
language model underneath to fall back on. It is also why `min_df=2` on a
synthetic corpus is dangerous — the vocabulary is memorising generator
templates, not the market's language.

### 4b. Two signals have the wrong sign

`coef_` for the `legit` class (negative = pushes toward fraud, which is what you
want for a risk signal):

| feature | coef_legit | max coef across fraud classes |
|---|---|---|
| `requests_identity_verification` | −2.156 | 1.373 |
| `requests_sensitive_credentials` | −1.901 | 2.010 |
| `has_currency_amount` | −1.526 | **2.724** |
| `has_shortcode` | −1.410 | 1.635 |
| `screen_mismatch_coaching` | −1.365 | 1.778 |
| `passive_consent_device_change` | −1.408 | 2.289 |
| `deceptive_subdomain` | −1.192 | 2.312 |
| `urgency_word_count` | −1.023 | 0.998 |
| `authorization_via_inbound_call` | −0.967 | 2.528 |
| `ussd_advance_fee_offer` | −0.702 | 2.073 |
| `requests_personal_id` | −0.688 | 1.872 |
| `has_unofficial_url` | −0.539 | 0.852 |
| `has_shortener_url` | −0.469 | 1.263 |
| `brand_lookalike_domain` | **−0.308** | 0.602 |
| `has_url` | −0.048 | 0.801 |
| `brand_spoof_indicator` | **+0.367** | 1.169 |
| `has_full_number` | **+0.740** | 0.989 |

- **`brand_spoof_indicator` is +0.367 toward legit.** Your own docstring calls
  this "a classic spoofing pattern." The model learned the opposite, because in
  your synthetic corpus legit telco messages routinely pair a brand with a
  short code ("Econet: dial \*143#"). The feature as defined (`brand AND (url OR
  shortcode)`) fires on almost all legitimate telco traffic, so it became a
  *legitimacy* marker. It is currently making S08 and S13 harder to catch.
- **`brand_lookalike_domain` is the second-weakest signal in the model
  (−0.308).** "The string `ecocash` appears in a domain we do not own" is close
  to a proof of phishing, and it is weighted below `has_shortener_url`. Cause:
  too few positive examples in training, and the feature can only fire when
  `URL_RE` matched in the first place.
- I confirmed the practical consequence with a clean, un-obfuscated probe:

  ```
  "EcoCash Alert: your account is on hold. Re-activate at
   http://ecocash-help.net/unlock or lose your balance."
   -> risk 0.258, label=legit
  ```

  Textbook phishing, explicit lookalike domain, three signals firing —
  **scored as legit, and scored *lower* than the same message with no URL at
  all** (0.296). The URL-family weights are not just weak, they are net
  counterproductive.

### 4c. Five signals never fire

`requests_personal_id`, `passive_consent_device_change`,
`authorization_via_inbound_call`, `screen_mismatch_coaching`,
`ussd_advance_fee_offer` — **0 firings across all 40 cases**, despite carrying
some of the largest fraud weights in the model (up to +2.53).

That is over-narrow detectors, not absent attacks. My suite contains textbook
instances of two of them:

- **S07** is an inbound-call OTP relay: *"Our agent will now call you to confirm
  this PIN."* Neither `CALL_PRIMING_LEX` nor `APPROVAL_LEX` matches. (It was
  caught anyway, at 0.932, by the credential-request signal.)
- **S14** is a USSD advance-fee scam: *"dial \*151\*2\*2\*0784556301\*250# and
  enter your PIN to receive your \$250 prize."* `_is_ussd_advance_fee_offer`
  requires an amplification promise from `AMPLIFICATION_WORDS` or a
  "receive ... back" + send-verb pair. Neither is present, so it returns False.

These detectors are matching your *generator's phrasing* rather than the
attack's *structure*. They will keep scoring near-zero on real traffic.

---

## 5. Recommendations, in priority order

### P0 — Canonicalise input before featurisation

Ships in `normalize.py` (attached). Call `canonicalise(text)` as the first line
of both `extract_features()` and `normalize_for_tfidf()`. It applies NFKC,
strips zero-width characters, folds Cyrillic/Greek confusables, re-fangs
`hxxp://` and `[.]`, de-leets only tokens that resolve to a known brand or scam
word, and collapses `W I N N E R`.

**Measured, no retraining:** recall 0.850 → 0.900, false positives unchanged at
4/20, **zero regressions** (S11 0.215 → 0.974; S02 +0.063; S15 +0.072; no
genuine case moved upward by more than 0.01).

Deliberately narrow choices worth keeping: de-leeting is vocabulary-gated
because global de-leeting turns `$45.00` into `s45.oo` and destroys
`has_currency_amount` on genuine receipts; the confusable table only covers
glyphs that are visually identical in an SMS font.

### P0 — Match bare domains in `URL_RE`

`URL_RE = r"https?://\S+|www\.\S+"` misses `ecocash-help.net/unlock` and
`zimra-refunds.claim-portal.co`, which zeroes *five* URL-derived features at
once. Add a scheme-less alternative anchored on a public-suffix list (a real
PSL, not `SUFFIX_LABELS` — `.help`, `.top`, `.online`, `.gle` all appear in this
suite and none are in your current set). Guard against matching sentence-final
words: require a known suffix, not just a dot.

### P1 — Fix the training data, not the weights

Do not hand-edit coefficients. `brand_spoof_indicator` and `has_full_number`
learned the wrong sign because of what the corpus contains, so:

1. **Break the separability.** Test F1 = 1.000 means your generator emits
   scam-only and legit-only feature combinations. Deliberately generate the
   *conflict* cases: legit messages with urgency + amount + short code (G04,
   G10, G13 shapes), scams with none of them (S04, S18 shapes). Until a linear
   model can no longer reach 1.0, it will not learn language.
2. **Split `brand_spoof_indicator`.** Brand + *official* domain and brand +
   *own* short code are legitimacy markers; brand + unofficial domain and brand
   + unknown full number are fraud markers. As one boolean they cancel and the
   majority class (legit) wins the sign.
3. **Train on canonicalised text**, and additionally augment with obfuscated
   variants of existing positives (homoglyph, leet, fullwidth, defanged). P0
   makes the attack visible; augmentation makes the char_wb n-grams robust to
   variants normalisation misses.
4. **Report a leakage-controlled metric.** Group-split so paraphrases of one
   template never straddle train/test, and treat `adversarial_macro_f1` as the
   headline number in `metrics.json` rather than a footnote. Consider promoting
   `linear_svm` (adversarial F1 0.303 vs 0.198) if you keep a linear model.
5. **Widen the five dead detectors** to structural triggers. Concretely, for the
   inbound-call family: any message that both supplies/references a code and
   references a *call* from the sender, without needing a specific approval
   verb.

### P1 — Add a pro-legit signal so amounts stop dominating

G10 fails because `has_currency_amount` (+2.724) has nothing opposing it.
`normalize.py` ships `has_transaction_reference()`: genuine automated financial
SMS carry a machine-minted reference (`Ref MP250817.1432.K84210`, `REV-88214`,
meter/policy numbers) because the sender has a back end; fraud SMS rarely do.
URLs are stripped first, otherwise a scam shortlink like `bit.ly/ecc-claim9`
matches the reference shape and hands the attacker a free legitimacy signal.

**Measured separation on the suite: 5/20 genuine, 0/20 smish.**

### P2 — Add `ussd_embeds_msisdn`

Also in `normalize.py`. A USSD string with a full MSISDN inside it
(`*151*2*2*0784556301*250#`) is an *addressed transfer to a third party*. No
legitimate telco template does this — `*171#` and `*151*2#` are menu paths.
Verified: fires on S14, does not fire on G05 (`Dial *171# to buy`) or G13
(biller code). This is wording-independent, unlike the current detector.

### P2 — Threshold

Sweep on this suite:

| threshold | recall | precision | F1 | FP | FN |
|---|---|---|---|---|---|
| 0.3 | 0.950 | 0.633 | 0.760 | 11 | 1 |
| 0.4 | 0.850 | 0.739 | 0.791 | 6 | 3 |
| **0.5** | 0.850 | 0.810 | 0.829 | 4 | 3 |
| 0.6 | 0.750 | 0.882 | 0.811 | 2 | 5 |
| 0.8 | 0.650 | 0.929 | 0.765 | 1 | 7 |

0.5 is the F1 optimum, so keep it. But note the risk distributions overlap
badly — smish spans 0.215–1.000, genuine spans 0.020–0.882 — so **no threshold
separates these classes.** Consider a two-band UI (warn ≥0.5, block ≥0.8)
rather than one verdict, and route the 0.4–0.7 band to your `llm_review.py`
path, which is exactly the population a linear model cannot resolve.

---

## 6. Caveats

- 40 cases is small: each result moves F1 by ~2.5 points. Treat the *failure
  families* as the finding, not the decimals.
- Labels are mine. G10 (family money request) and S04 (agent reversal) are
  near-identical in form and differ only by intent — if your product's policy is
  to warn on both, G10 is not a false positive and precision rises to 0.850.
- The model pickled from scikit-learn 1.9.0; I installed 1.9.0 to match before
  scoring, so there is no version-skew risk in these numbers.
- I did not retrain. All measured improvements are inference-time only, which is
  also their limitation — §5's P1 items are where the real gains are.

## 7. Files

| file | purpose |
|---|---|
| `corpus.py` / `corpus.csv` | the 40 cases, with `family` tags and a per-case note on what each attacks |
| `run_eval.py` | scores the suite in-process, prints confusion matrix, threshold sweep, signal firing rates, and per-case contributions |
| `normalize.py` | drop-in `canonicalise()` plus the two proposed features |
| `results.json` | full per-case output: risk, label, every firing signal, top contributions |

`run_eval.py` expects to run from the repo root with `ml/smishing_model.joblib`
present. Point `corpus.py` at your own message log to re-baseline on real
traffic.
