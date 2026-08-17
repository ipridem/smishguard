"""Pre-featurisation canonicalisation for SmishGuard.

Drop this in as app/smishing/normalize.py, then call `canonicalise(text)`
as the FIRST thing in both `extract_features()` and `normalize_for_tfidf()`.

Rationale: every signal in features.py — URL_RE, BRAND_LEX, the urgency and
credential lexicons — is a Latin-alphabet regex. An attacker who swaps one
codepoint (Cyrillic о for Latin o) or defangs a URL turns the ENTIRE feature
vector to zeros while the message stays perfectly readable to a human. That
is not a weight-tuning problem; the model literally cannot see the attack.
Normalising the input costs nothing at inference and needs no retraining
(though retraining ON canonicalised text is better still — see the report).

Measured on the 40-case adversarial suite: recall 0.850 -> 0.900,
false positives unchanged at 4/20, zero regressions.
"""
from __future__ import annotations

import re
import unicodedata

# Cyrillic / Greek / typographic lookalikes. NFKC does NOT fold these, because
# they are semantically distinct characters — which is exactly why they work as
# an evasion. Keep this list tight: fold only glyphs that are visually
# indistinguishable in a normal SMS font.
CONFUSABLES = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "і": "i", "ѕ": "s", "һ": "h", "ԁ": "d", "ᴏ": "o", "ɡ": "g", "ⅼ": "l",
    "А": "A", "Е": "E", "О": "O", "Р": "P", "С": "C", "У": "Y", "Х": "X",
    "І": "I", "Ѕ": "S", "Н": "H", "В": "B", "М": "M", "К": "K", "Т": "T",
    "ο": "o", "α": "a", "ρ": "p", "ε": "e", "ν": "v", "τ": "t", "ι": "i",
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "‛": "'", "’": "'",
    "“": '"', "”": '"',
}

# Zero-width and invisible formatting characters, used to split keywords
# without any visible change: "E​coCash".
ZERO_WIDTH_RE = re.compile(
    "[​‌‍‎‏⁠﻿­᠎]"
)

# Defanged / obfuscated URLs. Security tooling defangs links so they are not
# clickable; scammers defang them so your regex does not match while the victim
# still knows what to type.
DEFANG_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bh(?:xx|\*\*|__)p(s?)\s*:\s*//", re.IGNORECASE), r"http\1://"),
    (re.compile(r"\[\s*\.\s*\]|\(\s*\.\s*\)|\{\s*\.\s*\}"), "."),
    (re.compile(r"\s+(?:dot|d0t)\s+", re.IGNORECASE), "."),
    (re.compile(r"\[\s*:\s*\]|\(\s*:\s*\)"), ":"),
    (re.compile(r"\[\s*/\s*\]"), "/"),
]

_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a",
                       "5": "s", "7": "t", "$": "s", "@": "a"})

# De-leeting is applied ONLY to tokens whose de-leeted form lands in this
# vocabulary. Global de-leeting is actively harmful: it turns "$45.00" into
# "s45.oo" and kills has_currency_amount on genuine transaction receipts.
DELEET_VOCAB = frozenset("""
ecocash onemoney netone econet telecel zipit innbucks steward stewardbank
zimra potraz zesa mukuru cbz zipit nostro
account accounts password passcode pin otp code login logon signin
verify verification validate confirm confirmed authenticate
secure security help support service customer care
blocked block suspend suspended locked lock deactivate deactivated
unlock unblock reactivate activate restore reset update upgrade
balance wallet transfer transaction reversal reverse refund
claim prize winner won win bonus promo reward voucher
urgent immediately expire expires expiring final warning alert notice
card bank cash money airtime bundle token units
agent parcel delivery courier customs fee charge payment
""".split())

_TOKEN_RE = re.compile(r"[A-Za-z0-9@$]{3,}")
# "W I N N E R" / "c l i c k" — single letters separated by single spaces.
_SPACED_RE = re.compile(r"(?<![A-Za-z])(?:[A-Za-z][ \t]){2,}[A-Za-z](?![A-Za-z])")


def _fold_confusables(text: str) -> str:
    return "".join(CONFUSABLES.get(ch, ch) for ch in text)


def _deleet_token(match: re.Match) -> str:
    token = match.group(0)
    if not any(ch.isdigit() or ch in "@$" for ch in token):
        return token
    candidate = token.translate(_LEET)
    if candidate.lower().strip("@$") in DELEET_VOCAB:
        return candidate
    return token


def canonicalise(text: str) -> str:
    """Fold obfuscation that changes bytes but not human meaning.

    Order matters: NFKC first (fullwidth -> ASCII, so "：／／" becomes "://"),
    then invisibles, then confusables, then defanging, then de-leeting, then
    spaced-letter collapse.
    """
    if not text:
        return text
    out = unicodedata.normalize("NFKC", text)
    out = ZERO_WIDTH_RE.sub("", out)
    out = _fold_confusables(out)
    for pattern, replacement in DEFANG_RULES:
        out = pattern.sub(replacement, out)
    out = _TOKEN_RE.sub(_deleet_token, out)
    out = _SPACED_RE.sub(lambda m: m.group(0).replace(" ", "").replace("\t", ""), out)
    return out


# ----------------------------------------------------------------------
# Two proposed new engineered features, validated against the suite.
# Add these to extract_features() / FEATURE_NAMES / FEATURE_LABELS and
# RETRAIN — a new feature with no learned weight does nothing.
# ----------------------------------------------------------------------

# A USSD string with a full MSISDN embedded in it is a transfer to a THIRD
# PARTY. No legitimate telco template does this — "*171#", "*151*2#" are
# menu paths, not addressed transfers. This is the cleanest structural
# separator in the whole USSD family and it does not depend on the scam's
# wording, unlike the existing ussd_advance_fee_offer detector (which needs
# an explicit "double your money" style promise and therefore missed the
# variant in the suite).
USSD_EMBEDS_MSISDN_RE = re.compile(r"\*\d[\d*]*\*(?:0|263)\d{7,9}[\d*]*#")


def ussd_embeds_msisdn(text: str) -> bool:
    return bool(USSD_EMBEDS_MSISDN_RE.search(text))


# A genuine automated financial SMS almost always carries a machine-generated
# reference: "Ref MP250817.1432.K84210", "REV-88214", a meter or policy number.
# Fraud SMS rarely do, because the sender has no back-end to mint one. This is
# a NEGATIVE (pro-legit) signal, and the model currently has none of real
# quality — which is why genuine payment requests (school fees, a family
# contribution) get flagged: has_currency_amount alone carries +2.72 toward
# phishing_reversal_scam with nothing on the other side of the scale.
#
# Measured on the suite: fires on 5/20 genuine, 0/20 smish.
_STRIP_URL_RE = re.compile(
    r"https?://\S+|www\.\S+|\b\S+\.(?:com|net|org|co|zw|ru|top|help|online|gle|ly|me)\b/?\S*",
    re.IGNORECASE,
)
TXN_REFERENCE_RE = re.compile(
    r"\b(?:ref(?:erence)?|txn|trx|receipt|policy|meter|token|order|shipment|rev)\b"
    r"[\s:#\-]*"
    r"(?=[A-Z0-9.\-/]{6,})"   # long enough to be machine-generated
    r"(?=[^\s]*\d)"           # must contain a digit
    r"[A-Z0-9][A-Z0-9.\-/]{5,}",
    re.IGNORECASE,
)


def has_transaction_reference(text: str) -> bool:
    """URLs are stripped first: without that, a scam shortlink like
    'bit.ly/ecc-claim9' matches the reference shape and hands the attacker a
    pro-legit signal for free."""
    return bool(TXN_REFERENCE_RE.search(_STRIP_URL_RE.sub(" ", text)))
