"""Engineered SMS features, complementing TF-IDF word/char n-grams: shortcode
vs full-number presence, URLs, currency amounts, urgency lexicon, PIN/OTP
requests, and a brand-spoofing heuristic."""
import re
from urllib.parse import urlparse

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
# 3-5 bare digits, but never a calendar year — "changed on 29 Jul 2026" is a
# timestamp, not an SMS short code
SHORTCODE_RE = re.compile(r"(?<!\d)(?!(?:19|20)\d\d(?!\d))\d{3,5}(?!\d)")
# A short code is a DESTINATION — digits you are told to contact. Bare digits of
# that length are far more often a product name ("Microsoft 365"), a quantity, or
# a reference, so the digits only count when a contact cue sits immediately
# before them ($ anchors the cue to the end of the preceding window).
SHORTCODE_CUE_RE = re.compile(
    r"\b(?:to|on|at|via|dial|call|calling|sms|text|contact|reply"
    r"|ku|kuno|kwa|pa|fonerai|tumirai|thumela)\b[\s:,\-]*$",
    re.IGNORECASE,
)
FULL_NUMBER_RE = re.compile(r"(?<!\d)\d{9,}(?!\d)")
# Zimbabwe first, then the rest of the region — a ZW-only list is blind to most
# of the imported pan-African corpus, which silently skews anything the model
# learns from this feature.
CURRENCY_CODES = (
    "usd|zig|zwg|zwl|zar|kes|ngn|tzs|zmw|mzn|etb|ghs|rwf|ugx|bwp|mwk|xof|xaf|mad|egp"
)
CURRENCY_RE = re.compile(rf"\$\s?\d|\b(?:{CURRENCY_CODES})\s?\d", re.IGNORECASE)
# full-span versions of the above, used only to normalize numeric specifics
# out of text before TF-IDF vectorization (see normalize_for_tfidf below)
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\s?(?:am|pm)?\b", re.IGNORECASE)
TRACKING_ID_RE = re.compile(r"\b[A-Za-z]{1,6}\d{4,}\b")
CURRENCY_AMOUNT_RE = re.compile(
    rf"\$\s?\d[\d,]*(?:\.\d+)?|\b(?:{CURRENCY_CODES})\s?\d[\d,]*(?:\.\d+)?", re.IGNORECASE
)
URGENCY_WORDS = [
    "urgent", "immediately", "now", "act now", "suspend", "suspended",
    "blocked", "expire", "expires", "24 hours", "today only", "hurry", "asap",
    # Shona
    "kurumidza", "chimbidza", "ikozvino", "nhasi", "ndapota",
    # Ndebele
    "phangisa", "khathesi", "lamuhla", "masinyane",
]
PIN_OTP_WORDS = [
    "pin", "otp", "password",
    # a code the victim reads out to a caller is the same theft as one they
    # text back — the delivery channel differs, the credential doesn't
    "verification code", "security code", "one-time code", "one time code",
    "access code", "otp code",
    # Shona/Ndebele scam SMS keep "PIN"/"OTP" in English but wrap them in
    # local-language requests: tumirai/thumela = send
    "tumirai pin", "thumela i-pin",
]
# generic "verify your identity/account" language — a broader, weaker signal
# than an explicit PIN/OTP/password request (see requests_sensitive_credentials
# above). Split by part of speech: the verbs ARE the demand, so they stand
# alone; "verification" is a noun that appears in ordinary notifications
# ("your verification code is …"), so it needs a request cue like any noun.
VERIFY_VERBS = ["verify", "confirm code", "confirmai"]
VERIFY_NOUNS = ["verification"]
VERIFY_WORDS = VERIFY_VERBS + VERIFY_NOUNS   # union, for the explanation layer
# requests for a national ID / ID number — SIM-registration (POTRAZ) themed
# smishing is a real Zimbabwean vector distinct from PIN/OTP requests; phrased
# as "ID number" specifically so it doesn't fire on "bring your ID" (an
# in-person collection instruction, not a request to send it over SMS)
PERSONAL_ID_WORDS = ["id number", "national id", "identity number", "copy of your id"]
# real institutions' anti-phishing advisories ("we will never ask for your
# PIN") mention the same keywords as an actual request does; a plain
# substring match can't tell them apart, so keyword hits are only counted in
# sentences that don't also contain a negation word
NEGATION_WORDS = ["never", "won't", "will not", "don't", "do not", "cannot", "can't", "not ask"]
# "pin"/"otp"/"password"/"id number" are neutral NOUNS — they appear in real
# notifications ("your OTP is 123456", "your password was changed") exactly as
# often as in scams. The noun only signals a request when an imperative aimed
# at the user sits beside it, so the noun lexicons require one of these cues.
# (VERIFY_WORDS need no cue — "verify"/"confirm" are themselves the imperative.)
REQUEST_CUES = [
    "send", "reply", "enter", "provide", "share", "submit", "give", "forward",
    "confirm", "verify", "update", "input", "type", "need", "require", "text us",
    "complete", "finish",
    # disclosure over a voice call — "read it to the agent who calls you"
    "read", "tell", "repeat", "dictate",
    # Shona / Ndebele imperatives
    "tumirai", "thumela", "confirmai", "kukumbira", "ipai", "nyorai", "verengai",
]
BRAND_NAMES = [
    "ecocash", "onemoney", "innbucks", "telecash", "omari", "mukuru", "zipit",
]
# real institutions never link anywhere else in SMS; anything off-list is suspect
OFFICIAL_DOMAINS = {
    "ecocash.co.zw", "econet.co.zw", "onemoney.co.zw", "netone.co.zw",
    "innbucks.co.zw", "telecel.co.zw", "mukuru.com", "cbz.co.zw",
    "stewardbank.co.zw", "zimpost.co.zw",
}
# Public-suffix labels. If one shows up in the INTERIOR of a hostname rather
# than at its end, the host is wearing another domain as a costume:
# secure.zimpay.co.zw.account-review.com registers as account-review.com but
# reads as zimpay.co.zw at a glance. Brand-agnostic on purpose — BRAND_NAMES
# only catches impersonations of brands we happened to list, which is exactly
# how a lookalike for an unlisted brand slips through.
SUFFIX_LABELS = {
    "co", "com", "org", "net", "gov", "edu", "ac", "info", "biz",
    "zw", "za", "ke", "ng", "uk",
}
# A real anti-fraud control on an account-changing request (SIM swap, device
# link, mobile-banking transfer) requires ACTION to stop a request you didn't
# make. A message that instead makes silence the default for the fraud branch
# — "reply to CONFIRM, do nothing if it wasn't you" — inverts that: doing
# nothing lets the change proceed. This is a documented SIM-swap social-
# engineering pattern; it carries no PIN/URL/currency ask, so every other
# signal in this file is blind to it.
DEVICE_CHANGE_PHRASES = [
    "replacement handset", "new device", "new sim", "sim swap", "sim card",
    "move mobile banking", "move your mobile banking", "change your registered number",
    "transfer your line", "link a new device", "port your number",
]
PASSIVE_CONSENT_PHRASES = [
    "no response is needed", "no response needed", "no action is needed",
    "no action needed", "no further action",
]
# presence of any of these on the "if this wasn't you" branch means the
# message DOES ask for action to stop the change — the safe design
STOP_CUES = [
    "contact", "call", "report", "cancel", "block", "branch",
    "customer care", "helpdesk", "help desk",
]
# MFA-push / consent-phishing: an inbound call is primed, then the victim is
# told to APPROVE a prompt or READ a code "during the call" to complete a
# "security check". Legit push-approvals are user-initiated — a service never
# rings you and walks you through approving one. Often paired with a disarming
# "no PIN/code needed" (which also dodges the credential features). The tell is
# the conjunction of call-priming + an authorization instruction, so neither
# half fires anything on its own.
CALL_PRIMING_PHRASES = [
    "will call", "technician will call", "agent will call", "officer will call",
    "call you shortly", "during the call", "when we call", "when our agent calls",
    "our agent calls", "our agent will call", "is calling now", "will phone",
    "expect a call", "achakufonerai", "ari kufona",
]
# authentication-flavoured approval/disclosure verbs — deliberately narrow so a
# benign "technician will call to confirm the appointment" does not trip it
APPROVAL_CUES = [
    "approve", "allow", "accept", "authorise", "authorize",
    "tap approve", "tap allow", "tap accept",
    "read the code", "read out", "read the verification", "repeat the code",
    "repeat the security", "share the code", "provide the code",
    "tell them the", "dictate", "verengai",
]
SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "cutt.ly", "is.gd", "t.co", "goo.gl",
    "rb.gy", "tiny.cc", "rebrand.ly",
}

def _lexicon(words: list[str]) -> re.Pattern:
    """Compile a word list to a word-boundary regex.

    Plain `"pin" in text` matches "stopPINg", "shopping" and "opinion"; `"now"`
    matches "kNOW" and "snowfall". Substring matching on short lexicon entries
    manufactures signals out of ordinary English, so every lexicon goes through
    here — longest-first so "verification code" wins over a bare prefix.
    """
    alternatives = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
    return re.compile(rf"\b(?:{alternatives})\b", re.IGNORECASE)


URGENCY_LEX = _lexicon(URGENCY_WORDS)
PIN_OTP_LEX = _lexicon(PIN_OTP_WORDS)
VERIFY_VERB_LEX = _lexicon(VERIFY_VERBS)
VERIFY_NOUN_LEX = _lexicon(VERIFY_NOUNS)
VERIFY_LEX = _lexicon(VERIFY_WORDS)
PERSONAL_ID_LEX = _lexicon(PERSONAL_ID_WORDS)
NEGATION_LEX = _lexicon(NEGATION_WORDS)
REQUEST_CUE_LEX = _lexicon(REQUEST_CUES)
BRAND_LEX = _lexicon(BRAND_NAMES)
DEVICE_CHANGE_LEX = _lexicon(DEVICE_CHANGE_PHRASES)
PASSIVE_CONSENT_LEX = _lexicon(PASSIVE_CONSENT_PHRASES)
STOP_CUE_LEX = _lexicon(STOP_CUES)
CALL_PRIMING_LEX = _lexicon(CALL_PRIMING_PHRASES)
APPROVAL_LEX = _lexicon(APPROVAL_CUES)

FEATURE_NAMES = [
    "has_shortcode", "has_full_number", "has_url", "has_currency_amount",
    "urgency_word_count", "requests_sensitive_credentials", "brand_spoof_indicator",
    "has_shortener_url", "has_unofficial_url", "brand_lookalike_domain",
    "requests_identity_verification", "requests_personal_id",
    "deceptive_subdomain", "passive_consent_device_change",
    "authorization_via_inbound_call",
]


def _url_domains(text: str) -> list[str]:
    domains = []
    for match in URL_RE.findall(text):
        url = match if match.lower().startswith("http") else f"http://{match}"
        host = (urlparse(url).hostname or "").lower().removeprefix("www.")
        if host:
            domains.append(host.rstrip("."))
    return domains


def _is_deceptive_host(host: str) -> bool:
    """True when a public-suffix label sits inside the host instead of ending
    it — the domain-in-subdomain trick. The real registrable domain is always
    the last two labels, so anything suffix-shaped before them is decoration."""
    labels = host.split(".")
    return any(label in SUFFIX_LABELS for label in labels[:-2])


def _is_passive_consent_device_change(lower_text: str) -> bool:
    return bool(
        DEVICE_CHANGE_LEX.search(lower_text)
        and PASSIVE_CONSENT_LEX.search(lower_text)
        and not STOP_CUE_LEX.search(lower_text)
    )


def _is_authorization_via_inbound_call(lower_text: str) -> bool:
    return bool(CALL_PRIMING_LEX.search(lower_text) and APPROVAL_LEX.search(lower_text))


def _has_shortcode(text: str) -> bool:
    return any(
        SHORTCODE_CUE_RE.search(text[max(0, m.start() - 24):m.start()])
        for m in SHORTCODE_RE.finditer(text)
    )


def _is_official(host: str) -> bool:
    # exact or subdomain match; substring alone would let ecocash.co.zw.evil.tk pass
    return any(host == d or host.endswith("." + d) for d in OFFICIAL_DOMAINS)


def _requested_in_sentence(lower_text: str, lexicon: re.Pattern, needs_cue: bool) -> bool:
    """True when `words` appears as something the sender is ASKING FOR.

    Sentence-level, not whole-message: a negation (or a cue) elsewhere in a
    long message shouldn't mask — or manufacture — a request in another
    sentence. `needs_cue` is True for noun lexicons, which are neutral on
    their own; False for verb lexicons, which already carry the imperative.
    """
    for sentence in re.split(r"[.!?]", lower_text):
        if not lexicon.search(sentence):
            continue
        if NEGATION_LEX.search(sentence):
            continue
        if needs_cue and not REQUEST_CUE_LEX.search(sentence):
            continue
        return True
    return False


# name -> lexicon, for callers (the explanation layer) that need to tell
# "keyword mentioned" apart from "keyword requested" (requests_* above is
# request-only; a mention with no matching request means it showed up in a
# negated/advisory sentence, e.g. "we will never ask for your password")
MENTION_LEXICONS = {
    "requests_sensitive_credentials": PIN_OTP_LEX,
    "requests_identity_verification": VERIFY_LEX,
    "requests_personal_id": PERSONAL_ID_LEX,
}


def keyword_mentioned(feature_name: str, text: str) -> bool:
    return bool(MENTION_LEXICONS[feature_name].search(text))


def normalize_for_tfidf(text: str) -> str:
    """Replace numeric specifics — URLs, tracking/reference IDs, currency
    amounts, timestamps, phone numbers, shortcodes — with placeholder tokens
    before TF-IDF vectorization. The engineered feature extractor above still
    works on raw text (it needs the real values); this is purely so TF-IDF
    learns the scam *pattern* ("your OTP is <NUM>, click <URL>") instead of
    memorizing incidental digit strings from this synthetic corpus, e.g. "00"
    showing up as a vocabulary token from both "$50.00" and "09:00"."""
    text = URL_RE.sub(" URL_TOKEN ", text)
    text = TRACKING_ID_RE.sub(" TRACKING_ID_TOKEN ", text)
    text = CURRENCY_AMOUNT_RE.sub(" CURRENCY_AMOUNT_TOKEN ", text)
    text = TIME_RE.sub(" TIME_TOKEN ", text)
    text = FULL_NUMBER_RE.sub(" PHONE_TOKEN ", text)
    # not SHORTCODE_TOKEN: this path doesn't check for a contact cue, so a bare
    # 3-5 digit run here is just "a number was present"
    text = SHORTCODE_RE.sub(" NUM_TOKEN ", text)
    return text.lower()


def extract_features(text: str) -> list[float]:
    lower = text.lower()
    has_url = bool(URL_RE.search(text))
    has_shortcode = _has_shortcode(text)
    brand_mentioned = bool(BRAND_LEX.search(lower))
    domains = _url_domains(text)
    unofficial = [d for d in domains if not _is_official(d)]
    return [
        float(has_shortcode),
        float(bool(FULL_NUMBER_RE.search(text))),
        float(has_url),
        float(bool(CURRENCY_RE.search(text))),
        float(len(URGENCY_LEX.findall(lower))),
        float(_requested_in_sentence(lower, PIN_OTP_LEX, needs_cue=True)),
        # brand name mentioned alongside a URL or short callback number is a
        # classic spoofing pattern (official brands don't ask you to click a
        # random link or call a 3-5 digit number)
        float(brand_mentioned and (has_url or has_shortcode)),
        float(any(d in SHORTENER_DOMAINS for d in domains)),
        float(bool(unofficial)),
        # brand name embedded in a domain we don't own = lookalike (ecocash-verify.tk)
        float(any(brand in d for brand in BRAND_NAMES for d in unofficial)),
        float(_requested_in_sentence(lower, VERIFY_VERB_LEX, needs_cue=False)
              or _requested_in_sentence(lower, VERIFY_NOUN_LEX, needs_cue=True)),
        float(_requested_in_sentence(lower, PERSONAL_ID_LEX, needs_cue=True)),
        float(any(_is_deceptive_host(d) for d in unofficial)),
        float(_is_passive_consent_device_change(lower)),
        float(_is_authorization_via_inbound_call(lower)),
    ]


class SmishingFeatureExtractor(BaseEstimator, TransformerMixin):
    """scikit-learn-compatible transformer producing FEATURE_NAMES from raw
    SMS text, for use in a FeatureUnion alongside TF-IDF vectorizers."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([extract_features(text) for text in X])

    def get_feature_names_out(self, input_features=None):
        return np.array(FEATURE_NAMES)
