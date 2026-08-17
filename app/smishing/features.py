"""Engineered SMS features, complementing TF-IDF word/char n-grams: shortcode
vs full-number presence, URLs, currency amounts, urgency lexicon, PIN/OTP
requests, and a brand-spoofing heuristic."""
import re
from urllib.parse import urlparse

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from app.smishing.normalize import canonicalise, has_transaction_reference, ussd_embeds_msisdn
from app.smishing.psl_suffixes import PUBLIC_SUFFIXES

# Scheme-less bare domains ("ecocash-help.net/unlock") have no "https://" or
# "www." for the old URL_RE to anchor on, so five URL-derived features silently
# zeroed on them. Anchor the scheme-less alternative on a real public suffix
# instead of "any dot" — otherwise "arrive 15 min early.Please" manufactures a
# URL out of a missing space. Suffixes sorted longest-first so a compound
# suffix like "co.zw" wins over a bare "zw" it also contains.
_SUFFIX_ALT = "|".join(
    re.escape(s) for s in sorted(PUBLIC_SUFFIXES, key=len, reverse=True)
)
_DOMAIN_LABEL = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
# negative lookbehind excludes "@" so an email address's domain
# ("payroll@company.co.zw") isn't mistaken for a link
BARE_DOMAIN_RE = re.compile(
    rf"(?<![\w.@-])(?:{_DOMAIN_LABEL}\.)+(?:{_SUFFIX_ALT})(?![a-zA-Z0-9-])(?:/\S*)?",
    re.IGNORECASE,
)
URL_RE = re.compile(
    rf"https?://\S+|www\.\S+|{BARE_DOMAIN_RE.pattern}", re.IGNORECASE
)
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
NEGATION_WORDS = [
    "never", "won't", "will not", "don't", "do not", "cannot", "can't", "not ask",
    # a genuine merchant-payment authorization is gated on the user's OWN
    # prior action ("enter your PIN only if YOU intended to pay") — the
    # opposite shape of a scam, which asks unconditionally or gates on the
    # NEGATIVE case ("if this was NOT you, send your PIN"). Deliberately
    # narrow to self-referential confirmation phrasing, not a bare "if",
    # which a scam could otherwise wrap around any demand to fake safety.
    "only if you intended", "if you intended to", "if you initiated",
    "if this was you", "if you made this request", "if you requested this",
    "does not require approval", "does not require your pin",
    # a genuine beneficiary/payee check happens INSIDE the real app, with an
    # explicit reject-on-mismatch default — "verify X inside the app before
    # approving it" is the user checking their own screen, not disclosing
    # anything to the sender. That's structurally different from "verify your
    # account [by clicking/replying/calling]", which hands control to whoever
    # sent the message.
    "before approving", "before you approve", "reject the update if",
    "reject if any detail",
]
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
    # pan-African mobile-money/telco brands -- the imported hf_african_smishing
    # corpus (59% of all training rows) is dense with these; without them
    # brand_lookalike_domain and brand_with_*_channel were blind to most of
    # the real (non-synthetic) training data's brand-impersonation examples,
    # which is why they stayed underpowered despite being structurally sound
    "m-pesa", "mpesa", "mtn", "safaricom", "airtel", "orange", "vodacom",
]
# real institutions never link anywhere else in SMS; anything off-list is suspect
OFFICIAL_DOMAINS = {
    "ecocash.co.zw", "econet.co.zw", "onemoney.co.zw", "netone.co.zw",
    "innbucks.co.zw", "telecel.co.zw", "mukuru.com", "cbz.co.zw",
    "stewardbank.co.zw", "zimpost.co.zw",
    # regulator/utility/insurer domains missed above -- e.g. ZIMRA's
    # efiling.zimra.co.zw scored as an unofficial domain purely because
    # nobody had listed it
    "zimra.co.zw", "oldmutual.co.zw", "dhl.com",
}
# .gov.zw and .ac.zw are controlled delegations (state / Ministry of Higher
# Education respectively) -- unlike a generic .com an attacker can freely
# register a lookalike under, nobody outside the real institution can obtain
# a host here. Checked as a suffix so every university/ministry is covered
# without hand-listing each one (uz.ac.zw, msu.ac.zw, ... would otherwise
# need enumerating one at a time, same failure mode as the missing entries
# above).
OFFICIAL_SUFFIXES = (".gov.zw", ".ac.zw")
# Public-suffix labels. If one shows up in the INTERIOR of a hostname rather
# than at its end, the host is wearing another domain as a costume:
# secure.zimpay.co.zw.account-review.com registers as account-review.com but
# reads as zimpay.co.zw at a glance. Brand-agnostic on purpose — BRAND_NAMES
# only catches impersonations of brands we happened to list, which is exactly
# how a lookalike for an unlisted brand slips through.
SUFFIX_LABELS = frozenset(
    label for suffix in PUBLIC_SUFFIXES for label in suffix.split(".")
)
# A real anti-fraud control on an account-changing request (SIM swap, device
# link, mobile-banking transfer) requires ACTION to stop a request you didn't
# make. A message that instead makes silence the default for the fraud branch
# — "reply to CONFIRM, do nothing if it wasn't you" — inverts that: doing
# nothing lets the change proceed. This is a documented SIM-swap social-
# engineering pattern; it carries no PIN/URL/currency ask, so every other
# signal in this file is blind to it.
DEVICE_CHANGE_PHRASES = [
    "new device", "new sim", "sim swap", "sim card",
    "move mobile banking", "move your mobile banking", "change your registered number",
    "link a new device", "port your number",
    # compound-phrase forms of "linking" — deliberately NOT a bare "link"
    # verb stem in the proximity regex below, which collides with the
    # ordinary "click this link" (a URL) and produced real false positives
    "device-linking", "device linking", "linking session", "linking your sim",
    "linking your device", "finish linking", "device link",
]
# a literal phrase list loses this: "replacement handset" -> "handset transfer"
# -> "handset replacement" is the same two words in three orders. Proximity,
# not phrase, is what's stable — a change-noun and a change-verb within a few
# words of each other, in either order.
_DEVICE_CHANGE_NOUNS = r"handset|device|sim|profile|line"
# NOT "link\w*" here: unlike replace/transfer/migrate/swap/port, "link" collides
# with the ordinary "click this link" (a URL), which produced real false
# positives in testing ("update your device settings... this link... no action
# needed"). The linking-specific evasion is instead covered by exact compound
# phrases below (DEVICE_CHANGE_PHRASES), which don't fire on a bare URL mention.
_DEVICE_CHANGE_VERBS = r"replac\w*|transfer\w*|migrat\w*|swap\w*|port\w*"
DEVICE_CHANGE_PROXIMITY_RE = re.compile(
    rf"\b(?:{_DEVICE_CHANGE_NOUNS})\b(?:\W+\w+){{0,3}}\W+\b(?:{_DEVICE_CHANGE_VERBS})\b"
    rf"|\b(?:{_DEVICE_CHANGE_VERBS})\b(?:\W+\w+){{0,3}}\W+\b(?:{_DEVICE_CHANGE_NOUNS})\b",
    re.IGNORECASE,
)
PASSIVE_CONSENT_PHRASES = [
    "no response is needed", "no response needed", "no action is needed",
    "no action needed", "no further action", "no need to respond",
    "no need to reply", "nothing further is required",
    # "disregard/ignore this X" is the same silence-is-default framing as
    # "no action needed", just phrased as an instruction rather than a status
    "you may disregard", "disregard this message", "disregard this notice",
    "ignore this message", "ignore this notice",
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
# in-app payment-authorization hijack: a "refund"/"reversal" story sets up an
# in-app prompt the victim is told to accept — but a real mobile-money refund
# never routes through an "accept/release" confirmation at all, it just
# lands. The near-unfakeable tell is the message PRE-EXCUSING a mismatch
# between its own story and what the actual authorization screen will show
# ("may display the merchant name rather than 'refund'") — no legitimate
# notification ever needs to explain away its own prompt looking wrong,
# because a legitimate one doesn't look wrong.
SCREEN_PROMPT_WORDS = [
    "screen", "prompt", "notification", "pop-up", "popup", "in the app",
    "security alert", "alert", "notice",
]
# A quoted button/dialog label immediately followed by "appears" IS a prompt
# reference, regardless of whether the message ever says "screen"/"prompt" —
# this is the second time a scam evaded SCREEN_PROMPT_WORDS by vocabulary
# alone, so this one's structural: quote-mark + short phrase + "appears".
QUOTED_APPEARS_RE = re.compile(r"[\"“‘'][^\"”’']{2,50}[\"”’']\s*appears", re.IGNORECASE)
SCREEN_MISMATCH_CUES = [
    "rather than", "instead of", "even if it shows", "even if it says",
    "regardless of what", "not match", "different from what",
]
# Structural, not phrase-based, for the same reason: "may appear/show/display/
# refer/mention/read/list X" is the mismatch PREDICTION, in any wording.
MISMATCH_PREDICTION_RE = re.compile(
    r"\b(?:may|might|could)\b(?:\s+\w+){0,2}\s+"
    r"(?:appear|show|display|read|list|reflect|refer|mention|say)\b",
    re.IGNORECASE,
)
# "this is normal/expected/part of X" excuses the mismatch instead of naming
# it outright — same pre-excusing move, softer phrasing. "aligned"/"reconciled"
# generalizes "synchronisation" to any word implying background record-matching.
EXPLAIN_AWAY_RE = re.compile(
    r"\b(?:this|that|it)\s+is\s+(?:normal|expected|fine|routine|standard)\b"
    r"|\bpart of (?:the|our|a)\b(?:\s+\w+){0,2}\s+"
    r"(?:synchroni[sz]ation|sync|process|upgrade|update|verification|alignment|reconciliation)\b"
    r"|\b(?:being|are being)\s+(?:aligned|reconciled|synchroni[sz]ed|updated|verified)\b",
    re.IGNORECASE,
)
# Advance-fee airtime/USSD scam: "send airtime via *151*1*1# and get DOUBLE
# back" — a classic advance-fee trick, common across Zimbabwe and the region.
# Legit airtime-transfer templates ("Pay via *151#", "top up via *151#") use
# the same USSD-code shape constantly, so the code alone can't be the signal
# — it's the code PAIRED with a promise of getting back more than you send.
USSD_CODE_RE = re.compile(r"\*\d[\d*]{2,}#")
AMPLIFICATION_WORDS = [
    "double your airtime", "double your money", "get double", "receive double",
    "double back", "double it back", "get more back", "receive more back",
    "x2 your airtime", "instant bonus", "activation fee", "processing fee to claim",
    "small fee to release", "send to receive more",
]
# structural backstop: "receive/get [anything] back" catches phrasings the
# literal list above doesn't ("receive $10 airtime back instantly") without
# needing "double"/"bonus" specifically named. But a legit cashback offer
# ("Get 5% back on airtime top-ups, dial *151#") matches this shape too — the
# scam-specific part is sending the airtime to someone ELSE first, so this
# backstop only counts alongside an explicit send/transfer verb.
AMPLIFICATION_RECEIVE_BACK_RE = re.compile(r"\b(?:receive|get)\b(?:\s+\S+){0,3}\s+back\b", re.IGNORECASE)
SEND_TRANSFER_RE = re.compile(r"\b(?:send|transfer|forward)\b", re.IGNORECASE)
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
AMPLIFICATION_LEX = _lexicon(AMPLIFICATION_WORDS)
SCREEN_PROMPT_LEX = _lexicon(SCREEN_PROMPT_WORDS)
SCREEN_MISMATCH_LEX = _lexicon(SCREEN_MISMATCH_CUES)

FEATURE_NAMES = [
    "has_shortcode", "has_full_number", "has_url", "has_currency_amount",
    "urgency_word_count", "requests_sensitive_credentials",
    "brand_with_trusted_channel", "brand_with_untrusted_channel",
    "has_shortener_url", "has_unofficial_url", "brand_lookalike_domain",
    "requests_identity_verification", "requests_personal_id",
    "deceptive_subdomain", "passive_consent_device_change",
    "authorization_via_inbound_call", "screen_mismatch_coaching",
    "ussd_advance_fee_offer", "ussd_embeds_msisdn", "has_transaction_reference",
]


def _url_domains(text: str) -> list[str]:
    domains = []
    for match in URL_RE.findall(text):
        url = match if match.lower().startswith("http") else f"http://{match}"
        host = (urlparse(url).hostname or "").lower().removeprefix("www.")
        if host:
            domains.append(host.rstrip("."))
    return domains


def _registrable_suffix_length(labels: list[str]) -> int:
    """How many trailing labels form the real public suffix: 2 for a compound
    suffix like "co.zw", else 1. Assuming always-2 (the old behaviour) reads
    a 2-label suffix like "ac.zw" as domain="ac" + tld="zw", so a subdomain
    such as "uz" (University of Zimbabwe, "uz.ac.zw") lands in the "interior"
    slot and collides with "uz" (Uzbekistan's own ccTLD) in SUFFIX_LABELS."""
    if len(labels) >= 2 and ".".join(labels[-2:]) in PUBLIC_SUFFIXES:
        return 2
    return 1


def _is_deceptive_host(host: str) -> bool:
    """True when a public-suffix label sits inside the host instead of ending
    it — the domain-in-subdomain trick. The real registrable domain is the
    trailing suffix plus one label, so anything suffix-shaped before that is
    decoration."""
    labels = host.split(".")
    suffix_len = _registrable_suffix_length(labels)
    interior = labels[: -(suffix_len + 1)] if len(labels) > suffix_len + 1 else []
    return any(label in SUFFIX_LABELS for label in interior)


def _is_passive_consent_device_change(lower_text: str) -> bool:
    return bool(
        (DEVICE_CHANGE_LEX.search(lower_text) or DEVICE_CHANGE_PROXIMITY_RE.search(lower_text))
        and PASSIVE_CONSENT_LEX.search(lower_text)
        and not STOP_CUE_LEX.search(lower_text)
    )


def _is_authorization_via_inbound_call(lower_text: str) -> bool:
    return bool(CALL_PRIMING_LEX.search(lower_text) and APPROVAL_LEX.search(lower_text))


def _is_ussd_advance_fee_offer(text: str, lower_text: str) -> bool:
    if not USSD_CODE_RE.search(text):
        return False
    if AMPLIFICATION_LEX.search(lower_text):
        return True
    # the generic "get/receive ... back" backstop needs a send/transfer verb
    # alongside it — a legit cashback offer says "get back" without ever
    # asking you to send anything to anyone first
    return bool(AMPLIFICATION_RECEIVE_BACK_RE.search(lower_text) and SEND_TRANSFER_RE.search(lower_text))


def _is_screen_mismatch_coaching(lower_text: str) -> bool:
    has_prompt_ref = bool(SCREEN_PROMPT_LEX.search(lower_text) or QUOTED_APPEARS_RE.search(lower_text))
    has_mismatch = bool(
        SCREEN_MISMATCH_LEX.search(lower_text)
        or MISMATCH_PREDICTION_RE.search(lower_text)
        or EXPLAIN_AWAY_RE.search(lower_text)
    )
    return has_prompt_ref and has_mismatch


def _has_shortcode(text: str) -> bool:
    return any(
        SHORTCODE_CUE_RE.search(text[max(0, m.start() - 24):m.start()])
        for m in SHORTCODE_RE.finditer(text)
    )


def _is_official(host: str) -> bool:
    # exact or subdomain match; substring alone would let ecocash.co.zw.evil.tk pass
    if host.endswith(OFFICIAL_SUFFIXES):
        return True
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
    text = canonicalise(text)
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
    text = canonicalise(text)
    lower = text.lower()
    has_url = bool(URL_RE.search(text))
    has_shortcode = _has_shortcode(text)
    brand_mentioned = bool(BRAND_LEX.search(lower))
    domains = _url_domains(text)
    unofficial = [d for d in domains if not _is_official(d)]
    official = [d for d in domains if _is_official(d)]
    has_full_number = bool(FULL_NUMBER_RE.search(text))
    return [
        float(has_shortcode),
        float(has_full_number),
        float(has_url),
        float(bool(CURRENCY_RE.search(text))),
        float(len(URGENCY_LEX.findall(lower))),
        float(_requested_in_sentence(lower, PIN_OTP_LEX, needs_cue=True)),
        # brand + a channel the brand actually controls (its own domain, or a
        # short code) is what real telco/bank notices look like -- split from
        # the untrusted case below because as one boolean they cancelled out
        # and the model learned "brand + channel" as a legitimacy marker
        # (it fires on almost all real telco traffic, since short codes ARE
        # the normal channel)
        float(brand_mentioned and (bool(official) or has_shortcode)),
        # brand + a channel it does NOT control (someone else's domain, or an
        # arbitrary phone number) is the actual spoofing pattern
        float(brand_mentioned and (bool(unofficial) or has_full_number)),
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
        float(_is_screen_mismatch_coaching(lower)),
        float(_is_ussd_advance_fee_offer(text, lower)),
        # a USSD string with a full MSISDN inside it is an ADDRESSED transfer
        # to a third party -- no legit telco template does this ("*171#",
        # "*151*2#" are menu paths, not addressed transfers). Wording-
        # independent, unlike ussd_advance_fee_offer above, which needs an
        # explicit amplification promise and misses plain USSD-PIN scams.
        float(ussd_embeds_msisdn(text)),
        # genuine automated financial SMS carry a machine-minted reference
        # (Ref MP250817.1432.K84210, REV-88214, meter/policy numbers) because
        # the sender has a back end; fraud SMS rarely do. The model's first
        # real pro-legit signal -- has_currency_amount alone (+2.72 toward
        # fraud) currently has nothing opposing it, which is why a plain
        # "send money to this number" message reads as identical to a scam.
        float(has_transaction_reference(text)),
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
