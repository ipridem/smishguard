"""Template-based, ground-truth-labeled SMS corpus generator for the
smishing classifier: legit messages vs 5 fraud categories, with Shona/English
and Ndebele/English code-switched variants. Deterministic given --seed.

Usage:
    .venv/Scripts/python scripts/generate_sms.py --seed 42
Requires `alembic upgrade head` already run against the target DATABASE_URL.
"""
import argparse
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.config import Config
from app.extensions import db_session, init_db
from app.models.sms import SmsLabel, SmsMessage

BILLS = ["ZESA", "DStv", "municipal rates", "water bill", "school fees"]
FAKE_TLDS = ["tk", "cf", "ga", "xyz"]
FAKE_DOMAINS = ["ecocash-verify", "mobile-secure", "wallet-confirm", "moneyalert"]
# real shortener hosts with random (non-resolving) paths, a staple of delivery
# smishing — matches features.SHORTENER_DOMAINS so the feature gets training signal
FAKE_SHORTENERS = ["bit.ly", "tinyurl.com", "cutt.ly"]
COURIERS = ["ZIMPOST", "DHL", "FedEx", "Courier Connect", "Swift Couriers"]

FIRST_NAMES = [
    "Tendai", "Chipo", "Farai", "Rutendo", "Tapiwa", "Kudzai", "Nyasha", "Tafadzwa",
    "Blessing", "Anesu", "Simbarashe", "Vimbai", "Takudzwa", "Rumbidzai", "Munashe",
    "Sipho", "Nomsa", "Thabo", "Nkosana", "Precious",
]
LAST_NAMES = [
    "Moyo", "Ncube", "Dube", "Sibanda", "Chikwanha", "Mutasa", "Chirwa", "Gumbo",
    "Mupfumira", "Chibwe", "Zulu", "Mhlanga", "Chigumba", "Marufu", "Muzenda",
]


def random_name(rng: random.Random) -> str:
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"

# Templates use str.format() placeholders; unused keys in the values dict
# are simply ignored by format(), so one placeholder-value generator covers
# every template regardless of which subset of keys it references.
TEMPLATES: dict[SmsLabel, list[str]] = {
    SmsLabel.LEGIT: [
        "Confirmed. You have received ${amount} from {name}. New balance ${balance}. TxID:{ref}.",
        "Confirmed. You have paid ${amount} to {name} for {bill}. Balance ${balance}. TxID:{ref}.",
        "Your OTP is {otp}. Do not share this code with anyone. Valid for 5 minutes.",
        "Cashback! Get 5% back on airtime top-ups this week. Dial *151# to top up now.",
        # genuine airtime transfer to a known recipient — no amplification
        # promise, the mirror image of the advance-fee USSD scam below
        "Send ${amount} airtime to your family on {shortphone} using *151*1*1#. Standard network rates apply.",
        "Dear customer, your agent float request of ${amount} has been approved. Ref:{ref}.",
        "Zita renyu {name}, mari yenyu ye ${amount} yatumirwa kuAgent {agent}. TxID:{ref}.",
        "Statement: {count} transactions this month totalling ${amount}. Dial *151*2# for details.",
        "Agent {agent} confirms cash deposit of ${amount} into your wallet. Balance ${balance}.",
        "Reminder: your bill of ${amount} for {bill} is due tomorrow. Pay via *151#.",
        "Mutengo we{bill} ${amount} unofanira kubhadharwa mangwana. Shandisai *151# kubhadhara.",
        "Kuqinisekisiwe. Uthole ${amount} ku{name}. Ibhalansi entsha ${balance}. TxID:{ref}.",
        "Isikhumbuzo: i-bill yakho ye{bill} ${amount} izafuneka kusasa. Bhadala nge *151#.",
        "{courier}: your parcel {ref} has arrived at the depot. Bring your ID to collect during working hours.",
        "{courier}: parcel {ref} out for delivery today. The driver will call you on arrival.",
        "Download the new EcoCash app at https://www.ecocash.co.zw/app for faster payments.",
        "Check your OneMoney balance anytime at https://onemoney.co.zw or dial *111#.",
        "{bank}: We noticed a recent login attempt. This is only for your protection. Our team will never ask for your password by SMS. If this was not you, call customer support using the number on your card.",
        "{bank} security notice: we will never ask you to share your PIN or OTP. If someone asks, do not respond and report it to our branch.",
        "Reminder from {bank}: we will never call or SMS asking for your PIN, password, or OTP. Keep it secret always.",
        # account-activity notifications: state a fact about a credential
        # rather than asking for one — the pattern that reads as phishing to a
        # bag-of-words model but is the single most common legitimate alert
        "{saas}: Your password was successfully changed on {date} at {time}. If you did not perform this action, contact your IT Help Desk immediately.",
        "{saas}: New sign-in to your account from {city}. If this was you, no action is needed.",
        "{bank}: Your PIN was reset successfully at {time}. Call the number on your card if this was not you.",
        "{saas}: Two-factor authentication was enabled on your account on {date}. No action is needed.",
        # legit messages linking to a domain OUTSIDE the hand-curated official
        # allowlist. Without these, has_unofficial_url has zero legit support
        # in training and the model learns "not on my personal brand list" as
        # a phishing proxy — which flags every real company we didn't think to
        # enumerate (Microsoft, GitHub, Slack, any Zim business not in scope).
        # No other risk signal fires here on purpose: unofficial-domain alone
        # must not carry the weight; deceptive_subdomain/shortener/lookalike
        # are what should actually signal an attack.
        "{saas}: A sign-in request was made from a new device. If this was not you, review activity at {saas_url}",
        "{saas}: your workspace settings were updated on {date}. Review the change at {saas_url}",
        "{courier}: track your parcel {ref} at {saas_url}",
        "Your {bank} e-statement for {date} is ready. View it at {saas_url}",
        # correctly-designed device-change alerts: the fraud branch is ACTIVE —
        # you must call/report to STOP a request you didn't make. Same scenario
        # as the passive-consent scam above, opposite (safe) control design.
        "{brand}: A request to move your mobile banking to a new device was received. If this was NOT you, call {shortphone} immediately to block it.",
        "{network}: SIM swap requested on your number. If you did not request this, contact customer care now to cancel it before it completes.",
        # safe device-linking: the not-you branch is ACTIVE (call to cancel),
        # not silence-by-default — the mirror of the manufactured-trigger scam
        "{brand}: A device-linking session was opened using your number. If this was not you, call {shortphone} now to cancel it before it completes.",
        # a technician call NOT coupled with an approve/read instruction — so
        # authorization_via_inbound_call must stay silent (the discriminator is
        # the call+approve conjunction, not the word "technician")
        "{network}: our technician will call {date} to schedule your fibre installation. No action needed now.",
        "{bank}: your appointment with an advisor is confirmed for {date}. We will call to confirm the time.",
        # correctly-designed merchant-payment authorization: PIN entry is
        # gated on the user's OWN prior action, receiving money is explicitly
        # exempted, and a safe cancel path is offered — the mirror image of
        # the reversal-scam PIN demand above.
        "{brand}: A merchant-payment request for ${amount} was sent to your wallet. Enter your PIN only if you intended to pay that merchant. Receiving a cash-in does not require approval. Cancel the request if you do not recognise it.",
        "{brand}: Payment request of ${amount} received. This needs your PIN only if you initiated it. Incoming funds never require your PIN. Decline it if unfamiliar.",
        # correctly-designed beneficiary/payee check: verification happens
        # INSIDE the real app (nothing sent to the sender), gated on a
        # reject-on-mismatch default — the mirror image of the payee-swap scam.
        "{brand}: A beneficiary update was requested on your profile. Verify the beneficiary name and account suffix inside the {brand} app before approving it. Reject the update if any detail differs from the payee you intended to add.",
        "{brand}: A new payee was added to your account. Check the name and account suffix inside the app before approving. If anything looks wrong, reject it and contact support.",
    ],
    SmsLabel.PHISHING_CREDENTIAL: [
        "URGENT: Your account will be suspended. Verify your PIN now at {url} to avoid suspension.",
        "Security Alert: unusual activity detected. Confirm your PIN at {url} within 24 hours.",
        "Dear customer, your wallet is locked. Reply with your 4-digit PIN to unlock immediately.",
        "System upgrade requires PIN verification. Enter your PIN at {url} or lose access today.",
        "Ndapota confirmai PIN yenyu pa {url} kuti musarasikirwe ne account yenyu nhasi.",
        "Your OTP {otp} was requested. If this was not you, reply with your PIN to cancel now.",
        "Account verification required. Send your PIN and last transaction amount to {shortphone}.",
        "We detected login from a new device. Confirm PIN at {url} immediately to secure account.",
        "I-akhawunti yakho izavalwa lamuhla. Thumela i-PIN yakho ku {shortphone} masinyane.",
        "{courier}: Your package cannot be delivered due to an incomplete address. Update your info at {shorturl} to schedule redelivery.",
        "{courier} notice: parcel {ref} is held pending a ${amount} customs fee. Pay now at {shorturl} to avoid return.",
        "Delivery failed. Confirm your address and details at {shorturl} within 24 hours or your parcel will be returned.",
        "{courier}: parcel yenyu {ref} yakamira. Confirmai address yenyu pa {shorturl} nhasi kuti igoendeswa.",
        "Your mobile service has been upgraded. Reply with your national ID number to keep your line active.",
        "POTRAZ SIM registration notice: your line will be deactivated. Reply with your ID number within 24 hours.",
        "{network} network alert: complete your SIM registration now by replying with your identity number or lose service today.",
        # domain-in-subdomain phishing: calm, no urgency, no PIN request — the
        # deceptive host is the only real tell, which is the point
        "{brand}: A transfer of ${amount} to {name} is awaiting confirmation. If you did not initiate it, cancel through your account portal: {lookalike}",
        "{brand}: your statement is ready. Sign in to review it at {lookalike}",
        "{brand} security check: a new device was linked to your account. Review or remove it here: {lookalike}",
    ],
    SmsLabel.ACCOUNT_TAKEOVER: [
        # SIM-swap social engineering: no PIN, no link, no currency ask — the
        # only tell is that the fraud branch is passive. A real anti-fraud
        # control requires ACTION to stop a request you didn't make; this one
        # requires action only to confirm one you did, so doing nothing lets
        # a swap you never asked for go through. The goal is account control,
        # not a credential, hence its own label rather than phishing_credential.
        "{brand}: We received your request to move mobile banking to a replacement handset. The request expires today. If this was you, reply YES; otherwise no response is needed.",
        "{network}: A request to link a new SIM to your number was received. Reply CONFIRM if this was you. No action is needed if not.",
        "{brand}: your line is being ported to a new device. If this was you, reply YES to proceed. No further action required otherwise.",
        "{brand}: a replacement handset has been requested for your mobile banking. Reply YES within 2 hours to approve. Ignore this message if it was not you.",
        "{network}: SIM change pending for your number. To approve, reply APPROVE. No reply is needed otherwise.",
        # MFA-push / consent-phishing: primes an inbound call, then tells the
        # victim to approve a prompt or read a code "during the call". The
        # "no PIN/code will be requested" is disarming cover, not reassurance.
        "{brand}: Support case {ref} opened for your handset migration. A technician will call shortly. During the call, approve the in-app notification labelled Device recovery to complete the security check. No PIN or verification code will be requested.",
        "{brand}: our agent will call you now. When prompted, tap Approve on the security notification to finish verifying your device. We will not ask for your PIN.",
        "{network}: a technician is calling about your account recovery. Please approve the push notification you receive during the call. No code will be requested.",
        "{brand}: mumiriri wedu achakufona. Paunoona notification, dzvanya Approve kuti tipedze security check. Hatikumbire PIN.",
        # same passive-consent device-change trick, no call this time — the
        # "screen"/"notice" is an in-app prompt the victim is coached to accept
        "{brand}: The handset replacement recorded on your profile is awaiting a routine ownership check. When the security notice appears, choose Recognise device to preserve your current wallet limits. You may disregard this message once the notice clears.",
        "{brand}: Your handset transfer has reached the final validation stage. When the Device Protection screen appears, select This was me so the existing profile remains active. If no screen appears, no action is needed.",
        # session-hijack via a real security alert: the alert that would
        # normally warn the victim about the attacker's new device is
        # pre-excused as routine "synchronisation" instead of what it is
        "{brand}: Your replacement SIM is now paired with mobile banking. To keep access on this handset, open the next security alert and select Keep current session. The notice may refer to ending access on another device; this is part of the synchronisation.",
        "{network}: your SIM swap is syncing with your wallet. Open the account notice and choose Keep this device active. It may mention removing another device — that is expected during the sync.",
        # beneficiary/payee-swap: the same pre-excuse mechanism, no "screen"/
        # "alert" vocabulary at all this time — a quoted dialog label plus
        # "appears" stands in for it, and "being aligned" stands in for
        # "synchronisation". Goal is redirecting the victim's own payments.
        "{brand}: The beneficiary update you started has been reconciled with your saved contacts. When Confirm existing payee appears, select CONFIRM to retain scheduled-payment protection. A different account suffix may appear while records are being aligned.",
        "{brand}: Your saved payee list has been reviewed. When the Confirm payee details prompt shows, tap CONFIRM to keep automatic payments active. Some fields may read differently as your contacts are reconciled.",
        # manufactured-trigger evasion: mimics the SAFE conditional-self-check
        # pattern ("if you did this, tap Continue") but the attacker fabricated
        # the trigger — and "linking" evades the replace/transfer/migrate/swap
        # vocabulary the device-change detector otherwise relies on.
        "{brand}: A device-linking session was opened using your number. If you initiated this from a new phone, tap Continue when the prompt appears to finish linking. If you did not initiate this, no action is needed — the session will expire automatically.",
        "{network}: A SIM-linking request was raised on your line. If this was you setting up a new phone, tap Continue to finish linking. If not, no action is needed — unrecognised requests expire on their own.",
    ],
    SmsLabel.PHISHING_REVERSAL_SCAM: [
        "Sorry, I sent ${amount} to you by mistake. Please reverse it to {shortphone} urgently.",
        "System error: ${amount} was sent to your account in error. Refund via *151*{code}# now.",
        "Ndaita mistake, ndakutumira ${amount} zvisiri izvo. Dzoserai kuno {shortphone} ndapota.",
        "Your recent transfer of ${amount} was reversed due to error. Confirm at {url} today.",
        "Agent error: excess ${amount} credited to your wallet. Send back to {shortphone} today.",
        "Reversal pending: click {url} to accept the ${amount} refund to your account now.",
        "I am your cousin {name}, sent money to wrong number, please send ${amount} back urgently.",
        "Ngenze iphutha, ngithumele ${amount} kuwe. Ngicela uyibuyisele ku {shortphone} khathesi.",
        # reversal pretext + credential demand. Every other reversal template
        # asks for money back; none asked for a PIN, which left the two
        # mechanisms artificially disjoint — this hybrid is the pattern that
        # actually dominates real EcoCash fraud.
        "{brand}: A reversal request for ${amount} is pending on your wallet. If you did not request it, reply CANCEL followed by your 4-digit PIN to stop the transaction.",
        "{brand}: reversal of ${amount} pending on your account. Confirm your PIN to cancel it before it completes.",
        "Alert: ${amount} reversal in progress. To stop it, reply with your PIN now.",
        "{brand}: kudzoserwa kwe ${amount} kuri kuitika. Tumirai PIN yenyu kuti mumise izvi.",
        # push-payment authorization hijack: a real refund just lands, it never
        # routes through an accept/release prompt. The pre-excuse for the
        # screen not matching the story is the tell — no genuine notification
        # ever needs to explain away its own confirmation screen looking wrong.
        "{brand}: A ${amount} refund is being returned to your wallet. When the payment request appears in the app, select Accept to release the funds. The screen may display the merchant name rather than refund.",
        "{brand}: your ${amount} reversal is ready. Accept the request in the app to receive it — it may show as a payment to a merchant instead of a refund, that is normal.",
    ],
    SmsLabel.FAKE_AGENT: [
        "This is Agent {agent} from EcoCash. Send your PIN to process your cash-out of ${amount}.",
        "Agent line: dial {shortphone} to complete your cash-in of ${amount} today only.",
        "Muagent wenyu {agent} ari kukumbira PIN yenyu kuti akubatsirei nemari yenyu nhasi.",
        "Official EcoCash Agent {agent}: confirm your PIN at {shortphone} to receive ${amount}.",
        "New agent onboarding: verify your account by sending PIN to {shortphone} before {time}.",
        "Agent {agent} needs your PIN to reverse the failed transaction of ${amount} urgently.",
        # vishing setup: the SMS never asks for the code, it primes the victim
        # to read it aloud to the caller. Theft happens on the phone call.
        "{brand} Support: We are stopping the ${amount} cash-out now. Read the verification code sent to your phone when our agent calls you.",
        "{brand}: our agent will call you in 5 minutes. Please repeat the security code we sent so we can cancel the ${amount} transaction.",
        "Support ticket {ref} opened. Our officer is calling now — tell them the one-time code to verify your identity.",
        "{brand}: mumiriri wedu achakufonerai. Verengai code yatumirwa pafoni yenyu kuti timise ${amount}.",
    ],
    SmsLabel.PRIZE_SCAM: [
        "Congratulations! Your number has won ${amount} in the EcoCash promo. Claim at {url}.",
        "You have been selected to win a car! Send your PIN to {shortphone} to claim your prize.",
        "Makakunda mubairo we ${amount}! Tumirai PIN yenyu pa {shortphone} kuti mugamuchire.",
        "WINNER! Your SIM has been randomly selected for ${amount} bonus. Claim now at {url}.",
        "Free airtime worth ${amount} waiting for you. Reply YES and send your PIN to claim.",
        "Lucky draw winner! Confirm your details at {url} to receive your ${amount} prize today.",
        "Uwinile! Inombolo yakho iwine ${amount}. Thumela i-PIN yakho ku {shortphone} lamuhla.",
        # advance-fee airtime/USSD scam: send a small amount, promised a much
        # bigger amount back. The USSD code isn't the tell — legit airtime/
        # bill templates use *151# constantly too — it's the code paired
        # with a promised amplified return.
        "{network}: Send $2 airtime to {shortphone} using *151*1*1# and receive $10 airtime back instantly! Limited offer today.",
        "Congratulations! Dial *151*4*{shortphone}*5# to send a small activation fee in airtime and claim your ${amount} {brand} bonus.",
        "{network}: Double your airtime! Transfer any amount to {shortphone} via *145# and get double back within 5 minutes.",
        "Chikwata che{network}: Tumirai airtime ku {shortphone} nge*151*1*1# mugogashira zvakapetwa kaviri nekukurumidza.",
    ],
    SmsLabel.OTHER_FRAUD: [
        "For help with your blocked account, call our customer care on {shortphone} now.",
        "EcoCash support desk: {shortphone}. Call for urgent assistance with your ${amount} issue.",
        "Kana muchida rubatsiro fonerai {shortphone}, ndiwo customer care chaiwo weEcoCash.",
        "Your complaint ref {ref} requires you to call {shortphone} within 1 hour or it expires.",
        "Fake support: We noticed a failed transaction. Call {shortphone} to resolve immediately.",
        "Helpdesk notice: verify your identity by calling {shortphone} to unlock your wallet.",
    ],
}


def random_shortphone(rng: random.Random) -> str:
    return str(rng.randint(100, 99999))


def random_url(rng: random.Random) -> str:
    domain = rng.choice(FAKE_DOMAINS)
    tld = rng.choice(FAKE_TLDS)
    token = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=6))
    return f"http://{domain}.{tld}/{token}"


def lookalike_url(rng: random.Random) -> str:
    """Domain-in-subdomain trick: the real registrable domain is the attacker's,
    but an official-looking one is worn in front of it as a subdomain."""
    worn = rng.choice([
        "ecocash.co.zw", "onemoney.co.zw", "zimpay.co.zw", "stewardbank.co.zw",
        "cbz.co.zw", "netone.co.zw", "zimpost.co.zw", "cabs.co.zw",
    ])
    attacker = rng.choice([
        "account-review.com", "secure-login.net", "client-verify.com",
        "portal-access.org", "session-check.net",
    ])
    return f"https://secure.{worn}.{attacker}"


def _template_values(rng: random.Random) -> dict:
    amount = rng.uniform(5, 800)
    return {
        "amount": f"{amount:.2f}",
        "balance": f"{rng.uniform(amount, amount + 2000):.2f}",
        "name": random_name(rng),
        "agent": f"{rng.choice(['Harare', 'Bulawayo', 'Mutare', 'Gweru'])}{rng.randint(1, 99)}",
        "ref": f"TX{rng.randint(100000, 999999)}",
        "otp": f"{rng.randint(100000, 999999)}",
        "url": random_url(rng),
        "lookalike": lookalike_url(rng),
        "shorturl": f"http://{rng.choice(FAKE_SHORTENERS)}/{''.join(rng.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=7))}",
        "courier": rng.choice(COURIERS),
        "shortphone": random_shortphone(rng),
        "bill": rng.choice(BILLS),
        "count": rng.randint(3, 40),
        "code": f"{rng.randint(1000, 9999)}",
        "time": f"{rng.randint(1, 11)}{rng.choice(['am', 'pm'])}",
        "bank": rng.choice(["Steward Bank", "CBZ", "CABS", "FBC", "NMB", "ZB Bank"]),
        "network": rng.choice(["Econet", "NetOne", "Telecel"]),
        "saas": rng.choice(["Microsoft 365", "Google Workspace", "Zoom", "Slack"]),
        "saas_url": rng.choice([
            "https://account.microsoft.com/security", "https://myaccount.google.com/security",
            "https://slack.com/security", "https://github.com/settings/security",
            "https://www.dhl.com/track", "https://www.fedex.com/track",
        ]),
        "brand": rng.choice(["ZimPay", "EcoCash", "OneMoney", "Steward Bank", "CBZ", "CABS"]),
        "city": rng.choice(["Harare", "Bulawayo", "Gweru", "Mutare", "Masvingo"]),
        "date": f"{rng.randint(1, 28)} {rng.choice(['Jan', 'Mar', 'Jun', 'Jul', 'Sep', 'Nov'])} {rng.randint(2024, 2026)}",
    }


# Interchangeable phrasings for the same underlying concept — drawn directly
# from real evasions caught during development (word-order swaps, vocabulary
# substitutions that slipped past a fixed feature lexicon: "replacement
# handset" -> "handset transfer" -> "handset replacement"; "screen"/"prompt"
# -> "security alert"/"notice"; "may display X rather than Y" -> "may refer
# to X; this is part of the synchronisation"). Rewriting a template through
# a random pick from each group broadens lexical coverage per attack IDEA
# instead of training on one fixed phrasing per idea — the same principle as
# `obfuscate()` below, but at the phrase level instead of the character level.
PARAPHRASE_GROUPS: list[list[str]] = [
    ["no action is needed", "no action needed", "no further action required",
     "you may disregard this message", "ignore this notice", "no response is needed"],
    ["replacement handset", "handset replacement", "handset transfer",
     "device replacement", "device transfer", "handset migration",
     "device-linking session", "linking session", "device linking"],
    ["screen", "prompt", "notification", "security alert", "notice", "dialog"],
    ["rather than", "instead of", "as opposed to"],
    ["approve", "accept", "confirm", "authorise", "continue"],
    ["may display", "may show", "may appear as", "might display", "could show"],
    ["this is part of the synchronisation", "this is part of the process",
     "this is part of the reconciliation", "this is expected during the update"],
    ["technician will call", "agent will call", "officer will call",
     "representative will call", "during our courtesy call", "during the courtesy call"],
    ["read the code", "read out the code", "tell them the code", "share the code"],
    ["only if you intended", "only if you initiated", "only if this was you"],
    ["reject the update if", "decline it if", "cancel it if"],
    ["being aligned", "being reconciled", "being synchronised", "being updated"],
    # from the LLM-caught "courtesy call" evasion and the "device-linking"
    # near-miss (60.02% risk purely by TF-IDF luck, structural detectors silent)
    ["finish linking", "complete the linking", "finish the linking process"],
    ["no action is needed — the session will expire automatically",
     "no action is needed — unrecognised requests expire on their own",
     "no action is needed — the request will lapse by itself"],
]
_PARAPHRASE_PATTERNS = [
    (re.compile(rf"\b(?:{'|'.join(re.escape(p) for p in sorted(group, key=len, reverse=True))})\b", re.IGNORECASE), group)
    for group in PARAPHRASE_GROUPS
]


def paraphrase_text(text: str, rng: random.Random) -> str:
    """Swap each matched phrase for a random ALTERNATE from its group.
    Case-insensitive; alternates are inserted lowercase, which reads fine
    mid-sentence in SMS copy (existing templates are inconsistently cased too)."""
    for pattern, group in _PARAPHRASE_PATTERNS:
        def _swap(m, group=group):
            alternatives = [p for p in group if p.lower() != m.group(0).lower()]
            return rng.choice(alternatives) if alternatives else m.group(0)
        text = pattern.sub(_swap, text)
    return text


def gen_sms_messages(rng: random.Random, count: int, paraphrase_rate: float = 0.35) -> list[SmsMessage]:
    labels = list(TEMPLATES.keys())
    messages = []
    for _ in range(count):
        label = rng.choice(labels)
        template = rng.choice(TEMPLATES[label])
        text = template.format(**_template_values(rng))
        if rng.random() < paraphrase_rate:
            text = paraphrase_text(text, rng)
        messages.append(SmsMessage(text=text, label=label, source="generator"))
    return messages


def _write_dataset_card(path: Path, counts: Counter, seed: int, paraphrase_rate: float) -> None:
    lines = [
        "# Smishing SMS Dataset Card",
        "",
        f"Generated with seed {seed} by `scripts/generate_sms.py` "
        f"(paraphrase rate: {paraphrase_rate:.0%}).",
        "",
        "## Composition",
        "",
        "| Label | Count |",
        "|---|---|",
    ]
    for label, n in sorted(counts.items()):
        lines.append(f"| {label} | {n} |")
    lines += [
        "",
        "## Generation method",
        "",
        "Hand-written templates per label, parameterized with randomized names, "
        "amounts (USD/ZWG), shortcodes, and fake URLs. A subset of templates use "
        "Shona/English and Ndebele/English code-switching, reflecting real "
        "Zimbabwean mobile-money SMS traffic.",
        "",
        f"A random {paraphrase_rate:.0%} of rows are additionally rewritten through "
        "`PARAPHRASE_GROUPS` — interchangeable phrasings (word-order swaps, vocabulary "
        "substitutions) drawn directly from real evasions found during development, "
        "so training coverage per attack idea isn't limited to one fixed phrasing.",
        "",
        "## Limitations",
        "",
        "- Fully synthetic — no real personal data or real scam infrastructure "
        "(fake URLs use non-resolving placeholder domains or random shortener "
        "paths that resolve to nothing).",
        "- Roughly class-balanced by construction; does not reflect true "
        "real-world label prevalence (legitimate SMS vastly outnumber scams "
        "in practice).",
        "- Template-based generation caps lexical diversity relative to a "
        "purely human-authored corpus; adversarial (obfuscated) variants are "
        "generated separately at training/evaluation time, not stored here.",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--reset", action="store_true", help="Delete existing generator-sourced sms_messages first.")
    parser.add_argument("--dataset-card-out", default="data/sms_dataset_card.md")
    parser.add_argument("--paraphrase-rate", type=float, default=0.35,
                         help="fraction of rows rewritten via PARAPHRASE_GROUPS (0 disables).")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    init_db(Config.DATABASE_URL)

    if args.reset:
        db_session.query(SmsMessage).filter(SmsMessage.source == "generator").delete()
        db_session.commit()

    messages = gen_sms_messages(rng, args.count, paraphrase_rate=args.paraphrase_rate)
    db_session.add_all(messages)
    db_session.commit()

    counts = Counter(m.label.value for m in messages)
    out_path = Path(args.dataset_card_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_dataset_card(out_path, counts, args.seed, args.paraphrase_rate)

    print(f"generated {len(messages)} sms messages -> dataset card: {out_path}")
    for label, n in sorted(counts.items()):
        print(f"  {label}: {n}")


if __name__ == "__main__":
    main()
