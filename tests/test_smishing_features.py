from app.smishing.features import FEATURE_NAMES, extract_features, normalize_for_tfidf


def _feature_dict(text: str) -> dict:
    return dict(zip(FEATURE_NAMES, extract_features(text)))


def test_url_detected():
    f = _feature_dict("Verify your PIN at http://fake-verify.tk/abc123")
    assert f["has_url"] == 1.0


def test_shortcode_vs_full_number():
    assert _feature_dict("Call our line on 4521 now")["has_shortcode"] == 1.0
    assert _feature_dict("Call our line on 263771234567 now")["has_full_number"] == 1.0


def test_currency_amount_detected():
    assert _feature_dict("You have received $50 from John")["has_currency_amount"] == 1.0
    assert _feature_dict("No amount mentioned here at all")["has_currency_amount"] == 0.0


def test_urgency_word_count():
    f = _feature_dict("URGENT: act now, your account is suspended, expires today only")
    assert f["urgency_word_count"] >= 3


def test_pin_otp_keyword_detected():
    assert _feature_dict("Please verify your PIN now")["requests_sensitive_credentials"] == 1.0
    assert _feature_dict("Your parcel has been delivered")["requests_sensitive_credentials"] == 0.0


def test_verify_keyword_distinct_from_pin_otp():
    verify_only = _feature_dict(
        "Your bank account has been restricted. Verify your identity at http://secure-bank-verify-login.com"
    )
    assert verify_only["requests_identity_verification"] == 1.0
    assert verify_only["requests_sensitive_credentials"] == 0.0

    pin_request = _feature_dict("Reply with your PIN to unlock your wallet")
    assert pin_request["requests_sensitive_credentials"] == 1.0
    assert pin_request["requests_identity_verification"] == 0.0


def test_verification_noun_needs_a_cue_but_verb_stands_alone():
    """'verify' is itself the demand; 'verification' is a noun that shows up in
    ordinary notifications ('your verification code is …')."""
    assert _feature_dict("Verify your account now")["requests_identity_verification"] == 1.0
    assert _feature_dict("Your verification code is 481920")["requests_identity_verification"] == 0.0
    assert _feature_dict("Complete verification to proceed")["requests_identity_verification"] == 1.0


def test_requests_personal_id_distinct_from_in_person_collection():
    scam = _feature_dict("Your mobile service has been upgraded. Reply with your ID number to keep your line active.")
    assert scam["requests_personal_id"] == 1.0

    # a courier asking you to bring your ID in person to collect a parcel is
    # not the same signal as a scam asking you to text your ID number
    legit_pickup = _feature_dict("ZIMPOST: your parcel has arrived at the depot. Bring your ID to collect during working hours.")
    assert legit_pickup["requests_personal_id"] == 0.0


def test_normalize_for_tfidf_replaces_incidental_numbers():
    normalized = normalize_for_tfidf(
        "Meet at 09:00 or 13:00, ref FW238491, pay $50.00 now at http://scam.tk/x, call 4521"
    )
    assert "09:00" not in normalized
    assert "13:00" not in normalized
    assert "fw238491" not in normalized
    assert "50.00" not in normalized
    assert "scam.tk" not in normalized
    assert "time_token" in normalized
    assert "tracking_id_token" in normalized
    assert "currency_amount_token" in normalized
    assert "url_token" in normalized
    assert "num_token" in normalized


def test_credential_mentioned_as_fact_is_not_a_request():
    """A notification states a fact about a credential; a scam asks for one.
    Bag-of-words can't tell them apart on the noun alone — a request cue can."""
    notification = _feature_dict(
        "Microsoft 365: Your password was successfully changed on 29 Jul 2026 at 08:15 UTC. "
        "If you did not perform this action, contact your IT Help Desk immediately."
    )
    otp_delivery = _feature_dict("Your OTP is 275331. Valid for 5 minutes.")
    assert notification["requests_sensitive_credentials"] == 0.0
    assert otp_delivery["requests_sensitive_credentials"] == 0.0

    for demand in [
        "Reply with your PIN to unlock your wallet",
        "Send your PIN to 12345 immediately",
        "Enter your PIN at http://fake-verify.tk/x",
        "Tumirai PIN yenyu pa 8804",
    ]:
        assert _feature_dict(demand)["requests_sensitive_credentials"] == 1.0, demand


def test_passive_consent_device_change_detected():
    """SIM-swap social engineering: an account change that proceeds by default
    unless the victim acts. The tell is the inverted control design, not any
    PIN/URL/currency ask — so it must be caught by SHAPE."""
    scam = _feature_dict(
        "FinGuard: We received your request to move mobile banking to a replacement handset. "
        "The request expires today. If this was you, reply YES; otherwise no response is needed."
    )
    assert scam["passive_consent_device_change"] == 1.0

    # same scenario, safe control design: action required to STOP it → not flagged
    safe = _feature_dict(
        "FinGuard: A request to move your mobile banking to a new device was received. "
        "If this was NOT you, call 0808 immediately to block it."
    )
    assert safe["passive_consent_device_change"] == 0.0

    # passive-consent language with no device change (ordinary notice) → not flagged
    ordinary = _feature_dict("Your statement is ready. No action is needed.")
    assert ordinary["passive_consent_device_change"] == 0.0


def test_passive_consent_device_change_survives_word_order_paraphrase():
    """A literal phrase list loses this: 'replacement handset' -> 'handset
    transfer' -> 'handset replacement' is the same two words in three orders.
    Detection has to be proximity-based, not phrase-based, to survive it."""
    for paraphrase in [
        'FinGuard: Your handset transfer has reached the final validation stage. '
        'When the Device Protection screen appears, select "This was me" so the '
        'existing profile remains active. If no screen appears, no action is needed.',

        'FinGuard: The handset replacement recorded on your profile is awaiting a '
        'routine ownership check. When the security notice appears, choose '
        '"Recognise device" to preserve your current wallet limits. You may '
        'disregard this message once the notice clears.',
    ]:
        assert _feature_dict(paraphrase)["passive_consent_device_change"] == 1.0, paraphrase

    # the noun+verb proximity check must not fire without a passive-consent
    # phrase alongside it — device-change vocabulary alone is not the signal
    for benign in [
        "Your profile picture has been changed successfully.",
        "Your new device has been linked to your WiFi network.",
    ]:
        assert _feature_dict(benign)["passive_consent_device_change"] == 0.0, benign


def test_passive_consent_device_change_covers_linking_vocabulary():
    """'Linking' is a fourth paraphrase of the same device-change concept.
    Scored 0.6002 risk before this fix — a hair over the fraud threshold by
    TF-IDF luck alone, with every structural signal reading 'not found'.
    NOT a bare 'link' verb stem: that collides with the ordinary 'click this
    link' (a URL reference) and produced real false positives in testing."""
    attack = _feature_dict(
        'EcoCash: A device-linking session was opened using your number. If you '
        'initiated this from a new phone, tap "Continue" when the prompt appears '
        "to finish linking. If you did not initiate this, no action is needed "
        "— the session will expire automatically."
    )
    assert attack["passive_consent_device_change"] == 1.0

    for benign in [
        "Download the app using this link. Update your device settings anytime, "
        "no action needed right now.",
        "Use this link to view your device warranty. No action is needed to "
        "keep your warranty active.",
        "Your device has been linked to your account successfully.",
    ]:
        assert _feature_dict(benign)["passive_consent_device_change"] == 0.0, benign


def test_authorization_via_inbound_call():
    """MFA-push / consent-phishing: primed inbound call + an instruction to
    approve a prompt or read a code. Neither half is suspicious alone; the
    conjunction is the tell. Requests nothing, so no credential feature fires."""
    attacks = [
        "A technician will call shortly. During the call, approve the in-app notification to complete the security check. No PIN will be requested.",
        "Our agent will call now. When prompted, tap Approve on the security notification.",
        "Our officer is calling now — tell them the one-time code to verify your identity.",
    ]
    for a in attacks:
        f = _feature_dict(a)
        assert f["authorization_via_inbound_call"] == 1.0, a
        # the whole point: it asks for no credential in writing
        assert f["requests_sensitive_credentials"] == 0.0 or "code" in a.lower(), a

    for benign in [
        "Your fibre technician will call to confirm the installation window tomorrow.",
        "We will call you to discuss your loan application.",
        "Approve the payment in your app to complete checkout.",   # approve, but no inbound call
    ]:
        assert _feature_dict(benign)["authorization_via_inbound_call"] == 0.0, benign


def test_conditional_self_initiated_pin_entry_not_flagged():
    """A merchant-payment authorization is legitimately gated on the user's OWN
    prior action ('enter your PIN only if YOU intended to pay') — the mirror
    image of a scam, which demands unconditionally or gates on the NEGATIVE
    case ('if this was NOT you, send your PIN'). Both shapes must stay
    distinguishable — this isn't a blanket 'if' suppressor."""
    legit = _feature_dict(
        "FinGuard: A merchant-payment request for USD 72.00 was sent to your wallet. "
        "Enter your PIN only if you intended to pay that merchant. Receiving a cash-in "
        "does not require approval. Cancel the request if you do not recognise it."
    )
    assert legit["requests_sensitive_credentials"] == 0.0

    # the negative-case scam framing must still fire — "not" breaks the
    # self-referential phrase match, by design
    scam = _feature_dict("Your OTP 483920 was requested. If this was not you, reply with your PIN to cancel now.")
    assert scam["requests_sensitive_credentials"] == 1.0

    # an unconditional demand elsewhere must still fire regardless
    plain_demand = _feature_dict("Send your PIN to 12345 immediately to unlock your wallet")
    assert plain_demand["requests_sensitive_credentials"] == 1.0


def test_screen_mismatch_coaching():
    """Push-payment authorization hijack: a refund story sets up an in-app
    Accept prompt, then pre-excuses the prompt not matching the story. A real
    refund just lands — it never needs its own confirmation screen explained
    away in advance."""
    attacks = [
        'A USD 46.20 refund is being returned to your wallet. When the payment '
        'request appears in the app, select "Accept" to release the funds. '
        'The screen may display the merchant name rather than "refund."',

        "Your reversal is ready. Accept the request in the app to receive it "
        "— it may show as a payment to a merchant instead of a refund, that is normal.",
    ]
    for a in attacks:
        assert _feature_dict(a)["screen_mismatch_coaching"] == 1.0, a

    for benign in [
        "When the Device Protection screen appears, tap Recognise device.",
        "A USD 46.20 refund has been credited to your wallet. New balance USD 120.00.",
        "Please email us rather than calling, our lines are busy today.",
    ]:
        assert _feature_dict(benign)["screen_mismatch_coaching"] == 0.0, benign


def test_screen_mismatch_coaching_survives_alert_notice_paraphrase():
    """Same pre-excuse mechanism, different vocabulary: 'security alert'/
    'notice' instead of 'screen'/'prompt', and 'may refer to... this is part
    of the synchronisation' instead of 'may display... rather than'."""
    attack = _feature_dict(
        'FinGuard: Your replacement SIM is now paired with mobile banking. To keep '
        'access on this handset, open the next security alert and select "Keep '
        'current session." The notice may refer to ending access on another '
        "device; this is part of the synchronisation."
    )
    assert attack["screen_mismatch_coaching"] == 1.0
    assert attack["passive_consent_device_change"] == 0.0   # active ask, not silence-as-default

    for benign in [
        "Notice: this is part of our routine system maintenance tonight. No action needed.",
        "Security alert: your account was accessed from a new location.",
    ]:
        assert _feature_dict(benign)["screen_mismatch_coaching"] == 0.0, benign


def test_screen_mismatch_coaching_is_structural_not_phrase_based():
    """A third paraphrase in a row for this feature: no 'screen'/'prompt'/
    'alert'/'notice' word at all — just a quoted dialog label + 'appears',
    and 'may appear... being aligned' instead of 'may display... rather
    than'. Detection has to key on the SHAPE (a quoted label followed by
    'appears'; a modal-prediction verb; a background-process excuse), not
    another fixed phrase, or this becomes an endless list."""
    attack = _feature_dict(
        'FinGuard: The beneficiary update you started has been reconciled with your '
        'saved contacts. When "Confirm existing payee" appears, select CONFIRM to '
        "retain scheduled-payment protection. A different account suffix may appear "
        "while records are being aligned."
    )
    assert attack["screen_mismatch_coaching"] == 1.0

    for benign in [
        'Welcome to "EcoCash Plus" rewards program!',
        "Your bill amount may vary slightly due to taxes.",
    ]:
        assert _feature_dict(benign)["screen_mismatch_coaching"] == 0.0, benign


def test_ussd_advance_fee_offer():
    """Classic advance-fee trick: send airtime via a USSD code, get back more
    than you sent. The USSD code alone can't be the signal — legit airtime/
    bill templates use *151# constantly — it's the code paired with a promise
    of amplified return AND (for the generic 'get back' phrasing) a send/
    transfer verb, since a legit cashback offer says 'get back' too, just
    without ever asking you to send anything to anyone first."""
    attacks = [
        "Econet: Send $2 airtime to 0771234567 using *151*1*1# and receive $10 "
        "airtime back instantly! Limited offer today.",
        "Congratulations! Dial *151*4*0771234567*5# to send a small activation "
        "fee in airtime and claim your $100 EcoCash bonus.",
        "NetOne: Double your airtime! Transfer any amount to 0778889900 via "
        "*145# and get double back within 5 minutes.",
    ]
    for a in attacks:
        assert _feature_dict(a)["ussd_advance_fee_offer"] == 1.0, a

    for benign in [
        "Cashback! Get 5% back on airtime top-ups this week. Dial *151# to top up now.",
        "Reminder: your bill of $45.00 for DStv is due tomorrow. Pay via *151#.",
        "Statement: 12 transactions this month totalling $230.00. Dial *151*2# for details.",
        "Send $5 airtime to your child on 0771234567 using *151*1*1#. Standard network rates apply.",
    ]:
        assert _feature_dict(benign)["ussd_advance_fee_offer"] == 0.0, benign


def test_conditional_verify_inside_app_not_flagged():
    """A genuine beneficiary/payee check happens INSIDE the real app with an
    explicit reject-on-mismatch default — the user checking their own screen,
    not disclosing anything to whoever sent the SMS. That's a different shape
    from 'verify your account by clicking/replying/calling', which hands
    control to the sender."""
    legit = _feature_dict(
        "FinGuard: A beneficiary update was requested on your profile. Verify the "
        "beneficiary name and account suffix inside the FinGuard app before "
        "approving it. Reject the update if any detail differs from the payee "
        "you intended to add."
    )
    assert legit["requests_identity_verification"] == 0.0

    # a real verify-phishing demand must still fire
    scam = _feature_dict("Verify your account now at http://fake-verify.tk/x to avoid suspension.")
    assert scam["requests_identity_verification"] == 1.0


def test_unofficial_domain_alone_is_not_enough_to_call_it_deceptive():
    """has_unofficial_url will fire on any domain outside our hand-curated
    allowlist — including every real company we didn't think to enumerate.
    It must stay a weak signal on its own; deceptive_subdomain/shortener/
    lookalike are the features that should carry real weight."""
    real_company_link = _feature_dict(
        "Microsoft: A sign-in request was made from a new device. "
        "If this was not you, review activity at https://account.microsoft.com/security"
    )
    assert real_company_link["has_unofficial_url"] == 1.0   # correctly not on our allowlist
    assert real_company_link["deceptive_subdomain"] == 0.0
    assert real_company_link["brand_lookalike_domain"] == 0.0
    assert real_company_link["has_shortener_url"] == 0.0


def test_lexicons_match_whole_words_only():
    """Substring matching manufactures signals out of ordinary English:
    'stopping' contains 'pin', 'know' and 'snowfall' contain 'now'."""
    for innocent in [
        "We are stopping the cash-out",
        "Thanks for shopping with us",
        "In my opinion the service improved",
    ]:
        assert _feature_dict(innocent)["requests_sensitive_credentials"] == 0.0, innocent

    for innocent in ["Did you know the depot moved", "Heavy snowfall expected", "The cause is not known"]:
        assert _feature_dict(innocent)["urgency_word_count"] == 0.0, innocent

    # the real words still register
    assert _feature_dict("Send your PIN now")["requests_sensitive_credentials"] == 1.0
    assert _feature_dict("Act now, urgent")["urgency_word_count"] >= 2


def test_code_read_aloud_to_caller_is_a_credential_request():
    """Vishing setup: the SMS never asks for the code in writing, it primes the
    victim to read it to whoever calls. Same theft, different channel."""
    vishing = _feature_dict(
        "OneMoney Support: We are stopping the USD 60.00 cash-out now. Read the "
        "verification code sent to your phone when our agent calls you."
    )
    assert vishing["requests_sensitive_credentials"] == 1.0

    # a genuine code delivery still must not fire
    assert _feature_dict("Your verification code is 481920. Valid for 5 minutes.")[
        "requests_sensitive_credentials"] == 0.0


def test_currency_detected_across_regional_codes():
    """A ZW-only currency list is blind on most of the imported pan-African
    corpus, which silently skews whatever the model learns from this feature."""
    for amount in ["You received $50", "USD 45.00 pending", "Earn ZAR25000/week",
                   "pay ZMW50", "refund of NGN2500", "loan of ETB500", "TZS1000 detected"]:
        assert _feature_dict(amount)["has_currency_amount"] == 1.0, amount
    assert _feature_dict("No amount mentioned here at all")["has_currency_amount"] == 0.0


def test_deceptive_subdomain_is_brand_agnostic():
    """The lookalike must be caught by SHAPE, not by matching a brand list —
    BRAND_NAMES only ever covers impersonations we thought to enumerate."""
    unlisted_brand = _feature_dict(
        "ZimPay: A transfer of USD 38.50 to T Moyo is awaiting confirmation. "
        "Cancel via https://secure.zimpay.co.zw.account-review.com"
    )
    listed_brand = _feature_dict("Login at http://ecocash.co.zw.evil.tk/x")
    assert unlisted_brand["deceptive_subdomain"] == 1.0
    assert listed_brand["deceptive_subdomain"] == 1.0

    for honest in [
        "Download the app at https://www.ecocash.co.zw/app",
        "Check your balance at https://onemoney.co.zw",
        "Verify at http://ecocash-verify.tk/x",       # unofficial, but not this trick
        "Update your address: http://bit.ly/abc123",
    ]:
        assert _feature_dict(honest)["deceptive_subdomain"] == 0.0, honest


def test_shortcode_requires_a_contact_cue():
    """A short code is a destination you're told to contact — bare digits of
    the same length are usually a year, a product name, or a quantity."""
    for destination in [
        "Call our line on 4521 now",
        "Send your PIN to 12345 immediately",
        "Agent line: dial 8804 to complete your cash-in",
        "Confirm your PIN at 5544",
        "Tumirai PIN yenyu pa 8804",
        "Thumela i-PIN yakho ku 3321",
        "Verify your identity by calling 5544 to unlock your wallet",
    ]:
        assert _feature_dict(destination)["has_shortcode"] == 1.0, destination

    for not_a_destination in [
        "Microsoft 365: your password was changed",
        "Password changed on 29 Jul 2026",
        "Statement: 412 transactions this month",
        "Your parcel weighs 2500 grams",
    ]:
        assert _feature_dict(not_a_destination)["has_shortcode"] == 0.0, not_a_destination


def test_negated_pin_request_not_flagged():
    advisory = _feature_dict(
        "Steward Bank: We noticed a recent login attempt. Our team will never ask for your "
        "password by SMS. If this was not you, call customer support using the number on your card."
    )
    assert advisory["requests_sensitive_credentials"] == 0.0

    actual_request = _feature_dict("Send your password to unlock your account now")
    assert actual_request["requests_sensitive_credentials"] == 1.0

    # negation in one sentence shouldn't mask a real request in another
    mixed = _feature_dict("We will never call you. But please send your PIN to confirm.")
    assert mixed["requests_sensitive_credentials"] == 1.0


def test_shona_ndebele_scam_triggers_features():
    shona = _feature_dict("Ndapota tumirai PIN yenyu ikozvino kuti musavharirwe")
    assert shona["urgency_word_count"] >= 2  # ndapota + ikozvino
    assert shona["requests_sensitive_credentials"] == 1.0
    ndebele = _feature_dict("Thumela i-PIN yakho khathesi, i-akhawunti izavalwa lamuhla")
    assert ndebele["urgency_word_count"] >= 2  # khathesi + lamuhla
    assert ndebele["requests_sensitive_credentials"] == 1.0


def test_shortener_url_detected():
    assert _feature_dict("Update your address: http://bit.ly/abc123")["has_shortener_url"] == 1.0
    assert _feature_dict("Visit http://example.com/abc123 today")["has_shortener_url"] == 0.0


def test_official_vs_unofficial_domain():
    official = _feature_dict("Download the app at https://www.ecocash.co.zw/app")
    fake = _feature_dict("Verify at http://ecocash-verify.tk/x")
    subdomain_trick = _feature_dict("Login at http://ecocash.co.zw.evil.tk/x")
    assert official["has_unofficial_url"] == 0.0
    assert official["brand_lookalike_domain"] == 0.0
    assert fake["has_unofficial_url"] == 1.0
    assert fake["brand_lookalike_domain"] == 1.0
    assert subdomain_trick["has_unofficial_url"] == 1.0
    assert subdomain_trick["brand_lookalike_domain"] == 1.0


def test_brand_spoof_indicator_requires_brand_and_link_or_shortcode():
    spoofed = _feature_dict("EcoCash Support: verify at http://ecocash-verify.tk/x")
    legit_mention = _feature_dict("Your EcoCash statement is ready. Balance $80.")
    assert spoofed["brand_spoof_indicator"] == 1.0
    assert legit_mention["brand_spoof_indicator"] == 0.0
