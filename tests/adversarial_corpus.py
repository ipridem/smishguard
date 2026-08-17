"""
SmishGuard adversarial test corpus.

40 cases: 20 expert-level smishing (evasion-hardened), 20 genuine
false-positive traps that legitimately contain money / OTP / link language.

Zimbabwe context: EcoCash, OneMoney, ZIPIT, ZESA, POTRAZ, ZIMRA, RBZ.

label:  "smish" | "genuine"
family: the evasion technique (smish) or the trap type (genuine).
        Use this to see WHICH class of attack your model misses.
"""

CASES = [
    # ------------------------------------------------------------------
    # SMISHING - evasion-hardened
    # ------------------------------------------------------------------
    dict(id="S01", label="smish", family="homoglyph-cyrillic",
         text="EcоCash: Your wallet has been suspended due to unusual activity. "
              "Restore access now: http://ecocash-restore.help/verify",
         note="Cyrillic о in 'EcoCash'. TF-IDF token 'ecocash' will NOT match."),

    dict(id="S02", label="smish", family="spaced-keywords",
         text="C O N G R A T S ! You have w o n $500 in the EcoCash mega draw. "
              "Claim before 18:00 today: bit.ly/ecc-claim9",
         note="Whitespace-split keywords defeat word-level TF-IDF."),

    dict(id="S03", label="smish", family="no-url-callback",
         text="ECOCASH SECURITY: A transfer of $340.00 to an unknown number is pending "
              "on your wallet. If this was not you call our agent on 0783 442 197 "
              "within 10 minutes to cancel.",
         note="No link at all. If has_url dominates the model this scores low."),

    dict(id="S04", label="smish", family="agent-reversal",
         text="Hie, I am an EcoCash agent. I mistakenly sent $120 to your number instead "
              "of my client. Please send it back to 0771 903 556 so I do not lose my job. "
              "God bless you.",
         note="Classic reversal fraud. Polite, no urgency keywords, no link, no OTP ask."),

    dict(id="S05", label="smish", family="regulator-impersonation",
         text="POTRAZ NOTICE: SIM cards not re-registered by 20 August will be "
              "permanently deactivated. Complete re-registration here: "
              "https://potraz-simreg.co.zw.verify-now.net/form",
         note="Suffix-attached lookalike domain: real brand is a SUBDOMAIN of attacker domain."),

    dict(id="S06", label="smish", family="delivery-fee",
         text="Your parcel ZW884120 is held at Harare depot. Outstanding customs fee "
              "$2.50. Pay to release: https://track-parcel-zw.com/pay/884120",
         note="Small plausible amount, low-urgency tone, realistic tracking id."),

    dict(id="S07", label="smish", family="otp-relay",
         text="EcoCash: 448192 is your one time PIN. Do not share it with anyone. "
              "Our agent will now call you to confirm this PIN and secure your account.",
         note="Contains the real 'do not share' safety text, then subverts it. "
              "Models trained on OTP-format text often score these as genuine."),

    dict(id="S08", label="smish", family="at-sign-url",
         text="Account verification required. Log in: "
              "https://www.ecocash.co.zw@secure-login-zw.ru/auth",
         note="Real brand appears before @ so the URL LOOKS legitimate. "
              "Host is actually secure-login-zw.ru."),

    dict(id="S09", label="smish", family="ip-literal-url",
         text="OneMoney: unusual login detected from a new device. Secure your wallet: "
              "http://102.68.44.19:8080/onemoney/secure",
         note="Raw IP + non-standard port. Check you have an ip_literal_url signal."),

    dict(id="S10", label="smish", family="punycode",
         text="Steward Bank: your card ending 4417 is blocked. Unblock: "
              "https://xn--stewardbnk-9db.co.zw/unlock",
         note="Punycode homoglyph domain. Renders as a near-identical brand name."),

    dict(id="S11", label="smish", family="fullwidth-unicode",
         text="ＥｃｏＣａｓｈ： Ｙｏｕｒ　ａｃｃｏｕｎｔ　ｗｉｌｌ　ｂｅ　ｃｌｏｓｅｄ． "
              "Ｖｅｒｉｆｙ： ｈｔｔｐ：／／ｅｃｃ－ｖｅｒｉｆｙ．ｎｅｔ",
         note="Fullwidth codepoints. Even has_url regex fails (：／／ not ://). "
              "Needs NFKC normalisation before featurising."),

    dict(id="S12", label="smish", family="leetspeak",
         text="3c0Cash Alert: y0ur acc0unt is on h0ld. Re-activate at "
              "ecocash-he1p.net/unl0ck or lose your ba1ance.",
         note="Digit-for-letter substitution across every high-signal token."),

    dict(id="S13", label="smish", family="whatsapp-redirect",
         text="EcoCash Customer Care: we could not reach you regarding your reversal "
              "claim. Chat with your assigned officer on WhatsApp: "
              "https://wa.me/263779112044",
         note="wa.me is a legitimate domain, so domain-reputation signals pass. "
              "Moves the victim off-channel."),

    dict(id="S14", label="smish", family="ussd-injection",
         text="To receive your $250 EcoCash promo prize, dial *151*2*2*0784556301*250# "
              "and enter your PIN when prompted. Offer expires today.",
         note="The USSD string makes the VICTIM send money out. "
              "No link, no login page. Needs a ussd_with_msisdn signal."),

    dict(id="S15", label="smish", family="obfuscated-url",
         text="Reactivate your wallet. Copy this to your browser: "
              "hxxp://ecocash-verify[.]top/login (remove brackets)",
         note="Defanged URL to evade has_url / url_count entirely."),

    dict(id="S16", label="smish", family="long-benign-preamble",
         text="Good afternoon valued customer. Thank you for banking with us this year. "
              "As part of our annual customer records update, and in line with new "
              "regulations, kindly take a moment to confirm your details at your "
              "convenience using the secure portal below. "
              "https://records-update-zw.online/confirm",
         note="Payload is 1 sentence of 5. Dilutes TF-IDF weight and any "
              "urgency-word density signal computed per-token."),

    dict(id="S17", label="smish", family="tax-refund",
         text="ZIMRA: You are eligible for a tax refund of USD 418.60 for the 2025 "
              "assessment. Submit your bank details to receive payment: "
              "zimra-refunds.claim-portal.co",
         note="Positive framing (refund, not threat). Bare domain, no http:// prefix."),

    dict(id="S18", label="smish", family="advance-fee-job",
         text="Dear applicant, you have been shortlisted for the Data Clerk post "
              "(Harare, $650/month). To confirm your interview slot pay the $15 "
              "medical screening fee to EcoCash 0716 220 884 and send proof.",
         note="No link, no brand impersonation, professional register. "
              "Purely behavioural fraud: pay-to-be-hired."),

    dict(id="S19", label="smish", family="form-harvest",
         text="Nhamo Trust is giving out groceries hampers to 200 families. Register "
              "your household using your ID and EcoCash number here: "
              "https://forms.gle/9xKq2Lmn4TzP",
         note="forms.gle is a trusted domain. Charity framing. Harvests ID + MSISDN."),

    dict(id="S20", label="smish", family="code-mixed-shona",
         text="Mhoro, ndini wako. Ndarasika nefone yangu, ndakabhadhara $80 pa account "
              "yako nekukanganisa. Ndapota tumira kuna 0778 341 209 nhasi. Thanks.",
         note="Shona/English code-mixing. If training data is English-only, "
              "TF-IDF has near-zero coverage for these tokens."),

    # ------------------------------------------------------------------
    # GENUINE - false-positive traps
    # ------------------------------------------------------------------
    dict(id="G01", label="genuine", family="real-transaction",
         text="EcoCash: Confirmed. You have sent $45.00 to 0772134556 TENDAI MOYO. "
              "Fee $0.45. New balance $112.30. Ref MP250817.1432.K84210",
         note="Money + amount + phone number, all benign. Reference-code format is "
              "the strongest genuine signal here."),

    dict(id="G02", label="genuine", family="real-otp",
         text="Your Steward Bank verification code is 739104. Do not share this code "
              "with anyone, including bank staff. Expires in 5 minutes.",
         note="Near-identical surface form to S07. The pair S07/G01 is the "
              "single most important discrimination in this suite."),

    dict(id="G03", label="genuine", family="utility-token",
         text="ZESA Prepaid: Purchase successful. Token 4521 8873 0094 6612 3380. "
              "Units 42.6 kWh. Amount $12.00. Meter 37201884556.",
         note="Long digit strings can trip 'excessive numerals' heuristics."),

    dict(id="G04", label="genuine", family="school-fees",
         text="Chisipite Senior School: Fees for Term 3 ($420) are due by 29 August. "
              "Pay via ZIPIT to 0331 4478 2201 or at the bursar's office. "
              "Statements available on request.",
         note="Deadline + payment instruction + account number. Reads like a scam "
              "to a naive urgency+account-number model, but is legitimate."),

    dict(id="G05", label="genuine", family="telco-promo",
         text="NetOne: Get 2GB for $2 valid 7 days. Dial *171# to buy. T&Cs apply. "
              "To opt out of promotional messages reply STOP.",
         note="Promo + USSD code. Distinguish from S14: no MSISDN embedded in the "
              "USSD string and no PIN request."),

    dict(id="G06", label="genuine", family="real-courier",
         text="DHL: Your shipment 8842019773 is out for delivery today between "
              "10:00 and 14:00. Track at https://www.dhl.com/zw-en/home/tracking.html",
         note="Pairs with S06. Real first-party domain, no payment demand."),

    dict(id="G07", label="genuine", family="appointment",
         text="Reminder: your appointment with Dr Chikwava is on Tue 19 Aug at 09:30, "
              "Avenues Clinic. Please arrive 15 min early. Call 0242 251 555 to reschedule.",
         note="Callback number + time pressure, entirely benign."),

    dict(id="G08", label="genuine", family="bank-marketing",
         text="CBZ Bank: Apply for a personal loan up to $5,000 with repayment over "
              "24 months. Learn more at https://www.cbz.co.zw/personal-loans "
              "or visit any branch.",
         note="Money offer + link. Legitimate marketing on a first-party domain."),

    dict(id="G09", label="genuine", family="payroll",
         text="Payroll notice: August salaries have been processed and will reflect in "
              "your account by 25/08. Payslips are on the HR portal. Contact "
              "payroll@company.co.zw with queries.",
         note="Email address + money, no action link."),

    dict(id="G10", label="genuine", family="community-money",
         text="Good evening family. The funeral contribution is $10 per household. "
              "Please send to Aunt Rudo on 0772 884 110 before Friday so we can "
              "finalise arrangements. Thank you.",
         note="Send-money-to-this-number + deadline + emotional framing. "
              "Structurally almost identical to S04. Context is the only difference."),

    dict(id="G11", label="genuine", family="tax-legit",
         text="ZIMRA: Your 2025 ITF263 tax clearance certificate has been approved. "
              "Download it from the eServices portal at https://efiling.zimra.co.zw "
              "using your usual login.",
         note="Pairs with S17. Real domain, no credential or bank-detail request."),

    dict(id="G12", label="genuine", family="fraud-warning",
         text="EcoCash Security Alert: We will never call or SMS you to ask for your "
              "PIN or a one-time code. If someone does, hang up and report to 114.",
         note="Anti-fraud education. Densely packed with scam vocabulary "
              "(PIN, one-time code, call). A keyword-heavy model flags this."),

    dict(id="G13", label="genuine", family="insurance",
         text="Nyaradzo: Your monthly policy premium of $8.50 is due on 20 August. "
              "Pay via EcoCash Biller Code 011977 or at any branch. Policy NY-447120.",
         note="Biller code + due date + amount."),

    dict(id="G14", label="genuine", family="2fa-google",
         text="G-618402 is your Google verification code.",
         note="Very short message. Check your model is not biased toward "
              "'short message = suspicious' or vice versa."),

    dict(id="G15", label="genuine", family="agent-float",
         text="EcoCash Agent: Your float top-up of $500.00 was successful. "
              "Agent code 4471. Available float $612.40. Ref AG250817.0918.T2210",
         note="Uses the word 'agent' plus large amounts. Pairs with S04."),

    dict(id="G16", label="genuine", family="university",
         text="University of Zimbabwe: Registration for Semester 1 closes 31 August. "
              "Students with outstanding fees will not be registered. "
              "Portal: https://elearning.uz.ac.zw/registration",
         note="Threat of loss of service + deadline + link. All legitimate."),

    dict(id="G17", label="genuine", family="telco-maintenance",
         text="Econet: Scheduled network maintenance in Bulawayo on 18 Aug, 01:00-04:00. "
              "You may experience brief service interruptions. We apologise for the "
              "inconvenience.",
         note="Easy negative. Should score very low; if not, your intercept is too high."),

    dict(id="G18", label="genuine", family="dispute-followup",
         text="Your reversal request REV-88214 for $60.00 is being processed. "
              "You will be notified within 48 hours. Do not share your PIN with "
              "anyone during this process.",
         note="'Reversal' is a top scam token but appears here in the genuine flow. "
              "Pairs with S13."),

    dict(id="G19", label="genuine", family="shortlink-legit",
         text="Old Mutual Zimbabwe: your annual statement is ready. View it in the "
              "app or at https://bit.ly/OMZW-statements",
         note="Legitimate business using a shortener. If has_shortener alone pushes "
              "a message over threshold, this is a guaranteed false positive."),

    dict(id="G20", label="genuine", family="cash-in",
         text="EcoCash: Cash In of $200.00 from Agent 30471 SUNRISE SUPERMARKET "
              "was successful. New balance $243.85. Thank you for using EcoCash.",
         note="Large amount + agent + balance disclosure, fully genuine."),
]


def counts():
    from collections import Counter
    return Counter(c["label"] for c in CASES)


if __name__ == "__main__":
    import csv
    import sys

    print(f"{len(CASES)} cases: {dict(counts())}", file=sys.stderr)
    with open("corpus.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "label", "family", "text", "note"])
        w.writeheader()
        w.writerows(CASES)
    print("wrote corpus.csv", file=sys.stderr)
