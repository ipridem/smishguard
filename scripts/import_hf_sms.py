"""Import the HuggingFace African smishing parquet into sms_messages,
mapping its smishing_type onto our SmsLabel taxonomy. Deduplicates texts
within the file and skips texts already present in the table.

Usage:
    .venv/Scripts/python scripts/import_hf_sms.py path/to/train.parquet
"""
import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from app.config import Config
from app.extensions import db_session, init_db
from app.models.sms import SmsLabel, SmsMessage

SOURCE = "hf_african_smishing"

# The HF dataset has no reversal-scam or fake-agent categories; everything
# else folds into our existing labels.
TYPE_TO_LABEL = {
    "none": SmsLabel.LEGIT,
    "mobile_money_pin_theft": SmsLabel.PHISHING_CREDENTIAL,
    "bank_alert_fake": SmsLabel.PHISHING_CREDENTIAL,
    "account_suspension_threat": SmsLabel.PHISHING_CREDENTIAL,
    "tax_refund_scam": SmsLabel.PHISHING_CREDENTIAL,
    "delivery_notification": SmsLabel.PHISHING_CREDENTIAL,
    "prize_lottery_scam": SmsLabel.PRIZE_SCAM,
    "airtime_reward_scam": SmsLabel.PRIZE_SCAM,
    "loan_approval_scam": SmsLabel.OTHER_FRAUD,
    "job_offer_scam": SmsLabel.OTHER_FRAUD,
    "investment_scam": SmsLabel.OTHER_FRAUD,
    "government_impersonation": SmsLabel.OTHER_FRAUD,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parquet_path")
    parser.add_argument("--reset", action="store_true", help=f"Delete existing {SOURCE} rows first.")
    args = parser.parse_args()

    init_db(Config.DATABASE_URL)
    if args.reset:
        db_session.query(SmsMessage).filter(SmsMessage.source == SOURCE).delete()
        db_session.commit()

    df = pd.read_parquet(args.parquet_path)
    unknown = set(df["smishing_type"]) - set(TYPE_TO_LABEL)
    if unknown:
        raise SystemExit(f"Unmapped smishing_type values: {unknown}")

    df = df.drop_duplicates(subset="sms_text")
    existing = {t for (t,) in db_session.query(SmsMessage.text).all()}

    counts = Counter()
    rows = []
    for text, stype in zip(df["sms_text"], df["smishing_type"]):
        if text in existing:
            continue
        label = TYPE_TO_LABEL[stype]
        rows.append(SmsMessage(text=text, label=label, source=SOURCE))
        counts[label.value] += 1

    db_session.add_all(rows)
    db_session.commit()
    print(f"imported {len(rows)} messages from {args.parquet_path}")
    for label, n in sorted(counts.items()):
        print(f"  {label}: {n}")


if __name__ == "__main__":
    main()
