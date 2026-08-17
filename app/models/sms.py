"""SMS message corpus for the smishing classifier (labels + inference results)."""
import enum
from datetime import datetime, timezone

from sqlalchemy import Enum, Float, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from . import Base
from .types import UTCDateTime


class SmsLabel(str, enum.Enum):
    LEGIT = "legit"
    PHISHING_CREDENTIAL = "phishing_credential"
    PHISHING_REVERSAL_SCAM = "phishing_reversal_scam"
    FAKE_AGENT = "fake_agent"
    PRIZE_SCAM = "prize_scam"
    # account takeover: unauthorised account/SIM/device change. The goal is
    # control of the account, not a credential — the message requests nothing,
    # it relies on the victim NOT acting (see passive_consent_device_change).
    ACCOUNT_TAKEOVER = "account_takeover"
    OTHER_FRAUD = "other_fraud"


class SmsMessage(Base):
    __tablename__ = "sms_messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    label: Mapped[SmsLabel | None] = mapped_column(
        Enum(SmsLabel, native_enum=False, length=32), nullable=True
    )  # ground truth, set for training/eval corpus rows
    source: Mapped[str] = mapped_column(String(64), nullable=False)  # generator | scam_report | upload
    # groups every paraphrase/parameter-fill of the same template together, so
    # a grouped train/test split can keep them on the same side. Without this,
    # train_test_split scatters near-duplicate rows across both sides and the
    # test score measures memorisation, not generalisation. Null for rows not
    # sourced from a template (scam_report | upload).
    template_id: Mapped[str | None] = mapped_column(String(16), nullable=True)
    predicted_class: Mapped[SmsLabel | None] = mapped_column(
        Enum(SmsLabel, native_enum=False, length=32), nullable=True
    )
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), default=lambda: datetime.now(timezone.utc)
    )
