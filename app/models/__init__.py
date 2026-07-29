from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


from .sms import SmsMessage  # noqa: E402,F401  (registers model with Base.metadata)
