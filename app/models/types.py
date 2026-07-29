"""DateTime(timezone=True) doesn't round-trip tzinfo on SQLite — a value
written tz-aware comes back naive after a session expire+reload (fine on
Postgres, where it maps to TIMESTAMPTZ). Silent until something compares a
reloaded value against a fresh tz-aware datetime, which either crashes
(TypeError: can't compare offset-naive and offset-aware) or, worse,
produces a different serialization for the same instant depending on
whether it round-tripped through the DB yet.

Fixing it once here — every timestamp in the app is UTC by convention
(see README) — instead of requiring every caller to defensively re-attach
tzinfo. Purely an ORM-level wrapper; the underlying column type (and thus
the schema) is unchanged, so no migration is needed.
"""
from datetime import timezone

from sqlalchemy import DateTime
from sqlalchemy.types import TypeDecorator


class UTCDateTime(TypeDecorator):
    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None and value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value

    def process_result_value(self, value, dialect):
        if value is not None and value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value
