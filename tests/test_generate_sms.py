import random

from app.models.sms import SmsLabel
from scripts.generate_sms import gen_sms_messages

SHONA_MARKERS = ["ndapota", "mari", "yenyu", "nhasi", "mubairo", "makakunda", "muagent", "fonerai"]


def test_generation_is_deterministic_given_seed():
    a = [m.text for m in gen_sms_messages(random.Random(7), 200)]
    b = [m.text for m in gen_sms_messages(random.Random(7), 200)]
    assert a == b


def test_all_labels_represented():
    messages = gen_sms_messages(random.Random(1), 600)
    labels = {m.label for m in messages}
    assert labels == set(SmsLabel)


def test_includes_shona_code_switched_variants():
    messages = gen_sms_messages(random.Random(2), 400)
    shona_count = sum(
        1 for m in messages if any(marker in m.text.lower() for marker in SHONA_MARKERS)
    )
    assert shona_count > 0


def test_source_is_generator():
    messages = gen_sms_messages(random.Random(3), 20)
    assert all(m.source == "generator" for m in messages)
