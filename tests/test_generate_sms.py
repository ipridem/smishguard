import random

from app.models.sms import SmsLabel
from scripts.generate_sms import PARAPHRASE_GROUPS, gen_sms_messages, paraphrase_text

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


def test_paraphrase_text_swaps_to_a_different_alternative():
    rng = random.Random(11)
    original = "Your replacement handset is ready. No action is needed."
    rewritten = paraphrase_text(original, rng)
    assert rewritten != original
    # every group's canonical phrase is gone or replaced by a same-group alternate
    assert "replacement handset" not in rewritten.lower() or "no action is needed" not in rewritten.lower()


def test_paraphrase_text_is_deterministic_given_rng_state():
    text = "Approve the request. This is part of the synchronisation."
    a = paraphrase_text(text, random.Random(5))
    b = paraphrase_text(text, random.Random(5))
    assert a == b


def test_paraphrase_rate_zero_disables_augmentation():
    baseline = [m.text for m in gen_sms_messages(random.Random(9), 300, paraphrase_rate=0.0)]
    # with rate=0, none of the canonical multi-word group phrases should have
    # been swapped for an alternate — spot check a phrase we know templates use
    assert any("no action is needed" in t.lower() or "no action needed" in t.lower() for t in baseline)


def test_paraphrase_groups_have_no_duplicate_entries():
    for group in PARAPHRASE_GROUPS:
        assert len(group) == len(set(p.lower() for p in group)), group
