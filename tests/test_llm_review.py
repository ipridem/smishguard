"""groq_second_opinion is best-effort: no key, bad network, or a malformed
response must all fall back to None, never raise — classification has to
stand on its own regardless of what this call does."""
import httpx
import pytest

from app.smishing import llm_review


def test_returns_none_without_an_api_key(monkeypatch):
    monkeypatch.setattr(llm_review.Config, "GROQ_API_KEY", None)
    assert llm_review.groq_second_opinion("Send your PIN now") is None


def test_parses_a_well_formed_response(monkeypatch):
    monkeypatch.setattr(llm_review.Config, "GROQ_API_KEY", "test-key")

    def fake_post(url, headers, json, timeout):
        content = '{"verdict": "fraud", "confidence": 0.82, "reasoning": "asks for a PIN"}'
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"choices": [{"message": {"content": content}}]},
        )

    monkeypatch.setattr(llm_review.httpx, "post", fake_post)
    result = llm_review.groq_second_opinion("Send your PIN now")
    assert result == {"verdict": "fraud", "confidence": 0.82, "reasoning": "asks for a PIN"}


@pytest.mark.parametrize("content", [
    '{"verdict": "maybe", "confidence": 0.5, "reasoning": "unclear"}',   # invalid verdict
    '{"verdict": "fraud", "confidence": 1.5, "reasoning": "unclear"}',   # confidence out of range
    '{"verdict": "fraud"}',                                              # missing fields
    'not json at all',
])
def test_malformed_response_returns_none(monkeypatch, content):
    monkeypatch.setattr(llm_review.Config, "GROQ_API_KEY", "test-key")

    def fake_post(url, headers, json, timeout):
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"choices": [{"message": {"content": content}}]},
        )

    monkeypatch.setattr(llm_review.httpx, "post", fake_post)
    assert llm_review.groq_second_opinion("Send your PIN now") is None


def test_network_failure_returns_none(monkeypatch):
    monkeypatch.setattr(llm_review.Config, "GROQ_API_KEY", "test-key")

    def fake_post(*args, **kwargs):
        raise httpx.ConnectTimeout("timed out")

    monkeypatch.setattr(llm_review.httpx, "post", fake_post)
    assert llm_review.groq_second_opinion("Send your PIN now") is None


def test_non_2xx_response_returns_none(monkeypatch):
    monkeypatch.setattr(llm_review.Config, "GROQ_API_KEY", "test-key")

    def fake_post(url, headers, json, timeout):
        return httpx.Response(401, request=httpx.Request("POST", url), json={"error": "bad key"})

    monkeypatch.setattr(llm_review.httpx, "post", fake_post)
    assert llm_review.groq_second_opinion("Send your PIN now") is None
