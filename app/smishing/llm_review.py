"""Optional LLM second opinion, called only when the local model's risk score
is inconclusive (see api.RISK_LEGIT_THRESHOLD / RISK_FRAUD_THRESHOLD).

The local engineered features match known attack SHAPES — a lexicon or regex
someone had to write after seeing that exact evasion. An LLM instead reasons
about the coached ACTION, which generalizes past paraphrasing no lexicon has
seen yet. It does not replace the local model: it's a second opinion for the
narrow band where the primary, fast, free, explainable classifier is already
admitting it doesn't know.

Best-effort only. No API key, a network error, a timeout, or a malformed
response all fall back to None — the local classification stands on its own
either way. This must never be able to break classification.
"""
import json

import httpx

from app.config import Config

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT = (
    "You are a smishing (SMS phishing) analyst reviewing a single SMS message that a "
    "local classifier could not confidently label. Do not pattern-match on keywords — "
    "reason about WHAT ACTION the message is coaching the reader to take, and WHO "
    "BENEFITS from that action. Known evasions to watch for: pre-excusing a mismatch "
    "between the message's story and an in-app prompt ('the screen may show X, that's "
    "normal'); gating a fraudulent change on silence-by-default ('no action needed if "
    "this wasn't you'); routing authorization through a phone call or on-screen "
    "approval instead of asking for a credential in writing. A message can be safe "
    "even if it mentions PINs, verification, or account changes, if it describes a "
    "user-initiated, in-app action with a safe default. "
    'Respond with strict JSON only, no other text: '
    '{"verdict": "fraud" or "legit", "confidence": 0.0-1.0, "reasoning": "<=40 words"}'
)


def groq_second_opinion(text: str) -> dict | None:
    if not Config.GROQ_API_KEY:
        return None
    try:
        resp = httpx.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {Config.GROQ_API_KEY}"},
            json={
                "model": Config.GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                "temperature": 0,
                "max_tokens": 150,
                "response_format": {"type": "json_object"},
            },
            timeout=8.0,
        )
        resp.raise_for_status()
        parsed = json.loads(resp.json()["choices"][0]["message"]["content"])

        verdict = parsed["verdict"]
        confidence = float(parsed["confidence"])
        reasoning = str(parsed["reasoning"])
        if verdict not in ("fraud", "legit") or not (0.0 <= confidence <= 1.0):
            return None
        return {"verdict": verdict, "confidence": confidence, "reasoning": reasoning}
    except Exception:
        # network error, timeout, non-2xx, malformed JSON, missing/bad
        # fields — any of these just means "no second opinion this time"
        return None
