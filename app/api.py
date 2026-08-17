"""JSON API for the smishing classifier."""
import csv
import io
import os

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.config import Config
from app.smishing.llm_review import groq_second_opinion
from app.smishing.model import classify, load_model

router = APIRouter(prefix="/api")

MAX_BATCH_ROWS = 500
# Bands on `risk` (= P(any fraud class)), NOT on the 6-class argmax. Argmax
# confidence collapses when a scam splits its mass across several fraud
# classes, which is exactly when a security tool must not sound unsure.
# `risk` is what gates the verdict shown to the user; `confidence` (the
# class argmax) is informational only, surfaced separately in the UI as a
# note on which fraud category was matched — never used to override risk.
#
# Three bands, not one verdict: adversarial-suite risk distributions still
# overlap (genuine up to 0.84, smish down to 0.62 on the current model), so
# no single cutoff cleanly separates them. 0.5 remains the F1 optimum
# (measured via scripts/eval_adversarial.py's threshold sweep, both before
# and after retraining) and is kept as the legit/warn boundary; 0.8 adds a
# second, higher-confidence "block" tier so a 0.55-risk message doesn't read
# with the same urgency as a 0.98 one.
RISK_LEGIT_THRESHOLD = 0.4    # at or below: call it legitimate
RISK_FRAUD_THRESHOLD = 0.5    # at or above: warn — F1-optimal cutoff
RISK_BLOCK_THRESHOLD = 0.8    # at or above: block — high-confidence fraud
LOW_CONFIDENCE_THRESHOLD = 0.6  # applies to the *class* label only
# LLM second opinion band: wider than the legit/warn boundary on purpose.
# The population a linear model genuinely can't resolve (real vs. trap
# messages sitting a few points apart in risk) extends past 0.5 up toward
# where the "block" tier starts, not just the narrow strip around it.
LLM_REVIEW_RISK_LOW = 0.4
LLM_REVIEW_RISK_HIGH = 0.7

# uvicorn --reload watches .py files, not the .joblib artifact — a
# `scripts/train_model.py` run while the server is up would otherwise serve
# stale predictions from the pipeline loaded at process start until a manual
# restart. Reload whenever the file's mtime moves past what we last loaded.
_pipeline = None
_pipeline_mtime = None


def get_pipeline():
    global _pipeline, _pipeline_mtime
    mtime = os.path.getmtime(Config.SMISHING_MODEL_PATH)
    if _pipeline is None or mtime != _pipeline_mtime:
        _pipeline = load_model(Config.SMISHING_MODEL_PATH)
        _pipeline_mtime = mtime
    return _pipeline


class ClassifyRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class TokenContribution(BaseModel):
    token: str
    contribution: float


class RiskSignal(BaseModel):
    label: str
    present: bool
    weight: float
    note: str | None = None


class LlmOpinion(BaseModel):
    verdict: str
    confidence: float
    reasoning: str


class ClassifyResponse(BaseModel):
    label: str
    confidence: float | None   # certainty in the specific class
    risk: float | None         # P(any fraud class) — the "is this a scam?" answer
    top_tokens: list[TokenContribution]
    risk_signals: list[RiskSignal]
    llm_opinion: LlmOpinion | None = None


class BatchRow(ClassifyResponse):
    text: str


class Meta(BaseModel):
    low_confidence_threshold: float
    risk_fraud_threshold: float
    risk_legit_threshold: float
    risk_block_threshold: float
    max_batch_rows: int
    labels: list[str]
    llm_review_available: bool


@router.get("/meta", response_model=Meta)
def meta() -> Meta:
    """Thresholds and label vocabulary, so the UI doesn't hardcode them."""
    return Meta(
        low_confidence_threshold=LOW_CONFIDENCE_THRESHOLD,
        risk_fraud_threshold=RISK_FRAUD_THRESHOLD,
        risk_legit_threshold=RISK_LEGIT_THRESHOLD,
        risk_block_threshold=RISK_BLOCK_THRESHOLD,
        max_batch_rows=MAX_BATCH_ROWS,
        labels=sorted(get_pipeline().named_steps["clf"].classes_),
        llm_review_available=bool(Config.GROQ_API_KEY),
    )


@router.post("/classify", response_model=ClassifyResponse)
def classify_one(req: ClassifyRequest) -> dict:
    result = classify(get_pipeline(), req.text)
    # only the narrow band where the local model already admits it can't
    # tell — not every request, and never the batch path (cost + latency).
    # Config check here too (not just inside groq_second_opinion) so the
    # feature being off is a true no-op, not a call that immediately bails.
    risk = result.get("risk")
    if Config.GROQ_API_KEY and risk is not None and LLM_REVIEW_RISK_LOW < risk < LLM_REVIEW_RISK_HIGH:
        result["llm_opinion"] = groq_second_opinion(req.text)
    return result


@router.post("/classify/batch", response_model=list[BatchRow])
async def classify_batch(file: UploadFile = File(...)) -> list[dict]:
    """One SMS per row, first column. Extra columns are ignored."""
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="CSV files only.")
    try:
        text = (await file.read()).decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Could not read file — upload a UTF-8 encoded CSV.")

    rows = [r[0] for r in csv.reader(io.StringIO(text)) if r and r[0].strip()][:MAX_BATCH_ROWS]
    if not rows:
        raise HTTPException(status_code=400, detail="No messages found in that CSV.")
    pipeline = get_pipeline()
    return [{"text": r, **classify(pipeline, r)} for r in rows]
