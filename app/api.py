"""JSON API for the smishing classifier."""
import csv
import io
from functools import lru_cache

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.config import Config
from app.smishing.model import classify, load_model

router = APIRouter(prefix="/api")

MAX_BATCH_ROWS = 500
# Bands on `risk` (= P(any fraud class)), NOT on the 6-class argmax. Argmax
# confidence collapses when a scam splits its mass across several fraud
# classes, which is exactly when a security tool must not sound unsure.
RISK_FRAUD_THRESHOLD = 0.6    # at or above: call it fraud
RISK_LEGIT_THRESHOLD = 0.4    # at or below: call it legitimate
# between the two the model genuinely can't tell — that's when to warn
LOW_CONFIDENCE_THRESHOLD = 0.6  # applies to the *class* label only


@lru_cache(maxsize=1)
def get_pipeline():
    # the joblib artifact is ~2MB; load once per process, not per request
    return load_model(Config.SMISHING_MODEL_PATH)


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


class ClassifyResponse(BaseModel):
    label: str
    confidence: float | None   # certainty in the specific class
    risk: float | None         # P(any fraud class) — the "is this a scam?" answer
    top_tokens: list[TokenContribution]
    risk_signals: list[RiskSignal]


class BatchRow(ClassifyResponse):
    text: str


class Meta(BaseModel):
    low_confidence_threshold: float
    risk_fraud_threshold: float
    risk_legit_threshold: float
    max_batch_rows: int
    labels: list[str]


@router.get("/meta", response_model=Meta)
def meta() -> Meta:
    """Thresholds and label vocabulary, so the UI doesn't hardcode them."""
    return Meta(
        low_confidence_threshold=LOW_CONFIDENCE_THRESHOLD,
        risk_fraud_threshold=RISK_FRAUD_THRESHOLD,
        risk_legit_threshold=RISK_LEGIT_THRESHOLD,
        max_batch_rows=MAX_BATCH_ROWS,
        labels=sorted(get_pipeline().named_steps["clf"].classes_),
    )


@router.post("/classify", response_model=ClassifyResponse)
def classify_one(req: ClassifyRequest) -> dict:
    return classify(get_pipeline(), req.text)


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
