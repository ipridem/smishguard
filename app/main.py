"""FastAPI entrypoint: JSON API + the static single-page UI.

Run with:
    .venv/Scripts/python -m uvicorn app.main:app --reload --port 5001
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api import router

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(
    title="FinGuard Smishing Classifier",
    description="Classifies SMS messages as legitimate or one of five fraud categories, "
                "with per-feature explainability. Zimbabwe mobile-money context.",
)
app.include_router(router)


@app.middleware("http")
async def revalidate_static(request, call_next):
    """Force revalidation of the SPA assets.

    StaticFiles sets no Cache-Control, so browsers heuristically hard-cache
    app.js/styles.css and silently serve a stale UI against a fresh API. ETags
    still make the revalidation a cheap 304.
    """
    response = await call_next(request)
    if not request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


# mounted last and at the root so /api/* resolves first
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
