"""App configuration loaded from environment variables."""
import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///smishing.db")
    SMISHING_MODEL_PATH = os.environ.get("SMISHING_MODEL_PATH", "ml/smishing_model.joblib")
    # optional: LLM second opinion for inconclusive-risk messages (see
    # app/smishing/llm_review.py). Unset -> feature silently disabled.
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
