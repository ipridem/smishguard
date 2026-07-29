"""App configuration loaded from environment variables."""
import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///smishing.db")
    SMISHING_MODEL_PATH = os.environ.get("SMISHING_MODEL_PATH", "ml/smishing_model.joblib")
