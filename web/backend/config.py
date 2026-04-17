"""
config.py — Environment variable configuration for ArthSaathi backend.
Load from .env file or environment.
"""
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:password@localhost:5432/arthsaathi"
)
MODEL_DIR: str = os.getenv("MODEL_DIR", "D:/Project/models")
SECRET_KEY: str = os.getenv("SECRET_KEY", "changeme-secret-key")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
