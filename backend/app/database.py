"""Database connection and session management for FastAPI."""

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from backend.app.config import settings

engine: Engine = create_engine(settings.database_url)


def get_engine() -> Engine:
    """Dependency for routes that need DB access."""
    return engine