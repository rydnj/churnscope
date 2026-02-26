"""FastAPI application settings."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://churnscope:churnscope_dev@localhost:5433/churnscope"
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]
    api_prefix: str = "/api/v1"

    class Config:
        env_file = ".env"


settings = Settings()