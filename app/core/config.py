"""Application configuration using pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # Storage
    STORAGE_PATH: str = "storage/jobs"

    # Concurrency
    MAX_CONCURRENT_JOBS: int = 5

    # File expiry
    FILE_EXPIRY_HOURS: int = 24

    # Retry settings
    MAX_RETRY_ATTEMPTS: int = 3
    RETRY_BACKOFF_BASE: int = 2

    # Audio mixing
    BACKGROUND_VOLUME: float = 0.2

    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"


settings = Settings()
