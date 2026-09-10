from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables and `.env`."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    app_host: str = "127.0.0.1"
    app_port: int = Field(default=8000, ge=1, le=65535)
    frontend_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5173"]
    )
    data_dir: Path = Path("../AI-Builders-Hackathon-2026-Data/Data_Final")
    database_url: str = "sqlite:///./veritas.db"
    openrouter_api_key: str | None = None
    openrouter_model: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_connect_timeout_seconds: float = Field(default=5, gt=0, le=60)
    llm_read_timeout_seconds: float = Field(default=45, gt=0, le=300)
    llm_total_timeout_seconds: float = Field(default=60, gt=0, le=360)
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    llm_global_concurrency: int = Field(default=8, ge=1, le=100)
    llm_calls_per_analysis: int = Field(default=16, ge=1, le=100)
    llm_max_tokens: int = Field(default=1_200, ge=32, le=16_384)
    llm_max_response_bytes: int = Field(default=262_144, ge=1_024, le=2_000_000)
    llm_max_cost_usd_per_analysis: float = Field(default=1.0, gt=0, le=100)
    llm_temperature: float = Field(default=0.0, ge=0, le=2)
    prompt_version: str = "2026-09-10.1"
    llm_cache_enabled: bool = True
    llm_cache_retention_days: int = Field(default=30, ge=1, le=365)
    llm_cache_max_rows: int = Field(default=10_000, ge=1, le=1_000_000)
    analysis_pipeline_version: str = "2026-09-10.1"
    max_concurrent_analyses: int = Field(default=1, ge=1, le=16)
    max_queued_analyses: int = Field(default=10, ge=0, le=1_000)
    analysis_retry_after_seconds: int = Field(default=5, ge=1, le=3_600)
    sqlite_busy_timeout_ms: int = Field(default=5_000, ge=100, le=60_000)
    embedding_model_path_or_id: str | None = None
    retrieval_max_top_k: int = Field(default=10, ge=1, le=50)
    financial_low_runway_months: float = Field(default=6, ge=0, le=120)

    @field_validator("frontend_origins", mode="before")
    @classmethod
    def split_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [origin.strip().rstrip("/") for origin in value.split(",") if origin.strip()]
        return [origin.rstrip("/") for origin in value]

    @property
    def resolved_data_dir(self) -> Path:
        """Resolve a relative data directory from the backend root, never the CWD."""
        if self.data_dir.is_absolute():
            return self.data_dir
        backend_root = Path(__file__).resolve().parents[2]
        return (backend_root / self.data_dir).resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()
