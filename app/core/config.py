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
