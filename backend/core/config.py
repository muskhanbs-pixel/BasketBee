"""
BasketBee - central configuration.

Reads settings from the .env file in the project root, so nothing
(database path, LLM model, API keys) is hard-coded in the code.

Usage anywhere in the project:
    from backend.core.config import settings
    print(settings.database_url)
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]   # .../BasketBee


class Settings(BaseSettings):
    app_name: str = "BasketBee"
    # Absolute path, so the DB is found no matter which folder you run from
    database_url: str = f"sqlite:///{(PROJECT_ROOT / 'basketbee.db').as_posix()}"
    llm_provider: str = "ollama"
    llm_model: str = "llama3.2"

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()