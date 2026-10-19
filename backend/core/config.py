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
    # Where the tools find the mock ERP/WMS/TMS (the running FastAPI server)
    api_base_url: str = "http://127.0.0.1:8000"
    # "Today" in the BasketBee story - the last day of sales history
    business_date: str = "2026-10-18"

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()