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
    llm_provider: str = "ollama"          # ollama | openai | gemini
    llm_model: str = "llama3.2"
    llm_temperature: float = 0.0
    ollama_base_url: str = "http://localhost:11434"
    # Context window. Ollama's default can be 2048 (too small for our prompts);
    # 8192 needs ~1 GB more RAM. 4096 fits BasketBee's prompts with room to spare.
    ollama_num_ctx: int = 4096
    openai_api_key: str | None = None
    openai_base_url: str | None = None     # e.g. https://api.groq.com/openai/v1 for Groq
    google_api_key: str | None = None
    # How agent tools reach the mock ERP/WMS/TMS:
    #   inprocess = call the API inside Python (no uvicorn needed, great for dev)
    #   http      = real HTTP calls to the running server at api_base_url
    tools_transport: str = "inprocess"
    # Pre-compute forecasts in the background when the API starts (dashboard opens fast)
    warmup_on_startup: bool = True
    festival_date: str = "2026-11-08"     # next big festival (Diwali) shown in the app
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