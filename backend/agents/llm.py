"""
BasketBee - LLM provider factory.

One function, get_llm(), returns the chat model configured in .env.
Switching providers needs NO code change:

    LLM_PROVIDER=ollama   LLM_MODEL=llama3.1:8b        (free, local, offline demo)
    LLM_PROVIDER=groq     LLM_MODEL=openai/gpt-oss-120b   GROQ_API_KEY=...  (free, fast cloud)
    (Groq retires models often: llama-3.3-70b-versatile left the free tier on 16 Aug 2026.
     Current list: console.groq.com/docs/models)
    LLM_PROVIDER=openai   LLM_MODEL=gpt-4o-mini        OPENAI_API_KEY=...
    LLM_PROVIDER=gemini   LLM_MODEL=gemini-2.0-flash   GOOGLE_API_KEY=...

OpenAI / Gemini need one extra package (installed only if you use them):
    pip install langchain-openai
    pip install langchain-google-genai
"""

from langchain_core.language_models.chat_models import BaseChatModel

from backend.core.config import settings


class LLMConfigError(RuntimeError):
    pass


def get_llm(model: str | None = None, temperature: float | None = None) -> BaseChatModel:
    provider = settings.llm_provider.lower()
    model = model or settings.llm_model
    temperature = settings.llm_temperature if temperature is None else temperature

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model, temperature=temperature, base_url=settings.ollama_base_url,
            num_ctx=settings.ollama_num_ctx,   # default window can be 2048 tokens: too small!
            keep_alive="30m",                  # keep the model loaded between questions
        )

    if provider == "groq":
        # Groq: free cloud API, very fast. OpenAI-compatible, so we reuse ChatOpenAI.
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise LLMConfigError("Run: pip install langchain-openai") from e
        if not settings.groq_api_key:
            raise LLMConfigError("Set GROQ_API_KEY in .env (free key from console.groq.com)")
        extra = {}
        if "gpt-oss" in model:
            # Reasoning model: "low" effort = fewer tokens (free tier allows 8,000/min) and faster answers
            extra["reasoning_effort"] = "low"
        return ChatOpenAI(model=model, temperature=temperature, api_key=settings.groq_api_key,
                          base_url="https://api.groq.com/openai/v1", timeout=60, max_retries=3, **extra)

    if provider == "openai":
        # Also works for any OpenAI-compatible API (e.g. Groq) via OPENAI_BASE_URL
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise LLMConfigError("Run: pip install langchain-openai") from e
        if not settings.openai_api_key:
            raise LLMConfigError("Set OPENAI_API_KEY in .env")
        return ChatOpenAI(model=model, temperature=temperature, api_key=settings.openai_api_key,
                          base_url=settings.openai_base_url)

    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as e:
            raise LLMConfigError("Run: pip install langchain-google-genai") from e
        if not settings.google_api_key:
            raise LLMConfigError("Set GOOGLE_API_KEY in .env")
        return ChatGoogleGenerativeAI(model=model, temperature=temperature,
                                      google_api_key=settings.google_api_key)

    raise LLMConfigError(f"Unknown LLM_PROVIDER '{settings.llm_provider}' (use ollama, groq, openai or gemini)")