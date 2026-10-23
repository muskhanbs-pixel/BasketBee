
"""
BasketBee doctor - checks every part of the system and says exactly what to fix.

Run from the project root (venv active):
    python doctor.py            quick check (about 5 seconds)
    python doctor.py --llm      also asks the model a real question (can take a minute on CPU)
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

OK, BAD, WARN = "  [ OK ]", "  [FAIL]", "  [WARN]"
problems: list[str] = []


def fail(msg: str, fix: str):
    print(f"{BAD} {msg}")
    print(f"         fix: {fix}")
    problems.append(msg)


print("\nBasketBee doctor\n")

# 1. Python environment -----------------------------------------------------------
in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
if in_venv:
    print(f"{OK} virtual environment is active")
else:
    fail("virtual environment is NOT active", r"run  venv\Scripts\activate  (Mac/Linux: source venv/bin/activate)")

missing = []
for pkg in ("fastapi", "sqlalchemy", "sklearn", "langchain", "langgraph", "langchain_ollama", "streamlit", "httpx"):
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)
if missing:
    fail(f"missing packages: {', '.join(missing)}", "python -m pip install -r requirements.txt")
else:
    print(f"{OK} all packages installed")
    from backend.core.config import settings   # needs pydantic-settings

    # 2. Data -------------------------------------------------------------------------
    if not (ROOT / "data" / "seed" / "products.csv").exists():
        fail("seed data not generated", "python data/generate_data.py")
    elif not (ROOT / "basketbee.db").exists():
        fail("database not created", "python -m backend.db.seed")
    else:
        try:
            from sqlalchemy import func, select
            from backend.db.database import SessionLocal
            from backend.db.models import Product
            with SessionLocal() as db:
                n = db.scalar(select(func.count()).select_from(Product))
            print(f"{OK} database has {n} products") if n == 60 else \
                fail(f"database has {n} products (expected 60)", "python -m backend.db.seed")
        except Exception as e:  # noqa: BLE001
            fail(f"database can't be read ({e.__class__.__name__})", "python -m backend.db.seed")

    # 3. LLM provider -----------------------------------------------------------------
    import httpx
    model = settings.llm_model
    print(f"       LLM configured in .env: {settings.llm_provider} / {model}")
    if settings.llm_provider == "ollama":
        try:
            tags = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5).json()
            print(f"{OK} Ollama is running at {settings.ollama_base_url}")
            names = {m["name"] for m in tags.get("models", [])}
            wanted = {model, f"{model}:latest"}
            if names & wanted:
                print(f"{OK} model '{model}' is downloaded")
            else:
                fail(f"model '{model}' is not downloaded (you have: {', '.join(sorted(names)) or 'none'})",
                     f"ollama pull {model}   or set LLM_MODEL in .env to one you have")
        except httpx.HTTPError:
            fail(f"Ollama is NOT running at {settings.ollama_base_url}",
                 "open the Ollama app from the Start menu (or run  ollama serve  in a new terminal)")
    elif settings.llm_provider == "groq":
        try:
            import langchain_openai  # noqa: F401
        except ImportError:
            fail("Groq needs the langchain-openai package", "python -m pip install langchain-openai")
        if not settings.groq_api_key:
            fail("LLM_PROVIDER=groq but GROQ_API_KEY is empty",
                 "get a free key at console.groq.com and add GROQ_API_KEY=... to .env")
        else:
            try:
                r = httpx.get("https://api.groq.com/openai/v1/models", timeout=10,
                              headers={"Authorization": f"Bearer {settings.groq_api_key}"})
                if r.status_code == 401:
                    fail("Groq rejected the API key", "copy the key again from console.groq.com into .env")
                elif r.status_code == 200:
                    ids = {m["id"] for m in r.json().get("data", [])}
                    if model in ids:
                        print(f"{OK} Groq key works and model '{model}' is available")
                    else:
                        fail(f"Groq doesn't offer '{model}'",
                             "set LLM_MODEL to one of: " + ", ".join(sorted(i for i in ids if "whisper" not in i)[:8]))
                else:
                    fail(f"Groq answered HTTP {r.status_code}", "try again in a minute (rate limit?)")
            except httpx.HTTPError:
                fail("can't reach api.groq.com", "check your internet connection")
    elif settings.llm_provider == "openai" and not settings.openai_api_key:
        fail("LLM_PROVIDER=openai but OPENAI_API_KEY is empty", "add OPENAI_API_KEY=... to .env")
    elif settings.llm_provider == "gemini" and not settings.google_api_key:
        fail("LLM_PROVIDER=gemini but GOOGLE_API_KEY is empty", "add GOOGLE_API_KEY=... to .env")

    # 4. Optional: a real round trip with tool calling ------------------------------
    if "--llm" in sys.argv and not problems:
        print("       asking the model a test question (first run loads it; can take a minute)...")
        try:
            from langchain_core.messages import HumanMessage
            from backend.agents.llm import get_llm
            from backend.agents.tools import get_stock_level
            t0 = time.perf_counter()
            reply = get_llm().bind_tools([get_stock_level]).invoke(
                [HumanMessage("How much ghee do we have in the south warehouse? Use the tool.")])
            secs = time.perf_counter() - t0
            if reply.tool_calls:
                print(f"{OK} model called a tool correctly ({secs:.0f}s): {reply.tool_calls[0]['name']}")
            else:
                print(f"{WARN} model answered without calling a tool ({secs:.0f}s). "
                      f"Agents still work (BasketBee rescues text tool calls), but a bigger model is better.")
        except Exception as e:  # noqa: BLE001
            text = str(e)
            if any(w in text.lower() for w in ("does not exist", "model_not_found", "decommissioned")):
                fail(f"{settings.llm_provider} doesn't offer the model '{model}' (it may have been retired)",
                     "for Groq set  LLM_MODEL=openai/gpt-oss-120b  in .env (list: console.groq.com/docs/models)")
            elif any(w in text for w in ("bad_alloc", "out of memory", "requires more system memory",
                                         "unable to allocate")):
                fail("not enough free RAM to load the model",
                     "close Chrome/other apps and quit+reopen Ollama; or set OLLAMA_NUM_CTX=4096 in .env; "
                     "or use a smaller model: ollama pull llama3.2  then LLM_MODEL=llama3.2 in .env")
            else:
                fail(f"model call failed: {e.__class__.__name__}: {text[:150]}",
                     "check the Ollama window for errors, or try a smaller model (llama3.2)")

    # 5. Servers (information only) ----------------------------------------------------
    for name, url, how in (("API", "http://127.0.0.1:8000/api/health", "python run.py"),
                           ("app", "http://127.0.0.1:8501", "python run.py")):
        try:
            httpx.get(url, timeout=3)
            print(f"{OK} {name} is running ({url.rsplit('/', 2)[0] if name == 'API' else url})")
        except httpx.HTTPError:
            print(f"       {name} is not running right now (start it with: {how})")

print("\nAll good! Start BasketBee with:  python run.py\n" if not problems
      else f"\n{len(problems)} problem(s) found. Fix them in order, then run  python doctor.py  again.\n")
sys.exit(1 if problems else 0)