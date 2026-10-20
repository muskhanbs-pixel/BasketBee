"""
BasketBee - start everything with one command.

    python run.py

Starts the API (http://127.0.0.1:8000) and the app (http://localhost:8501),
and stops both when you press Ctrl+C.
"""

import subprocess
import sys
import time
import urllib.request

API = [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000"]
APP = [sys.executable, "-m", "streamlit", "run", "frontend/app.py", "--server.port", "8501"]


def wait_for_api(seconds: int = 60) -> bool:
    for _ in range(seconds * 2):
        try:
            urllib.request.urlopen("http://127.0.0.1:8000/api/health", timeout=1)
            return True
        except Exception:   # noqa: BLE001
            time.sleep(0.5)
    return False


if __name__ == "__main__":
    print("Starting the BasketBee API...")
    api = subprocess.Popen(API)
    if not wait_for_api():
        print("The API did not start. Check the messages above.")
        api.terminate()
        sys.exit(1)
    print("API ready. Starting the app at http://localhost:8501 ...")
    app = subprocess.Popen(APP)
    try:
        app.wait()
    except KeyboardInterrupt:
        pass
    finally:
        for p in (app, api):
            p.terminate()
        print("BasketBee stopped.")