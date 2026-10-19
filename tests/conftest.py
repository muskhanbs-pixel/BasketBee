"""
Shared test fixtures.

temp_db: a throw-away COPY of basketbee.db. The API, the agent tools and the
supervisor all use it during the test, so your real database is never changed.
"""

import shutil

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.core.config import PROJECT_ROOT
from backend.db.database import get_db
from backend.main import app


@pytest.fixture
def temp_db(tmp_path):
    real_db = PROJECT_ROOT / "basketbee.db"
    if not real_db.exists():
        pytest.skip("Run `python -m backend.db.seed` first")
    copy = tmp_path / "test.db"
    shutil.copy(real_db, copy)
    engine = create_engine(f"sqlite:///{copy.as_posix()}", connect_args={"check_same_thread": False})
    TestSession = sessionmaker(bind=engine, autoflush=False)

    def override_get_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    yield TestSession
    app.dependency_overrides.pop(get_db, None)
    engine.dispose()