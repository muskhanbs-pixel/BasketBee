"""
Shared test fixtures.

Tests NEVER use your real basketbee.db (which changes as you approve orders).
Instead, once per test run, a pristine database is built from data/seed/*.csv:

  pristine_db  built once; the whole API reads it during tests
  temp_db      a throw-away copy of the pristine database for tests that write
               (recommendations, approvals, purchase orders)
"""

import shutil

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.core.config import PROJECT_ROOT, settings
from backend.db.database import get_db
from backend.db.seed import seed
from backend.main import app

settings.warmup_on_startup = False   # no background forecasting during tests


def _make_override(path):
    engine = create_engine(f"sqlite:///{path.as_posix()}", connect_args={"check_same_thread": False})
    Session = sessionmaker(bind=engine, autoflush=False)

    def override_get_db():
        db = Session()
        try:
            yield db
        finally:
            db.close()
    return engine, Session, override_get_db


@pytest.fixture(scope="session")
def pristine_db(tmp_path_factory):
    if not (PROJECT_ROOT / "data" / "seed" / "products.csv").exists():
        pytest.exit("Run `python data/generate_data.py` first", returncode=1)
    path = tmp_path_factory.mktemp("pristine") / "basketbee.db"
    engine = create_engine(f"sqlite:///{path.as_posix()}")
    seed(engine, verbose=False)
    engine.dispose()
    return path


@pytest.fixture(scope="session", autouse=True)
def use_pristine_db(pristine_db, tmp_path_factory):
    """Every test talks to a fresh copy of the seed data, never the real database."""
    copy = tmp_path_factory.mktemp("shared") / "basketbee.db"
    shutil.copy(pristine_db, copy)
    engine, _, override = _make_override(copy)
    app.dependency_overrides[get_db] = override
    yield copy
    app.dependency_overrides.pop(get_db, None)
    engine.dispose()


@pytest.fixture
def temp_db(pristine_db, tmp_path):
    copy = tmp_path / "test.db"
    shutil.copy(pristine_db, copy)
    engine, Session, override = _make_override(copy)
    previous = app.dependency_overrides.get(get_db)
    app.dependency_overrides[get_db] = override
    yield Session
    if previous:
        app.dependency_overrides[get_db] = previous
    else:
        app.dependency_overrides.pop(get_db, None)
    engine.dispose()