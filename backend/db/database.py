"""
BasketBee - database connection (M1 Data layer).

Provides:
    engine        - the SQLAlchemy connection to the database
    SessionLocal  - creates database sessions
    Base          - parent class for all table models (see models.py)
    get_db()      - FastAPI dependency that opens/closes a session per request

Switching SQLite -> PostgreSQL later only needs a new DATABASE_URL in .env:
    DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/basketbee
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from backend.core.config import settings

# SQLite needs this flag because FastAPI may use the connection from different threads
connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}

engine = create_engine(settings.database_url, connect_args=connect_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """All table models inherit from this."""


def get_db():
    """Give one FastAPI request its own session, and always close it afterwards."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()