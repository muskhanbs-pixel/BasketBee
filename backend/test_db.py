from database import engine
from sqlalchemy import text

with engine.connect() as connection:
    result = connection.execute(text("SELECT sqlite_version()"))
    print("SQLite connected successfully!")
    print("SQLite version:", result.scalar())