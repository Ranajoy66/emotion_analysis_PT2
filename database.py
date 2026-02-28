# database.py

import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv()

MYSQL_USER = "root"
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD")
MYSQL_HOST = "localhost"
MYSQL_DB = "emotion_db"

# Step 1: Create database if not exists (RUNS ONLY ONCE)
def create_database_if_not_exists():
    temp_engine = create_engine(
        f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}"
    )

    with temp_engine.connect() as conn:
        conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {MYSQL_DB}"))
        conn.commit()

    temp_engine.dispose()


# Call this function immediately when file loads
create_database_if_not_exists()

# Step 2: Create main engine
DATABASE_URL = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}/{MYSQL_DB}"
engine = create_engine(DATABASE_URL, echo=False)

SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()
