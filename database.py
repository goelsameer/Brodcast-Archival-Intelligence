from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, declarative_base
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> None:
        return None

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://postgres.igugfmbitwzxawfnoehs:6BBqXxE2ABvG3kRi@aws-1-ap-northeast-1.pooler.supabase.com:5432/postgres",
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

Base = declarative_base()

#Currently the functions below are not being used as the database schema is managed through SQLAlchemy models and migrations.

def _ensure_caption_segment_columns() -> None:
    inspector = inspect(engine)
    if "caption_segments" not in inspector.get_table_names():
        return

    existing_columns = {column["name"] for column in inspector.get_columns("caption_segments")}
    if "searchable_text" not in existing_columns:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE caption_segments ADD COLUMN searchable_text TEXT"))

    if "embedding" not in existing_columns:
        with engine.begin() as connection:
            if engine.dialect.name == "postgresql":
                connection.execute(text("ALTER TABLE caption_segments ADD COLUMN embedding vector(384)"))
            else:
                connection.execute(text("ALTER TABLE caption_segments ADD COLUMN embedding TEXT"))


def _ensure_clip_record_columns() -> None:
    inspector = inspect(engine)
    if "clip_records" not in inspector.get_table_names():
        return

    existing_columns = {column["name"] for column in inspector.get_columns("clip_records")}
    columns_to_add = {
        "watchlist_item_id": "INTEGER",
        "matched_text": "VARCHAR(255)",
        "sentiment": "VARCHAR(20)",
        "context_text": "TEXT",
    }

    with engine.begin() as connection:
        for column_name, column_type in columns_to_add.items():
            if column_name not in existing_columns:
                connection.execute(text(f"ALTER TABLE clip_records ADD COLUMN {column_name} {column_type}"))


def _ensure_watchlist_item_columns() -> None:
    inspector = inspect(engine)
    if "watchlist_items" not in inspector.get_table_names():
        return

    existing_columns = {column["name"] for column in inspector.get_columns("watchlist_items")}
    columns_to_add = {
        "reference_image_path": "VARCHAR(1024)",
        "face_embedding": "vector(512)" if engine.dialect.name == "postgresql" else "TEXT",
    }

    with engine.begin() as connection:
        for column_name, column_type in columns_to_add.items():
            if column_name not in existing_columns:
                connection.execute(text(f"ALTER TABLE watchlist_items ADD COLUMN {column_name} {column_type}"))


def init_db() -> None:
    if engine.dialect.name == "postgresql":
        try:
            with engine.begin() as connection:
                connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        except Exception:
            pass
    Base.metadata.create_all(bind=engine)
    _ensure_caption_segment_columns()
    _ensure_clip_record_columns()
    _ensure_watchlist_item_columns()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
