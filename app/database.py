from __future__ import annotations

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import DATABASE_URL

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def _import_models():
    from app import models  # noqa: F401  (register tables on Base.metadata)

def init_db() -> None:
    """Create the extension, tables, and indexes. Idempotent."""

    _import_models()

    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    Base.metadata.create_all(bind=engine)

    with engine.begin() as conn:
        # migration: documents.department -> section (older schema)
        if conn.execute(text(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = 'documents' AND column_name = 'department'"
        )).fetchone():
            conn.execute(text("ALTER TABLE documents RENAME COLUMN department TO section"))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS documents_embedding_hnsw_idx "
            "ON documents USING hnsw (embedding vector_cosine_ops)"
        ))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS documents_fts_idx ON documents USING gin (fts)"
        ))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS documents_section_idx ON documents (section)"
        ))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS sessions_token_idx ON sessions (token)"
        ))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS conversations_user_idx ON conversations (user_id)"
        ))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS messages_conversation_idx ON messages (conversation_id)"
        ))
