from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, Computed, DateTime, Float, Integer, Text, func
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column
from pgvector.sqlalchemy import Vector

from app.config import EMBED_DIM
from app.database import Base


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    section: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(Text, nullable=False, default="")
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    preview: Mapped[str] = mapped_column(Text, nullable=False, default="")
    embedding: Mapped[list] = mapped_column(Vector(EMBED_DIM), nullable=False)
    fts: Mapped[str] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', content)", persisted=True)
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class QueryLog(Base):
    __tablename__ = "queries"

    id: Mapped[int] = mapped_column(primary_key=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    top_hits: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    best_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    grounded: Mapped[bool] = mapped_column(Boolean, nullable=False)
    refused: Mapped[bool] = mapped_column(Boolean, nullable=False)
    answer_preview: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
