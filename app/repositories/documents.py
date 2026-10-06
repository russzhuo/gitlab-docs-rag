from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Document


def vector_search(db: Session, qvec: list, n: int) -> list[tuple[Document, float]]:
    dist = Document.embedding.cosine_distance(qvec)
    stmt = select(Document, (1 - dist).label("score")).order_by(dist).limit(n)
    return [(row[0], float(row[1])) for row in db.execute(stmt).all()]


def text_search(db: Session, qvec: list, question: str, n: int) -> list[tuple[Document, float]]:
    query = func.websearch_to_tsquery("english", question)
    dist = Document.embedding.cosine_distance(qvec)
    stmt = (
        select(Document, (1 - dist).label("score"))
        .where(Document.fts.op("@@")(query))
        .order_by(func.ts_rank(Document.fts, query).desc())
        .limit(n)
    )
    return [(row[0], float(row[1])) for row in db.execute(stmt).all()]
