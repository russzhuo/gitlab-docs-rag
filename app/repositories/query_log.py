from __future__ import annotations

from sqlalchemy.orm import Session

from app.models import QueryLog


def add_query(db: Session, question: str, top_hits: list,
              best_score: float, grounded: bool, refused: bool, answer_preview: str) -> None:
    db.add(
        QueryLog(
            question=question,
            top_hits=top_hits,
            best_score=best_score,
            grounded=grounded,
            refused=refused,
            answer_preview=answer_preview,
        )
    )
