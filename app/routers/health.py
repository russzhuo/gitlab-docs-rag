from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import func, select

from app.database import SessionLocal
from app.models import Document

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    try:
        with SessionLocal() as db:
            n = db.scalar(select(func.count(Document.id)))
        return {"ok": True, "chunks": n}
    except Exception as e:
        return {"ok": False, "error": str(e)}
