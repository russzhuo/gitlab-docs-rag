from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import require_user
from app.services import history_service
from starlette import status

router = APIRouter(tags=["history"])


@router.get("/conversations")
def conversations(user: dict = Depends(require_user), db: Session = Depends(get_db)):
    return history_service.list_conversations(db, user["id"])


@router.get("/conversations/{conversation_id}/messages")
def messages(conversation_id: int, user: dict = Depends(require_user),
             db: Session = Depends(get_db)):
    msgs = history_service.list_messages(db, user["id"], conversation_id)
    if msgs is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return msgs
