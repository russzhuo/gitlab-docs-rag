from __future__ import annotations

from sqlalchemy.orm import Session

from app.repositories import history as hist_repo


def list_conversations(db: Session, user_id: int) -> list[dict]:
    return [
        {"id": c.id, "title": c.title, "updated_at": c.updated_at.isoformat()}
        for c in hist_repo.list_conversations(db, user_id)
    ]


def list_messages(db: Session, user_id: int, conversation_id: int) -> list[dict] | None:
    conv = hist_repo.get_conversation(db, conversation_id, user_id)
    if conv is None:
        return None
    return [
        {
            "role": m.role,
            "content": m.content,
            "citations": m.citations,
            "refused": m.refused,
            "created_at": m.created_at.isoformat(),
        }
        for m in hist_repo.list_messages(db, conversation_id)
    ]


def persist_exchange(db: Session, user_id: int, conversation_id: int | None,
                     question: str, answer: str, citations: list, refused: bool) -> int:
    if conversation_id is None:
        title = question.strip().replace("\n", " ")[:60] or "New chat"
        conv = hist_repo.create_conversation(db, user_id, title)
        conversation_id = conv.id
    elif hist_repo.get_conversation(db, conversation_id, user_id) is None:
        raise PermissionError("Conversation not found")
    hist_repo.add_message(db, conversation_id, "user", question, [], False)
    hist_repo.add_message(db, conversation_id, "assistant", answer, citations, refused)
    hist_repo.touch_conversation(db, conversation_id)
    db.commit()
    return conversation_id
