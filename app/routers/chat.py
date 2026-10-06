from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session

from app.database import SessionLocal, get_db
from app.dependencies import require_user
from app.schemas import ChatIn, ConversationEvent, ErrorEvent
from app.services import history_service, rag_service
from starlette import status

router = APIRouter(tags=["chat"])
logger = logging.getLogger(__name__)


@router.post("/chat")
def chat(body: ChatIn, user: dict = Depends(require_user), db: Session = Depends(get_db)):
    try:
        result = rag_service.answer(db, body.question)
    except Exception:
        logger.exception("answer failed")
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={
            "error": "answer failed",
            "answer": "",
            "refused": False,
            "citations": [],
            "hits": [],
        })
    logger.info("%s: %s", "refused" if result.get("refused") else "answered", body.question[:100])
    try:
        conversation_id = history_service.persist_exchange(
            db, user["id"], body.conversation_id, body.question,
            result["answer"], result["citations"], result["refused"],
        )
    except PermissionError:
        return JSONResponse(status_code=status.HTTP_403_FORBIDDEN, content={"error": "Conversation not found"})
    result["conversation_id"] = conversation_id
    return result


@router.post("/chat/stream")
def chat_stream(body: ChatIn, user: dict = Depends(require_user)):
    def gen():
        # Own the session for the lifetime of the stream (dependency teardown
        # timing is not reliable with StreamingResponse).
        db = SessionLocal()
        try:
            answer_text = ""
            citations: list = []
            refused = False
            for ev in rag_service.stream_answer(db, body.question):
                if ev.type == "meta":
                    citations = [c.model_dump() for c in ev.citations]
                elif ev.type == "token":
                    answer_text += ev.text
                elif ev.type == "refused":
                    refused = True
                    answer_text = ev.answer
                yield f"data: {ev.model_dump_json()}\n\n"
            try:
                conversation_id = history_service.persist_exchange(
                    db, user["id"], body.conversation_id, body.question,
                    answer_text, citations, refused,
                )
                # Emitted only after the exchange is committed, so the client
                # learns the (possibly newly created) conversation id and can
                # reconcile without racing the DB write.
                yield f"data: {ConversationEvent(id=conversation_id).model_dump_json()}\n\n"
            except Exception:
                logger.exception("persist failed")
        except Exception:
            logger.exception("stream failed")
            yield f"data: {ErrorEvent(message='stream failed').model_dump_json()}\n\n"
        finally:
            db.close()

    return StreamingResponse(gen(), media_type="text/event-stream")
