from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from app.schemas.history import Citation


class ChatIn(BaseModel):
    question: str = Field(min_length=3)
    conversation_id: int | None = None


# --- SSE stream events (POST /chat/stream) ---
# Centralized here so each event's type + shape lives in one place. The service
# yields them and the router forwards them (via .model_dump_json()) and reads
# them back (via .type / attribute access).


class MetaEvent(BaseModel):
    type: Literal["meta"] = "meta"
    citations: list[Citation]


class TokenEvent(BaseModel):
    type: Literal["token"] = "token"
    text: str


class DoneEvent(BaseModel):
    type: Literal["done"] = "done"


class RefusedEvent(BaseModel):
    type: Literal["refused"] = "refused"
    answer: str
    citations: list[Citation] = Field(default_factory=list)


class ConversationEvent(BaseModel):
    type: Literal["conversation"] = "conversation"
    id: int


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    message: str


StreamEvent = Annotated[
    Union[
        MetaEvent,
        TokenEvent,
        DoneEvent,
        RefusedEvent,
        ConversationEvent,
        ErrorEvent,
    ],
    Field(discriminator="type"),
]
