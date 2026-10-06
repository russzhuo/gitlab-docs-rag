from typing import Literal

from pydantic import BaseModel


class Citation(BaseModel):
    n: int
    title: str
    section: str
    source: str
    score: float


class ConversationOut(BaseModel):
    id: int
    title: str
    updated_at: str


class MessageOut(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    citations: list[Citation]
    refused: bool
    created_at: str
