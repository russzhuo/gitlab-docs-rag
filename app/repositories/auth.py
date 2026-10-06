from __future__ import annotations

from datetime import datetime

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.models import AuthSession, User


def get_user_by_username(db: Session, username: str) -> User | None:
    return db.scalar(select(User).where(User.username == username))


def create_user(db: Session, username: str, password_hash: str) -> User:
    user = User(username=username, password_hash=password_hash)
    db.add(user)
    db.flush()  # populate user.id without committing
    return user


def create_session(db: Session, user_id: int, token: str, expires_at: datetime) -> AuthSession:
    sess = AuthSession(user_id=user_id, token=token, expires_at=expires_at)
    db.add(sess)
    return sess


def delete_session_by_token(db: Session, token: str) -> None:
    sess = db.scalar(select(AuthSession).where(AuthSession.token == token))
    if sess is not None:
        db.delete(sess)


def get_user_by_token(db: Session, token: str) -> User | None:
    return db.scalar(
        select(User)
        .join(AuthSession, AuthSession.user_id == User.id)
        .where(AuthSession.token == token, AuthSession.expires_at > func.now())
    )

def delete_expired(db: Session, before: datetime, batch_size: int) -> int:
    result = db.execute(
        delete(AuthSession)
        .where(AuthSession.expires_at < before)
        .limit(batch_size)
    )
    return result.rowcount
