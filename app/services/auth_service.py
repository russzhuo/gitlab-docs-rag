from __future__ import annotations

import logging
import secrets
from datetime import datetime, timedelta, timezone

import bcrypt
from fastapi import HTTPException, Response
from sqlalchemy.orm import Session

from app.repositories import auth as auth_repo
from starlette import status

SESSION_COOKIE = "session"
SESSION_TTL_DAYS = 30
logger = logging.getLogger(__name__)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def _expiry() -> datetime:
    return datetime.now(timezone.utc) + timedelta(days=SESSION_TTL_DAYS)


def register(db: Session, username: str, password: str) -> tuple[dict, str]:
    username = username.strip()
    if not username:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username required")
    if auth_repo.get_user_by_username(db, username) is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username already taken")
    user = auth_repo.create_user(db, username, hash_password(password))
    token = _issue_token(db, user.id)
    return {"id": user.id, "username": user.username}, token


def login(db: Session, username: str, password: str) -> tuple[dict, str]:
    user = auth_repo.get_user_by_username(db, username.strip())
    if user is None or not verify_password(password, user.password_hash):
        logger.warning("failed login for %r", username)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")
    token = _issue_token(db, user.id)
    return {"id": user.id, "username": user.username}, token


def logout(db: Session, token: str) -> None:
    if token:
        auth_repo.delete_session_by_token(db, token)
        db.commit()


def get_user_by_token(db: Session, token: str) -> dict | None:
    user = auth_repo.get_user_by_token(db, token)
    if user is None:
        return None
    return {"id": user.id, "username": user.username}


def set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        httponly=True,
        samesite="lax",
        max_age=SESSION_TTL_DAYS * 24 * 3600,
    )


def _issue_token(db: Session, user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    auth_repo.create_session(db, user_id, token, _expiry())
    db.commit()
    return token

def remove_expired(db: Session, batch_size: int = 500) -> int:
    now = datetime.now(timezone.utc)
    total = 0
    while True:
        deleted = auth_repo.delete_expired(db, now, batch_size)
        db.commit()
        total += deleted
        if deleted < batch_size:
            break
    return total