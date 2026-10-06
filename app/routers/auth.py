from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import require_user
from app.schemas import AuthIn
from app.services import auth_service

router = APIRouter(prefix="/auth", tags=["auth"])
logger = logging.getLogger(__name__)


@router.post("/register")
def register(body: AuthIn, response: Response, db: Session = Depends(get_db)):
    user, token = auth_service.register(db, body.username, body.password)
    auth_service.set_session_cookie(response, token)
    logger.info("registered user %r", body.username)
    return user


@router.post("/login")
def login(body: AuthIn, response: Response, db: Session = Depends(get_db)):
    user, token = auth_service.login(db, body.username, body.password)
    auth_service.set_session_cookie(response, token)
    logger.info("user %r logged in", body.username)
    return user


@router.post("/logout")
def logout(request: Request, response: Response, db: Session = Depends(get_db)):
    token = request.cookies.get(auth_service.SESSION_COOKIE)
    auth_service.logout(db, token)
    response.delete_cookie(auth_service.SESSION_COOKIE)
    return {"ok": True}


@router.get("/me")
def me(user: dict = Depends(require_user)):
    return user
