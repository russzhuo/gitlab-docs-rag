from __future__ import annotations

from datetime import datetime, timezone
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.staticfiles import StaticFiles

from apscheduler.schedulers.background import BackgroundScheduler
from app.services import auth_service

from app.config import FRONTEND_ORIGIN
from app.database import SessionLocal, init_db
from app.routers import auth, chat, health, history
from app.schemas import (
    Citation,
    ConversationEvent,
    ConversationOut,
    DoneEvent,
    ErrorEvent,
    MessageOut,
    MetaEvent,
    RefusedEvent,
    TokenEvent,
    UserOut,
)

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)

scheduler = BackgroundScheduler()

def _cleanup_job():
    try:
        db = SessionLocal()
        try:
            count = auth_service.remove_expired(db)
        finally:
            db.close()
        if count:
            logger.info("removed %d expired sessions", count)
    except Exception:
        logger.exception("session cleanup failed")

@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("starting up")
    try:
        init_db()
    except Exception:
        logger.exception("init_db failed at startup")
    else:
        logger.info("database schema ready")

    scheduler.add_job(
        _cleanup_job,
        trigger="interval",
        next_run_time=datetime.now(timezone.utc),
        hours=1,
        id="session_cleanup",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    scheduler.start()
    logger.info("scheduler started")

    yield

    scheduler.shutdown(wait=False)
    logger.info("scheduler stopped")


app = FastAPI(title="Handbook RAG", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(history.router)
app.include_router(chat.router)
app.include_router(health.router)

app.mount("/static", StaticFiles(directory="app/static"), name="static")


# Expose the public API schemas as OpenAPI components so the frontend can
# generate matching TypeScript types. The SSE events are yielded by a generator
# (not a response_model) and the REST routes don't declare response models, so
# none of these would be registered by FastAPI automatically.
PUBLIC_API_MODELS = (
    MetaEvent,
    TokenEvent,
    DoneEvent,
    RefusedEvent,
    ConversationEvent,
    ErrorEvent,
    Citation,
    ConversationOut,
    MessageOut,
    UserOut,
)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        routes=app.routes,
    )
    schemas = openapi_schema.setdefault("components", {}).setdefault("schemas", {})
    for model in PUBLIC_API_MODELS:
        schema = model.model_json_schema(ref_template="#/components/schemas/{model}")
        # `ref_template` externalizes nested models, so the local $defs copy is
        # redundant (and confuses some generators).
        schema.pop("$defs", None)
        # The event discriminator field has a default, so Pydantic omits it from
        # `required`; force it back so generated TS unions stay discriminated.
        if "type" in schema.get("properties", {}):
            required = schema.setdefault("required", [])
            if "type" not in required:
                required.append("type")
        schemas.setdefault(model.__name__, schema)
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi
