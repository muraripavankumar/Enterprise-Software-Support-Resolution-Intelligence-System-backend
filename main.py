from contextlib import asynccontextmanager
import importlib
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from psycopg import connect

from app.api.routes.ingestion import router as ingestion_router
from app.api.routes.retrieval import router as retrieval_router
from app.api.routes.auth import router as auth_router
from app.api.routes.jira import router as jira_router
from app.agents.graph import close_support_graph_checkpointer
from app.core.config import settings
from app.core.langfuse import flush_langfuse
from app.core.logging import get_logger
from app.middleware.request_logging import RequestLoggingMiddleware
from app.services.tools.sql_tool import close_sql_tool, warm_sql_tool
from app.services.tools.vector_tool import ensure_retrieval_indexes

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        ensure_retrieval_indexes()
    except Exception:
        logger.exception("retrieval index preparation failed during startup")
    try:
        warm_sql_tool()
    except Exception:
        logger.exception("SQL tool warmup failed during startup")
    yield
    flush_langfuse()
    close_sql_tool()
    await close_support_graph_checkpointer()


app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.add_middleware(RequestLoggingMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allowed_methods,
    allow_headers=settings.cors_allowed_headers,
)


@app.get("/", summary="API root", description="Return a minimal health message for the ERIS API.")
def root():
    """Return a minimal API root message."""

    return {"message": "Enterprise RAG API Running"}


@app.get("/health", summary="Health check", description="Return a simple process health response.")
def health():
    """Return a simple process health response."""

    return {"status": "healthy"}


@app.get("/health/live", summary="Liveness check", description="Return whether the API process is alive.")
def health_live() -> dict[str, str]:
    """Return liveness status without dependency checks."""

    return {"status": "alive"}


def _check_database() -> tuple[bool, str]:
    try:
        with connect(settings.database_dsn, connect_timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return True, "ok"
    except Exception as exc:
        logger.exception("database readiness check failed")
        return False, str(exc)


async def _check_redis() -> tuple[bool, str]:
    if not settings.redis_url:
        return True, "not_configured"

    try:
        redis_async = importlib.import_module("redis.asyncio")
    except ImportError:  # pragma: no cover - dependency is optional at import time.
        return False, "redis package unavailable"

    client: Any = None
    try:
        client = redis_async.Redis.from_url(settings.redis_url, decode_responses=True)
        await client.ping()
        return True, "ok"
    except Exception as exc:
        logger.exception("redis readiness check failed")
        return False, str(exc)
    finally:
        if client is not None:
            await client.close()


@app.get("/health/ready", summary="Readiness check", description="Check database, Redis, and auth readiness.")
async def health_ready() -> dict[str, Any]:
    """Return readiness status for critical runtime dependencies."""

    checks: dict[str, dict[str, Any]] = {}

    db_ok, db_message = _check_database()
    checks["database"] = {"ok": db_ok, "detail": db_message}

    redis_ok, redis_message = await _check_redis()
    checks["redis"] = {"ok": redis_ok, "detail": redis_message}

    auth_ok = True
    auth_detail = "ok"
    if settings.enable_auth0:
        try:
            settings.validate_for_auth()
        except Exception as exc:
            auth_ok = False
            auth_detail = str(exc)
    checks["auth"] = {"ok": auth_ok, "detail": auth_detail}

    overall_ok = all(item["ok"] for item in checks.values())
    return {"status": "ready" if overall_ok else "degraded", "checks": checks}


app.include_router(ingestion_router, prefix=settings.api_v1_prefix)
app.include_router(retrieval_router, prefix=settings.api_v1_prefix)
app.include_router(auth_router, prefix=settings.api_v1_prefix)
app.include_router(jira_router, prefix=settings.api_v1_prefix)
