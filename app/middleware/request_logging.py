from __future__ import annotations

import time
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.core.logging import get_logger
from app.core.logging_context import reset_request_context, set_request_context

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Attach request correlation context and emit safe request lifecycle logs."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        request_id = request.headers.get("x-request-id") or str(uuid4())
        trace_id = request.headers.get("x-trace-id") or "-"
        request_token, trace_token = set_request_context(request_id, trace_id)
        started = time.perf_counter()
        logger.info(
            "http_request_started",
            extra={
                "method": request.method,
                "path": request.url.path,
                "client_host": request.client.host if request.client else None,
            },
        )
        try:
            response = await call_next(request)
        except Exception:
            logger.error(
                "http_request_failed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": int((time.perf_counter() - started) * 1000),
                },
                exc_info=True,
            )
            reset_request_context(request_token, trace_token)
            raise

        response.headers["X-Request-ID"] = request_id
        logger.info(
            "http_request_completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": int((time.perf_counter() - started) * 1000),
            },
        )
        reset_request_context(request_token, trace_token)
        return response
