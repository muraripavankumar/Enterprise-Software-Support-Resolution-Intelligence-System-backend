from __future__ import annotations

from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="-")


def set_request_context(request_id: str, trace_id: str = "-") -> tuple[object, object]:
    """Set per-request logging context and return reset tokens."""

    request_token = request_id_var.set(request_id or "-")
    trace_token = trace_id_var.set(trace_id or "-")
    return request_token, trace_token


def reset_request_context(request_token: object, trace_token: object) -> None:
    """Reset per-request logging context after the wrapped request completes."""

    request_id_var.reset(request_token)  # type: ignore[arg-type]
    trace_id_var.reset(trace_token)  # type: ignore[arg-type]


def current_request_id() -> str:
    """Return the active request id for log enrichment."""

    return request_id_var.get()


def current_trace_id() -> str:
    """Return the active trace id for log enrichment."""

    return trace_id_var.get()
