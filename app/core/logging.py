import logging
import json
import sys
from datetime import datetime, timezone

from app.core.config import settings
from app.core.logging_context import current_request_id, current_trace_id


def _configure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8", errors="replace")


class _RequestContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = current_request_id()
        if not hasattr(record, "trace_id"):
            record.trace_id = current_trace_id()
        return True


class _JsonFormatter(logging.Formatter):
    _reserved = set(logging.makeLogRecord({}).__dict__.keys()) | {"message", "asctime"}
    _sensitive = {"authorization", "access_token", "refresh_token", "password", "api_key", "token", "jwt"}

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
            "trace_id": getattr(record, "trace_id", "-"),
        }
        for key, value in record.__dict__.items():
            if key in self._reserved or key in payload:
                continue
            if any(sensitive in key.lower() for sensitive in self._sensitive):
                payload[key] = "[REDACTED]"
            elif isinstance(value, (str, int, float, bool)) or value is None:
                payload[key] = value
            else:
                payload[key] = str(value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True)


_LOGGING_CONFIGURED = False


def _configure_logging() -> None:
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    root = logging.getLogger()
    root.setLevel(getattr(logging, settings.log_level, logging.INFO))
    root.handlers.clear()

    handler = logging.StreamHandler()
    handler.addFilter(_RequestContextFilter())
    if settings.log_format == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] request_id=%(request_id)s trace_id=%(trace_id)s %(message)s"))

    root.addHandler(handler)
    _LOGGING_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    _configure_utf8_stdio()
    _configure_logging()
    return logging.getLogger(name)
