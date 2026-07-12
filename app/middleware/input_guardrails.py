import re
import time
from dataclasses import dataclass, field
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.retrieval import RetrievalRequest

logger = get_logger(__name__)

try:
    import redis.asyncio as redis_async
except ImportError:  # pragma: no cover - dependency is optional at import time.
    redis_async = None


PROMPT_INJECTION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bignore\s+(all\s+)?(previous|prior|above|system|developer)\s+instructions\b",
        r"\bdisregard\s+(previous|prior|above|system|developer)\s+instructions\b",
        r"\breveal\s+(the\s+)?(system|developer)\s+prompt\b",
        r"\bshow\s+(me\s+)?(the\s+)?(system|developer)\s+prompt\b",
        r"\bprint\s+(the\s+)?hidden\s+(prompt|instructions)\b",
        r"\bjailbreak\b",
        r"\bDAN\s+mode\b",
        r"\boverride\s+(your\s+)?(instructions|policies|guardrails)\b",
        r"\btool\s+call\s+payload\b",
    ]
]

UNSAFE_REQUEST_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\boverride\s+SLA\b",
        r"\bsuppress\s+(notification|alert|escalation)\b",
        r"\bclose\s+(the\s+)?ticket\s+without\s+(a\s+)?fix\b",
        r"\bbypass\s+(escalation|approval|RBAC|authorization|auth)\b",
        r"\bdisable\s+(audit|logging|alerts|guardrails)\b",
        r"\bdelete\s+(audit\s+)?logs?\b",
        r"\bexfiltrate\b",
        r"\baccess\s+another\s+customer\b",
        r"\bignore\s+RBAC\b",
        r"\bthe\s+document\s+says\s+ignore\b",
        r"\bfollow\s+instructions\s+in\s+the\s+document\s+instead\b",
        r"\bdocument\s+injection\b",
    ]
]

HARD_BLOCK_UNSAFE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\boverride\s+SLA\b",
        r"\bsuppress\s+(notification|alert|escalation)\b",
        r"\bbypass\s+(escalation|approval|RBAC|authorization|auth)\b",
        r"\bdisable\s+(audit|logging|alerts|guardrails)\b",
        r"\bdelete\s+(audit\s+)?logs?\b",
        r"\bexfiltrate\b",
        r"\baccess\s+another\s+customer\b",
        r"\bignore\s+RBAC\b",
        r"\bthe\s+document\s+says\s+ignore\b",
        r"\bfollow\s+instructions\s+in\s+the\s+document\s+instead\b",
        r"\bdocument\s+injection\b",
    ]
]

ESCALATION_SAFE_OPERATIONAL_ACTION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\b(mark|set|change|update)\b.{0,80}\b(incident|ticket|case|alert)\b.{0,80}\b(resolved|closed|done|fixed)\b",
        r"\b(mark|set|change|update)\b.{0,80}\b(unresolved|open|active)\b.{0,80}\b(resolved|closed|done|fixed)\b",
        r"\b(resolve|close)\b.{0,80}\b(incident|ticket|case|alert)\b",
        r"\b(incident|ticket|case|alert)\b.{0,80}\b(resolve|close|mark\s+resolved|mark\s+closed)\b",
    ]
]

HIGH_RISK_OPERATIONAL_TERMS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bdata\s+loss\b",
        r"\bsecurity\s+(breach|vulnerability|incident|exposure)\b",
        r"\bcredential(s)?\s+(breach|leak|exposed|compromised)\b",
        r"\bproduction\s+(outage|impact|down)\b",
        r"\bcritical\s+(incident|alert|outage)\b",
    ]
]

CHITCHAT_INPUT_PATTERNS = [
    re.compile(pattern)
    for pattern in [
        r"^\s*(hi|hello|hey|hiya|yo|good\s+(morning|afternoon|evening))[\s!.?]*$",
        r"^\s*(hi|hello|hey)[\s,!.]*(eris|there)?[\s,!.]*(how\s+are\s+you|how's\s+it\s+going)?[\s!.?]*$",
        r"^\s*(how\s+are\s+you|how's\s+it\s+going|how\s+do\s+you\s+do)[\s!.?]*$",
        r"^\s*(thanks|thank\s+you|thank\s+you\s+so\s+much|thx|appreciate\s+it)[\s!.?]*$",
        r"^\s*(bye|goodbye|see\s+you|talk\s+later|take\s+care)[\s!.?]*$",
        r"^\s*(ok|okay|k|cool|great|nice|got\s+it|understood|yes|no|test|testing)[\s!.?]*$",
    ]
]

SENSITIVE_INPUT_PATTERNS = [
    (
        "email",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "[REDACTED_EMAIL]",
    ),
    (
        "phone",
        re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)"),
        "[REDACTED_PHONE]",
    ),
    (
        "secret",
        re.compile(
            r"(?i)\b(?:api[\s_-]?key|secret|token|password|client[\s_-]?secret)\s*[:=]\s*['\"]?[A-Za-z0-9._\-]{8,}"
        ),
        "[REDACTED_SECRET]",
    ),
    (
        "bearer_token",
        re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{10,}"),
        "[REDACTED_SECRET]",
    ),
]

_MEMORY_RATE_LIMITS: dict[str, tuple[int, float]] = {}
_REDIS_CLIENT: Any = None


class GuardrailViolation(Exception):
    """Raised when a request fails an input guardrail."""

    def __init__(
        self,
        error: str,
        detail: str,
        status_code: int = 400,
        retry_after: int | None = None,
    ) -> None:
        super().__init__(detail)
        self.error = error
        self.detail = detail
        self.status_code = status_code
        self.retry_after = retry_after


@dataclass
class InputGuardrailResult:
    sanitized_question: str
    metadata: dict[str, Any] = field(default_factory=dict)
    guardrail_flags: list[dict[str, Any]] = field(default_factory=list)
    redactions: dict[str, int] = field(default_factory=dict)


def _flag(name: str, passed: bool, reason: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "passed": passed,
        "severity": "low" if passed else "high",
        "reason": reason,
        "metadata": metadata or {},
    }


def _first_match(patterns: list[re.Pattern[str]], text: str) -> str | None:
    for pattern in patterns:
        if pattern.search(text):
            return pattern.pattern
    return None


def _safe_escalation_action_match(text: str) -> str | None:
    action_match = _first_match(ESCALATION_SAFE_OPERATIONAL_ACTION_PATTERNS, text)
    if not action_match:
        return None
    high_risk_match = _first_match(HIGH_RISK_OPERATIONAL_TERMS, text)
    return high_risk_match or action_match


def _is_chitchat_input(text: str) -> bool:
    normalized_text = " ".join(text.lower().strip().split())
    return bool(normalized_text and _first_match(CHITCHAT_INPUT_PATTERNS, normalized_text))


def _scrub_sensitive_input(text: str) -> tuple[str, dict[str, int]]:
    redactions: dict[str, int] = {}
    sanitized = text
    for name, pattern, replacement in SENSITIVE_INPUT_PATTERNS:
        sanitized, count = pattern.subn(replacement, sanitized)
        if count:
            redactions[name] = count
    return sanitized, redactions


def _rate_limit_identity(request: RetrievalRequest) -> tuple[str, str]:
    metadata = dict(request.metadata or {})
    subject = str(
        metadata.get("jwt_sub")
        or metadata.get("sub")
        or request.user_id
        or metadata.get("user_id")
        or "anonymous"
    )
    role = str(metadata.get("user_role") or metadata.get("role") or "anonymous").lower()
    return subject[:128], role[:64]


def _rate_limit_for_role(role: str) -> int:
    if "admin" in role:
        return settings.rate_limit_admin_limit
    if "manager" in role or "l2" in role or "l3" in role:
        return settings.rate_limit_manager_limit
    if "support" in role or "agent" in role:
        return settings.rate_limit_support_agent_limit
    return settings.rate_limit_default_limit


def _get_redis_client() -> Any:
    global _REDIS_CLIENT
    if not settings.redis_url or redis_async is None:
        return None
    if _REDIS_CLIENT is None:
        _REDIS_CLIENT = redis_async.Redis.from_url(settings.redis_url, decode_responses=True)
    return _REDIS_CLIENT


def _cleanup_memory_rate_limits(now: float) -> None:
    stale_keys = [key for key, (_, reset_at) in _MEMORY_RATE_LIMITS.items() if now >= reset_at]
    for key in stale_keys:
        _MEMORY_RATE_LIMITS.pop(key, None)


async def _check_rate_limit(request: RetrievalRequest, request_id: str) -> dict[str, Any]:
    subject, role = _rate_limit_identity(request)
    limit = _rate_limit_for_role(role)
    window = max(1, settings.rate_limit_window_seconds)
    key = f"rate_limit:support_query:{role}:{subject}"
    redis_client = _get_redis_client()
    require_distributed_limiter = settings.require_redis_for_rate_limiting_non_local and not settings.is_local_env

    if require_distributed_limiter and redis_client is None:
        logger.error(
            "rate limit backend unavailable in non-local environment request_id=%s app_env=%s",
            request_id,
            settings.app_env,
        )
        raise GuardrailViolation(
            "rate_limit_backend_unavailable",
            "Rate-limit backend is unavailable. Please retry shortly.",
            status_code=503,
            retry_after=5,
        )

    if redis_client is not None:
        try:
            count = int(await redis_client.incr(key))
            if count == 1:
                await redis_client.expire(key, window)
            ttl = int(await redis_client.ttl(key))
            retry_after = ttl if ttl > 0 else window
            if count > limit:
                logger.warning(
                    "input guardrail rate limit blocked request_id=%s role=%s subject=%s",
                    request_id,
                    role,
                    subject,
                )
                raise GuardrailViolation(
                    "rate_limit_exceeded",
                    "Too many retrieval requests for this user or role.",
                    status_code=429,
                    retry_after=retry_after,
                )
            return {"backend": "redis", "limit": limit, "remaining": max(0, limit - count), "window_seconds": window}
        except GuardrailViolation:
            raise
        except Exception:
            logger.exception("redis rate limit check failed; falling back to in-memory limiter")
            if require_distributed_limiter:
                raise GuardrailViolation(
                    "rate_limit_backend_unavailable",
                    "Rate-limit backend is unavailable. Please retry shortly.",
                    status_code=503,
                    retry_after=5,
                )

    now = time.time()
    _cleanup_memory_rate_limits(now)
    count, reset_at = _MEMORY_RATE_LIMITS.get(key, (0, now + window))
    if now >= reset_at:
        count, reset_at = 0, now + window
    count += 1
    _MEMORY_RATE_LIMITS[key] = (count, reset_at)
    retry_after = max(1, int(reset_at - now))
    if count > limit:
        logger.warning(
            "input guardrail in-memory rate limit blocked request_id=%s role=%s subject=%s",
            request_id,
            role,
            subject,
        )
        raise GuardrailViolation(
            "rate_limit_exceeded",
            "Too many retrieval requests for this user or role.",
            status_code=429,
            retry_after=retry_after,
        )
    return {"backend": "memory", "limit": limit, "remaining": max(0, limit - count), "window_seconds": window}


async def apply_input_guardrails(request: RetrievalRequest, request_id: str) -> InputGuardrailResult:
    """Validate, scrub, and rate-limit a retrieval request before agent execution."""

    flags: list[dict[str, Any]] = []
    query = request.question.strip()
    min_length = settings.input_guardrail_min_query_length
    max_length = settings.input_guardrail_max_query_length
    is_chitchat = _is_chitchat_input(query)

    if not min_length <= len(query) <= max_length and not (is_chitchat and 0 < len(query) <= max_length):
        flags.append(_flag("ig_1_pydantic_input_validation", False, "Query length is outside allowed bounds."))
        raise GuardrailViolation(
            "invalid_input",
            f"Question must be between {min_length} and {max_length} characters.",
            status_code=400,
        )
    flags.append(
        _flag(
            "ig_1_pydantic_input_validation",
            True,
            "Request schema and query length passed."
            if len(query) >= min_length
            else "Short chitchat input allowed before support routing.",
            {"chitchat_precheck": is_chitchat},
        )
    )

    injection_match = _first_match(PROMPT_INJECTION_PATTERNS, query)
    if injection_match:
        flags.append(
            _flag(
                "ig_2_prompt_injection_detection",
                False,
                "Prompt injection pattern detected.",
                {"pattern": injection_match},
            )
        )
        logger.warning("input guardrail blocked prompt injection request_id=%s", request_id)
        raise GuardrailViolation(
            "prompt_injection_detected",
            "Request contains adversarial prompt-injection instructions.",
            status_code=400,
        )
    flags.append(_flag("ig_2_prompt_injection_detection", True, "No prompt injection pattern detected."))

    safe_escalation_match = _safe_escalation_action_match(query)
    unsafe_match = _first_match(UNSAFE_REQUEST_PATTERNS, query)
    hard_block_match = _first_match(HARD_BLOCK_UNSAFE_PATTERNS, query)
    if hard_block_match:
        flags.append(
            _flag(
                "ig_5_unsafe_request_document_injection",
                False,
                "Unsafe support action or document-injection pattern detected.",
                {"pattern": hard_block_match},
            )
        )
        logger.warning(
            "input guardrail blocked unsafe request request_id=%s pattern=%s",
            request_id,
            hard_block_match,
        )
        raise GuardrailViolation(
            "unsafe_request",
            "Request asks for an unsafe action or contains document-injection instructions.",
            status_code=400,
        )

    if unsafe_match:
        if safe_escalation_match:
            flags.append(
                _flag(
                    "ig_5_unsafe_request_document_injection",
                    True,
                    "Unsafe operational mutation request converted to human escalation path.",
                    {
                        "pattern": unsafe_match,
                        "safe_escalation_pattern": safe_escalation_match,
                        "manual_action_required": True,
                    },
                )
            )
        else:
            flags.append(
                _flag(
                    "ig_5_unsafe_request_document_injection",
                    False,
                    "Unsafe support action or document-injection pattern detected.",
                    {"pattern": unsafe_match},
                )
            )
            logger.warning(
                "input guardrail blocked unsafe request request_id=%s pattern=%s safe_escalation=%s",
                request_id,
                unsafe_match,
                safe_escalation_match,
            )
            raise GuardrailViolation(
                "unsafe_request",
                "Request asks for an unsafe action or contains document-injection instructions.",
                status_code=400,
            )
    elif safe_escalation_match:
        flags.append(
            _flag(
                "ig_5_unsafe_request_document_injection",
                True,
                "Operational state-change request allowed only as manual escalation.",
                {"pattern": safe_escalation_match, "manual_action_required": True},
            )
        )
    else:
        flags.append(_flag("ig_5_unsafe_request_document_injection", True, "No unsafe request pattern detected."))

    sanitized_query, redactions = _scrub_sensitive_input(query)
    flags.append(
        _flag(
            "ig_3_pii_detection",
            True,
            "Sensitive input was scrubbed." if redactions else "No sensitive input detected.",
            {"redactions": redactions},
        )
    )
    if redactions:
        logger.info("input guardrail scrubbed sensitive input request_id=%s redaction_types=%s", request_id, list(redactions))

    rate_limit_metadata = await _check_rate_limit(request, request_id)
    flags.append(_flag("ig_4_rate_limit", True, "Rate limit check passed.", rate_limit_metadata))

    metadata = dict(request.metadata or {})
    metadata["request_id"] = request_id
    if safe_escalation_match:
        metadata["manual_action_required"] = True
        metadata["forced_route_decision"] = "high_risk"
        metadata["guardrail_escalation_reason"] = (
            "Request asks for an operational state change. ERIS may prepare a human escalation, "
            "but must not perform the change automatically."
        )
    metadata["input_guardrails"] = {
        "flags": flags,
        "redactions": redactions,
        "sanitized": bool(redactions),
        "rate_limit": rate_limit_metadata,
    }
    return InputGuardrailResult(
        sanitized_question=sanitized_query,
        metadata=metadata,
        guardrail_flags=flags,
        redactions=redactions,
    )
