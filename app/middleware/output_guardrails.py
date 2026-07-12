import re
from typing import Any

from app.core.logging import get_logger
from app.schemas.retrieval import RetrievalResponse

logger = get_logger(__name__)

SENSITIVE_OUTPUT_PATTERNS = [
    (
        "email",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "[REDACTED_EMAIL]",
    ),
    (
        "phone",
        re.compile(r"(?<![\w-])(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4})(?![\w-])"),
        "[REDACTED_PHONE]",
    ),
    (
        "ssn",
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "[REDACTED_SSN]",
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

UNSAFE_POLICY_ACTION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\boverride\s+SLA\b",
        r"\bsuppress\s+(notification|alert|escalation)\b",
        r"\bclose\s+(the\s+)?ticket\s+without\s+(a\s+)?fix\b",
        r"\bbypass\s+(escalation|approval|RBAC|authorization|auth)\b",
        r"\bdisable\s+(audit|logging|alerts|guardrails)\b",
        r"\bdelete\s+(audit\s+)?logs?\b",
    ]
]


def _redact_text(text: str) -> tuple[str, dict[str, int]]:
    redactions: dict[str, int] = {}
    sanitized = text
    for name, pattern, replacement in SENSITIVE_OUTPUT_PATTERNS:
        sanitized, count = pattern.subn(replacement, sanitized)
        if count:
            redactions[name] = redactions.get(name, 0) + count
    return sanitized, redactions


def _merge_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + value


def _sanitize_value(value: Any, redactions: dict[str, int]) -> Any:
    if isinstance(value, str):
        sanitized, counts = _redact_text(value)
        _merge_counts(redactions, counts)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_value(item, redactions) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_value(item, redactions) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize_value(item, redactions) for key, item in value.items()}
    return value


def _first_unsafe_policy_match(text: str) -> str | None:
    for pattern in UNSAFE_POLICY_ACTION_PATTERNS:
        if pattern.search(text):
            return pattern.pattern
    return None


def _derive_citations(response_data: dict[str, Any]) -> list[str]:
    citations = [str(citation) for citation in response_data.get("citations") or [] if str(citation).strip()]
    if citations:
        return sorted(set(citations))

    derived = []
    for node in response_data.get("source_nodes") or []:
        source_file = dict(node).get("source_file")
        if source_file:
            derived.append(str(source_file))
    return sorted(set(derived))


def _is_document_backed(response_data: dict[str, Any]) -> bool:
    if response_data.get("source_nodes"):
        return True
    try:
        return int(response_data.get("chunk_count") or 0) > 0
    except (TypeError, ValueError):
        return False


def _block_response(response_data: dict[str, Any], error: str, answer: str) -> RetrievalResponse:
    response_data["success"] = False
    response_data["answer"] = answer
    response_data["error"] = error
    return RetrievalResponse.model_validate(response_data)


def apply_output_guardrails(response: RetrievalResponse, request_id: str) -> RetrievalResponse:
    """Apply final response guardrails before returning data to the API client."""

    response_data = response.model_dump()

    citations = _derive_citations(response_data)
    if citations:
        response_data["citations"] = citations
    elif _is_document_backed(response_data):
        logger.warning("output guardrail OG-1 blocked response without citations request_id=%s", request_id)
        return _block_response(
            response_data,
            "output_guardrail_og_1_source_attribution_failed",
            (
                "The automated response was suppressed because it did not include required source "
                "attribution. This request requires human review before a final answer is provided."
            ),
        )

    unsafe_policy_match = _first_unsafe_policy_match(str(response_data.get("answer") or ""))
    if unsafe_policy_match:
        logger.warning(
            "output guardrail OG-3 blocked unsafe policy action request_id=%s pattern=%s",
            request_id,
            unsafe_policy_match,
        )
        return _block_response(
            response_data,
            "output_guardrail_og_3_policy_action_failed",
            (
                "The automated response was blocked because it proposed an unsafe operational action. "
                "This request requires mandatory escalation."
            ),
        )

    redactions: dict[str, int] = {}
    response_data = _sanitize_value(response_data, redactions)
    if redactions:
        logger.warning(
            "output guardrail OG-2 redacted sensitive data request_id=%s redaction_types=%s",
            request_id,
            list(redactions),
        )

    return RetrievalResponse.model_validate(response_data)
