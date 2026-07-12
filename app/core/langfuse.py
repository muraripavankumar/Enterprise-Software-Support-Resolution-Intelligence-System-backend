import os
import time
from collections.abc import Callable, Mapping
from typing import Any, Literal, TypeVar, overload

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])
ObservationType = Literal[
    "generation",
    "embedding",
    "span",
    "agent",
    "tool",
    "chain",
    "retriever",
    "evaluator",
    "guardrail",
]


def _configure_langfuse_env() -> bool:
    if not settings.langfuse_configured:
        return False

    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key or "")
    os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key or "")
    os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_host or "")
    return True


_LANGFUSE_ENABLED = _configure_langfuse_env()

try:
    if _LANGFUSE_ENABLED:
        from langfuse import get_client as _get_langfuse_client
        from langfuse import observe as _langfuse_observe
    else:
        _get_langfuse_client = None
        _langfuse_observe = None
except Exception:
    logger.exception("Langfuse SDK could not be initialized; tracing is disabled.")
    _LANGFUSE_ENABLED = False
    _get_langfuse_client = None
    _langfuse_observe = None


@overload
def observe(func: F) -> F:
    ...


@overload
def observe(
    func: None = None,
    *,
    name: str | None = None,
    as_type: ObservationType | None = None,
    capture_input: bool | None = None,
    capture_output: bool | None = None,
) -> Callable[[F], F]:
    ...


def observe(
    func: F | None = None,
    *,
    name: str | None = None,
    as_type: ObservationType | None = None,
    capture_input: bool | None = None,
    capture_output: bool | None = None,
) -> F | Callable[[F], F]:
    """Langfuse observe decorator with a no-op fallback for local/dev runs."""

    if _LANGFUSE_ENABLED and _langfuse_observe is not None:
        return _langfuse_observe(
            func,
            name=name,
            as_type=as_type,
            capture_input=capture_input,
            capture_output=capture_output,
        )

    def decorator(inner: F) -> F:
        return inner

    if func is not None:
        return decorator(func)
    return decorator


def _client() -> Any | None:
    if not _LANGFUSE_ENABLED or _get_langfuse_client is None:
        return None
    try:
        return _get_langfuse_client()
    except Exception:
        logger.exception("Langfuse client is unavailable.")
        return None


def get_langfuse_client() -> Any | None:
    """Return the configured Langfuse client, or None when tracing is disabled."""

    return _client()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "value"):
        return getattr(value, "value")
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _state_value(state: Mapping[str, Any] | None, key: str, default: Any = None) -> Any:
    if not state:
        return default
    return _json_safe(state.get(key, default))


def _failed_guardrails(state: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    flags = state.get("guardrail_flags", []) if state else []
    failed: list[dict[str, Any]] = []
    for flag in flags:
        if isinstance(flag, Mapping) and flag.get("passed") is False:
            failed.append(
                {
                    "name": _json_safe(flag.get("name")),
                    "severity": _json_safe(flag.get("severity")),
                    "reason": _json_safe(flag.get("reason")),
                }
            )
    return failed


def _state_output_summary(state: Mapping[str, Any] | None) -> dict[str, Any]:
    chunks = state.get("retrieved_chunks", []) if state else []
    sql_results = state.get("sql_results", []) if state else []
    return {
        "intent": _state_value(state, "intent"),
        "route_decision": _state_value(state, "route_decision"),
        "severity": _state_value(state, "severity"),
        "severity_priority": _state_value(state, "severity_priority"),
        "confidence_score": _state_value(state, "confidence_score"),
        "escalation_flag": bool(state.get("escalation_flag", False)) if state else False,
        "escalation_target": _state_value(state, "escalation_target"),
        "retrieved_chunk_count": len(chunks) if isinstance(chunks, list) else 0,
        "sql_result_count": len(sql_results) if isinstance(sql_results, list) else 0,
        "citation_count": len(state.get("citations", [])) if state else 0,
        "error_count": len(state.get("errors", [])) if state else 0,
    }


def trace_agent_state(
    *,
    agent_name: str,
    input_state: Mapping[str, Any] | None,
    output_state: Mapping[str, Any] | None,
    started_at: float,
    tool_used: str | None = None,
    tokens_consumed: int | None = None,
) -> None:
    client = _client()
    if client is None:
        return

    latency_ms = int((time.perf_counter() - started_at) * 1000)
    metadata = {
        "agent_name": agent_name,
        "latency_ms": latency_ms,
        "tool_used": tool_used,
        "tokens_consumed": tokens_consumed,
        "guardrail_triggers": _failed_guardrails(output_state),
        **_state_output_summary(output_state),
    }
    try:
        client.update_current_span(
            input={
                "query": _state_value(input_state, "query"),
                "conversation_id": _state_value(input_state, "conversation_id"),
                "user_id": _state_value(input_state, "user_id"),
                "route_decision": _state_value(input_state, "route_decision"),
                "metadata": _state_value(input_state, "metadata", {}),
            },
            output=_state_output_summary(output_state),
            metadata=_json_safe(metadata),
            level="ERROR" if metadata["error_count"] else "DEFAULT",
        )
    except Exception:
        logger.exception("Failed to update Langfuse agent span for %s.", agent_name)


def trace_tool_result(
    *,
    tool_name: str,
    question: str,
    result: Mapping[str, Any],
    started_at: float,
    tokens_consumed: int | None = None,
) -> None:
    client = _client()
    if client is None:
        return

    source_nodes = result.get("source_nodes") or []
    similarity_scores = []
    if isinstance(source_nodes, list):
        for node in source_nodes:
            if isinstance(node, Mapping):
                score = node.get("similarity_score")
                if score is not None:
                    similarity_scores.append(score)

    metadata = {
        "tool_name": tool_name,
        "latency_ms": int((time.perf_counter() - started_at) * 1000),
        "tokens_consumed": tokens_consumed,
        "chunk_count": result.get("chunk_count") or len(source_nodes) if isinstance(source_nodes, list) else 0,
        "similarity_scores": similarity_scores,
        "sql_query": result.get("sql_query"),
        "tables_used": result.get("table_used") or result.get("tables_used"),
        "row_count": result.get("row_count"),
        "execution_path": result.get("execution_path"),
        "phase_timings_ms": result.get("phase_timings_ms"),
        "error": result.get("error"),
    }
    try:
        client.update_current_span(
            input={"query": question},
            output={
                "answer": result.get("answer"),
                "chunk_count": metadata["chunk_count"],
                "citations": result.get("citations"),
                "row_count": result.get("row_count"),
                "error": result.get("error"),
            },
            metadata=_json_safe(metadata),
            level="ERROR" if result.get("error") else "DEFAULT",
        )
    except Exception:
        logger.exception("Failed to update Langfuse tool span for %s.", tool_name)


def trace_guardrail_event(
    *,
    name: str,
    passed: bool,
    reason: str,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    client = _client()
    if client is None:
        return
    try:
        client.update_current_span(
            metadata=_json_safe(
                {
                    "guardrail_event": {
                        "name": name,
                        "passed": passed,
                        "reason": reason,
                        "metadata": dict(metadata or {}),
                    }
                }
            ),
            level="DEFAULT" if passed else "WARNING",
            status_message=None if passed else reason,
        )
    except Exception:
        logger.exception("Failed to update Langfuse guardrail event for %s.", name)


def trace_final_response(
    *,
    trace_id: str,
    question: str,
    response_data: Mapping[str, Any],
) -> None:
    """Attach the final user-visible response payload to a Langfuse trace."""

    client = _client()
    if client is None or not trace_id:
        return

    output = {
        "answer": _json_safe(response_data.get("answer")),
        "success": _json_safe(response_data.get("success")),
        "mode": _json_safe(response_data.get("mode")),
        "route_decision": _json_safe(response_data.get("route_decision")),
        "intent": _json_safe(response_data.get("intent")),
        "severity": _json_safe(response_data.get("severity")),
        "confidence_score": _json_safe(response_data.get("confidence_score")),
        "escalation_flag": _json_safe(response_data.get("escalation_flag")),
        "escalation_target": _json_safe(response_data.get("escalation_target")),
        "citations": _json_safe(response_data.get("citations")),
        "citation_references": _json_safe(response_data.get("citation_references")),
        "structured_result": _json_safe(response_data.get("structured_result")),
        "answer_quality": _json_safe(response_data.get("answer_quality")),
        "quality_warnings": _json_safe(response_data.get("quality_warnings")),
        "cache_status": _json_safe(response_data.get("cache_status")),
        "error": _json_safe(response_data.get("error")),
    }
    metadata = {
        "payload": "final_user_visible_response",
        "latency_ms": _json_safe(response_data.get("latency_ms")),
        "agent_trace_latency_ms": _json_safe(response_data.get("agent_trace_latency_ms")),
        "runtime_overhead_ms": _json_safe(response_data.get("runtime_overhead_ms")),
        "tools_used": _json_safe(response_data.get("tools_used")),
        "source_node_count": len(response_data.get("source_nodes") or []),
        "agent_trace_count": len(response_data.get("agent_trace") or []),
    }

    try:
        observation = client.start_observation(
            name="final_user_response",
            as_type="span",
            trace_context={"trace_id": trace_id},
            input={"query": question},
            output=output,
            metadata=_json_safe(metadata),
            level="ERROR" if response_data.get("error") else "DEFAULT",
        )
        observation.set_trace_io(input={"query": question}, output=output)
        observation.end()
    except Exception:
        logger.exception("Failed to attach final response to Langfuse trace %s.", trace_id)


def flush_langfuse() -> None:
    client = _client()
    if client is None:
        return
    try:
        client.flush()
    except Exception:
        logger.exception("Failed to flush Langfuse events.")
