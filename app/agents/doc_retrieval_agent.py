import logging
import time
from datetime import datetime, timezone
from typing import Any

from app.core.langfuse import observe, trace_agent_state
from app.orchestration.state import (
    AgentTraceEvent,
    ExecutionResult,
    ProgressUpdate,
    RetrievedChunk,
    SupportOrchestrationState,
    VerificationOutcome,
)
from app.services.tools.vector_tool import execute_vector_evidence

logger = logging.getLogger(__name__)

AGENT_NAME = "document_retrieval_agent"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_list(state: SupportOrchestrationState, key: str, item: Any) -> None:
    values = list(state.get(key, []))
    values.append(item)
    state[key] = values  # type: ignore[literal-required]


def _clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


def _metadata_value(metadata: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = metadata.get(key)
        if value is not None:
            return value
    return None


def _to_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _source_node_to_chunk(source_node: dict[str, Any]) -> RetrievedChunk:
    metadata = dict(source_node.get("metadata") or {})
    page_number = source_node.get("page_number")
    if page_number is None:
        page_number = _metadata_value(metadata, "page_number", "page", "page_label")

    score = source_node.get("similarity_score")
    if score is None:
        score = _metadata_value(metadata, "hybrid_score", "vector_score", "keyword_score")

    return {
        "chunk_text": str(source_node.get("chunk_text") or source_node.get("text") or ""),
        "source_file": source_node.get("source_file") or _metadata_value(metadata, "source_file", "file_name", "source"),
        "page_number": _to_int_or_none(page_number),
        "content_type": source_node.get("content_type") or metadata.get("content_type"),
        "category": source_node.get("category") or metadata.get("category"),
        "score": _clamp_score(score),
        "metadata": metadata,
    }


def _progress(step_id: str, status: str, message: str) -> ProgressUpdate:
    return {
        "step_id": step_id,
        "agent_name": AGENT_NAME,
        "status": status,
        "message": message,
        "timestamp": _utc_now(),
    }


def _trace(action: str, status: str, input_summary: str | None, output_summary: str | None, latency_ms: int | None) -> AgentTraceEvent:
    return {
        "agent_name": AGENT_NAME,
        "action": action,
        "input_summary": input_summary,
        "output_summary": output_summary,
        "status": status,
        "timestamp": _utc_now(),
        "latency_ms": latency_ms,
    }


def _execution_result(
    chunks: list[RetrievedChunk],
    citations: list[str],
    top_score: float,
    error: str | None = None,
) -> ExecutionResult:
    return {
        "step_id": "document-retrieval",
        "agent_name": AGENT_NAME,
        "result_type": "document_retrieval",
        "summary": f"retrieved_chunks={len(chunks)}; top_score={top_score:.2f}",
        "data": {
            "chunk_count": len(chunks),
            "citations": citations,
            "top_similarity_score": top_score,
        },
        "error": error,
        "timestamp": _utc_now(),
    }


def _verification(chunks: list[RetrievedChunk], top_score: float, error: str | None = None) -> VerificationOutcome:
    passed = error is None and bool(chunks)
    return {
        "check_name": "document_retrieval_complete",
        "passed": passed,
        "score": top_score if chunks else 0.0,
        "reason": "Document retrieval returned source chunks and confidence was updated."
        if passed
        else error or "Document retrieval returned no source chunks.",
        "corrective_action": None if passed else "Try a different route, ask for clarification, or escalate if business impact is high.",
        "metadata": {"chunk_count": len(chunks)},
    }


def _is_unsupported_lookup(state: SupportOrchestrationState) -> bool:
    metadata = dict(state.get("metadata") or {})
    debug = dict(metadata.get("intent_classifier_debug") or {})
    matched = dict(debug.get("matched_hits") or {})
    local_matched = dict(debug.get("local_matched_hits") or {})
    return bool(matched.get("unsupported_lookup") or local_matched.get("unsupported_lookup"))


def _update_confidence(state: SupportOrchestrationState, chunks: list[RetrievedChunk]) -> float:
    if not chunks:
        existing = state.get("confidence_score")
        return _clamp_score(existing) if existing is not None else 0.0

    top_score = max(_clamp_score(chunk.get("score")) for chunk in chunks)
    existing = _clamp_score(state.get("confidence_score")) if state.get("confidence_score") is not None else 0.0
    if _is_unsupported_lookup(state):
        state["confidence_score"] = existing
        return top_score
    state["confidence_score"] = max(existing, top_score)
    return top_score


@observe(name=AGENT_NAME, as_type="agent", capture_input=False, capture_output=False)
async def retrieve_documents(state: SupportOrchestrationState) -> SupportOrchestrationState:
    """Retrieve documentation evidence and store it in orchestration state."""

    started = time.perf_counter()
    next_state: SupportOrchestrationState = dict(state)
    query = str(next_state.get("query") or "")

    _append_list(next_state, "progress_updates", _progress("document-retrieval", "started", "Document retrieval started."))

    chunks: list[RetrievedChunk] = []
    citations: list[str] = []
    error: str | None = None
    status = "completed"

    try:
        result = await execute_vector_evidence(query)
        error = result.get("error")
        source_nodes = result.get("source_nodes") or []
        chunks = [_source_node_to_chunk(dict(source_node)) for source_node in source_nodes]
        citations = [str(citation) for citation in result.get("citations", []) if citation]

        next_state["retrieved_chunks"] = chunks
        next_state["citations"] = citations

        if error:
            status = "failed"
            next_state["retrieved_chunks"] = []
            _append_list(next_state, "errors", f"Document retrieval failed: {error}")
        elif not chunks:
            _append_list(next_state, "errors", "Document retrieval returned no source chunks.")
    except Exception as exc:
        logger.exception("Document retrieval agent failed")
        error = str(exc)
        status = "failed"
        next_state["retrieved_chunks"] = []
        _append_list(next_state, "errors", f"Document retrieval failed: {error}")

    top_score = _update_confidence(next_state, chunks)
    latency_ms = int((time.perf_counter() - started) * 1000)
    output_summary = f"retrieved_chunks={len(chunks)}; top_score={top_score:.2f}; status={status}"

    _append_list(next_state, "execution_results", _execution_result(chunks, citations, top_score, error))
    _append_list(next_state, "verification_outcomes", _verification(chunks, top_score, error))
    _append_list(
        next_state,
        "agent_trace",
        _trace(
            action="retrieve_documents",
            status=status,
            input_summary=f"query_length={len(query)}",
            output_summary=output_summary,
            latency_ms=latency_ms,
        ),
    )
    _append_list(next_state, "progress_updates", _progress("document-retrieval", "completed", output_summary))

    trace_agent_state(
        agent_name=AGENT_NAME,
        input_state=state,
        output_state=next_state,
        started_at=started,
        tool_used="vector_evidence_retrieval",
    )

    return next_state
