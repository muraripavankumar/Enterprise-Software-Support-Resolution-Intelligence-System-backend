import asyncio
import json
import re
import time
from uuid import uuid4
from pathlib import Path, PureWindowsPath
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.core.auth import AuthenticatedUser, auth_metadata, require_permissions, validate_token
from app.core.logging import get_logger
from app.core.semantic_cache import cache_route_for_query, get_semantic_cache
from app.evaluation.langfuse_scores import attach_runtime_slo_scores
from app.evaluation.slo_config import SLO_CONFIG_BY_NAME
from app.middleware.input_guardrails import GuardrailViolation, apply_input_guardrails
from app.middleware.output_guardrails import apply_output_guardrails
from app.schemas.retrieval import (
    CitationReference,
    IncidentEvidenceRecord,
    IncidentInvestigationEvidence,
    JiraTrackingEvidence,
    QualitySLOWarning,
    RetrievalErrorResponse,
    RetrievalMode,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalSourceNode,
    StructuredQueryResult,
)
from app.services.retrieval_service import RetrievalService

router = APIRouter(prefix="/retrieval", tags=["retrieval"])
logger = get_logger(__name__)

SLO_THRESHOLDS_MS = {
    "intent_classifier": 1000,
    "rag_retrieval": 8000,
    "direct_sql": 2000,
    "nl2sql_fallback": 10000,
    "total_response": 15000,
    "jira_tracking": 45000,
}

SAFE_SOURCE_METADATA_KEYS = {
    "source",
    "source_file",
    "original_filename",
    "page_number",
    "content_type",
    "category",
    "chunk_index",
    "table_index",
    "image_index",
}

MOJIBAKE_REPLACEMENTS = {
    "â€™": "'",
    "â€˜": "'",
    "â€œ": '"',
    "â€�": '"',
    "â€“": "-",
    "â€”": "-",
    "â€¦": "...",
    "â€¢": "-",
    "Â ": " ",
    "Â": "",
}

TICKET_EVIDENCE_TABLES = {"support_tickets"}
INCIDENT_EVIDENCE_TABLES = {"incident_logs"}
USER_VISIBLE_QUALITY_SLOS = {
    "faithfulness",
    "answer_relevance",
    "llm_judge_quality",
    "context_precision",
    "retrieval_recall_at_5",
}


def _latency_ms(started_at: float) -> int:
    return int((time.perf_counter() - started_at) * 1000)


def _agent_trace_latency_ms(response: RetrievalResponse) -> int:
    total = 0
    for event in response.agent_trace or []:
        if event.latency_ms is None:
            continue
        try:
            latency = int(event.latency_ms)
        except (TypeError, ValueError):
            continue
        if latency > 0:
            total += latency
    return total


def _with_runtime_timing(response: RetrievalResponse, started_at: float) -> RetrievalResponse:
    total_latency = _latency_ms(started_at)
    trace_latency = _agent_trace_latency_ms(response)
    return response.model_copy(
        update={
            "latency_ms": total_latency,
            "agent_trace_latency_ms": trace_latency,
            "runtime_overhead_ms": max(0, total_latency - trace_latency),
        }
    )


def _quality_value(response: RetrievalResponse, metric: str) -> float | None:
    quality = response.answer_quality
    if metric == "faithfulness":
        return quality.faithfulness_score if quality else None
    if metric == "answer_relevance":
        return quality.answer_relevance_score if quality else None
    if metric == "llm_judge_quality":
        return quality.overall_quality_score if quality else None
    if metric in {"context_precision", "retrieval_recall_at_5"}:
        route = str(response.route_decision or response.mode.value or "").lower()
        document_backed = route in {"rag", "hybrid", "vector"} or bool(response.source_nodes or response.citations)
        if not document_backed:
            return None
        return 1.0 if response.source_nodes else 0.0
    return None


def _quality_warning_message(display_name: str, value: float, target: float) -> str:
    return (
        f"{display_name} is {value:.0%}, below the configured target of {target:.0%}. "
        "Treat this answer as provisional and verify the cited evidence or rerun the query with more specific details."
    )


def _quality_slo_warnings(response: RetrievalResponse) -> list[QualitySLOWarning]:
    warnings: list[QualitySLOWarning] = []
    for metric in sorted(USER_VISIBLE_QUALITY_SLOS):
        config = SLO_CONFIG_BY_NAME.get(metric)
        if config is None:
            continue
        value = _quality_value(response, metric)
        if value is None:
            continue
        passed = value >= config.target if config.higher_is_better else value <= config.target
        if passed:
            continue
        warnings.append(
            QualitySLOWarning(
                metric=metric,
                display_name=config.display_name,
                value=value,
                target=config.target,
                message=_quality_warning_message(config.display_name, value, config.target),
            )
        )
    return warnings


def _with_quality_slo_warnings(response: RetrievalResponse) -> RetrievalResponse:
    warnings = _quality_slo_warnings(response)
    if not warnings:
        return response

    warning_lines = [f"- {warning.message}" for warning in warnings]
    answer = str(response.answer or "").rstrip()
    if "Quality notice:" not in answer:
        answer = f"{answer}\n\nQuality notice:\n" + "\n".join(warning_lines)

    final_answer_model = dict(response.final_answer_model or {})
    final_answer_model["quality_warnings"] = [warning.model_dump() for warning in warnings]

    return response.model_copy(
        update={
            "answer": answer,
            "quality_warnings": warnings,
            "final_answer_model": final_answer_model or response.final_answer_model,
        }
    )


def _with_cache_metadata(
    response: RetrievalResponse,
    *,
    status: str,
    route: str,
    strategy: str | None = None,
) -> RetrievalResponse:
    return response.model_copy(
        update={
            "cache_status": status,
            "cache_route": route,
            "cache_strategy": strategy,
        }
    )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    for bad, good in MOJIBAKE_REPLACEMENTS.items():
        text = text.replace(bad, good)
    return text


def _display_source(value: Any) -> str | None:
    if value is None:
        return None
    text = _clean_text(value).strip()
    if not text:
        return None
    if "\\" in text or ":" in text:
        return PureWindowsPath(text).name
    return Path(text).name


def _safe_source_metadata(raw_metadata: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in SAFE_SOURCE_METADATA_KEYS:
        if key not in raw_metadata:
            continue
        value = raw_metadata.get(key)
        if key in {"source", "source_file", "original_filename"}:
            metadata[key] = _display_source(value)
        elif isinstance(value, str):
            metadata[key] = _clean_text(value)
        else:
            metadata[key] = _json_safe(value)
    return {key: value for key, value in metadata.items() if value is not None}


def _clean_citations(citations: Any) -> list[str]:
    return sorted({source for citation in citations or [] if (source := _display_source(citation))})


def _citation_parts(value: Any) -> tuple[str | None, int | None, str | None]:
    raw = _clean_text(value).strip()
    if not raw:
        return None, None, None

    page_number = None
    page_match = re.search(r"\bpages?\s+(\d+)\b", raw, flags=re.IGNORECASE)
    if page_match:
        page_number = int(page_match.group(1))
        raw_source = raw[: page_match.start()].rstrip(" ,;")
    else:
        raw_source = raw

    document_name = _display_source(raw_source)
    if not document_name:
        return None, page_number, None
    return document_name, page_number, raw_source


def _citation_references(
    *,
    citations: list[str],
    nodes: list[RetrievalSourceNode],
) -> list[CitationReference]:
    grouped: dict[str, dict[str, Any]] = {}

    def add(document_name: str | None, page_number: Any = None, source_file: Any = None) -> None:
        if not document_name:
            return
        key = document_name.strip()
        if not key:
            return
        entry = grouped.setdefault(key, {"pages": set(), "source_file": source_file or document_name})
        try:
            if page_number is not None:
                entry["pages"].add(int(page_number))
        except (TypeError, ValueError):
            pass

    for node in nodes:
        add(node.source_file, node.page_number, node.source_file)

    for citation in citations:
        document_name, page_number, source_file = _citation_parts(citation)
        add(document_name, page_number, source_file)

    references = []
    for index, document_name in enumerate(sorted(grouped), start=1):
        entry = grouped[document_name]
        references.append(
            CitationReference(
                citation_id=index,
                document_name=document_name,
                pages=sorted(entry["pages"]),
                source_file=_display_source(entry.get("source_file")) or document_name,
            )
        )
    return references


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _source_nodes(result: dict[str, Any], include_sources: bool) -> list[RetrievalSourceNode]:
    if not include_sources:
        return []

    nodes: list[RetrievalSourceNode] = []
    for raw_node in result.get("source_nodes", []) or []:
        metadata = _safe_source_metadata(dict(raw_node.get("metadata", {}) or {}))
        source_file = _display_source(raw_node.get("source_file") or metadata.get("source_file") or metadata.get("source"))
        nodes.append(
            RetrievalSourceNode(
                text_preview=_clean_text(raw_node.get("chunk_text") or raw_node.get("text_preview") or "")[:500],
                source_file=source_file,
                page_number=metadata.get("page_number"),
                content_type=raw_node.get("content_type") or metadata.get("content_type"),
                category=_clean_text(raw_node.get("category") or metadata.get("category") or "") or None,
                similarity_score=_to_float(raw_node.get("similarity_score") or raw_node.get("score")),
                metadata=metadata,
            )
        )
    return nodes


def _tools_used(result: dict[str, Any], nodes: list[RetrievalSourceNode]) -> list[str]:
    tools = [str(tool) for tool in result.get("tools_used", []) or [] if str(tool).strip()]
    if nodes and "document_retrieval" not in tools:
        tools.append("document_retrieval")
    if result.get("structured_result") and "sql_agent" not in tools:
        tools.append("sql_agent")
    if result.get("incident_investigation") and "incident_investigator" not in tools:
        tools.append("incident_investigator")
    return tools


def _has_permission(current_user: AuthenticatedUser, permission: str) -> bool:
    return permission in set(current_user.permissions)


def _table_names_from_value(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, list):
        return {str(item).strip() for item in value if str(item).strip()}
    text = str(value)
    return {
        item.strip()
        for item in re.split(r"[,;\s]+", text)
        if item.strip()
    }


def _structured_result_tables(result: dict[str, Any]) -> set[str]:
    structured = dict(result.get("structured_result") or {})
    return _table_names_from_value(structured.get("table_used") or structured.get("tables_used"))


def _sql_result_tables(result: dict[str, Any]) -> set[str]:
    return _table_names_from_value(result.get("table_used") or result.get("tables_used"))


def _strip_answer_lines(answer: str, blocked_terms: set[str]) -> str:
    lines = []
    for line in str(answer or "").splitlines():
        lowered = line.lower()
        if any(term in lowered for term in blocked_terms):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _restricted_answer(missing_permissions: list[str]) -> str:
    readable = ", ".join(missing_permissions)
    return (
        "I can process support questions for your account, but this result contains operational evidence "
        f"that requires additional permission: {readable}. Please use a support_agent, support_manager, "
        "or admin account to view ticket or incident details."
    )


def _scope_result_to_permissions(result: dict[str, Any], current_user: AuthenticatedUser) -> dict[str, Any]:
    """Remove evidence fields the current principal is not allowed to read."""

    scoped = dict(result)
    tools = [str(tool) for tool in list(scoped.get("tools_used") or [])]
    missing: list[str] = []

    tables = _structured_result_tables(scoped)
    has_ticket_data = bool(tables & TICKET_EVIDENCE_TABLES)
    has_incident_data = bool(tables & INCIDENT_EVIDENCE_TABLES) or bool(scoped.get("incident_investigation"))

    if has_ticket_data and not _has_permission(current_user, "read:tickets"):
        scoped["structured_result"] = None
        tools = [tool for tool in tools if tool != "sql_agent"]
        missing.append("read:tickets")

    if has_incident_data and not _has_permission(current_user, "read:incidents"):
        if tables & INCIDENT_EVIDENCE_TABLES:
            scoped["structured_result"] = None
            tools = [tool for tool in tools if tool != "sql_agent"]
        scoped["incident_investigation"] = None
        tools = [tool for tool in tools if tool != "incident_investigator"]
        missing.append("read:incidents")

    if missing:
        scoped["answer"] = _restricted_answer(sorted(set(missing)))
        scoped["citations"] = []
        scoped["source_nodes"] = []
        scoped["chunk_count"] = 0
        scoped["confidence_score"] = None

    if scoped.get("jira_tracking_result") and not _has_permission(current_user, "trigger:escalation"):
        scoped["jira_tracking_result"] = {}
        tools = [tool for tool in tools if tool != "jira_mcp_tool"]
        scoped["answer"] = _strip_answer_lines(
            str(scoped.get("answer") or ""),
            {"jira", "atlassian", "browse/"},
        )

    scoped["tools_used"] = tools
    return scoped


def _scope_sql_result_to_permissions(result: dict[str, Any], current_user: AuthenticatedUser) -> dict[str, Any]:
    scoped = dict(result)
    tables = _sql_result_tables(scoped)
    missing: list[str] = []
    if tables & TICKET_EVIDENCE_TABLES and not _has_permission(current_user, "read:tickets"):
        missing.append("read:tickets")
    if tables & INCIDENT_EVIDENCE_TABLES and not _has_permission(current_user, "read:incidents"):
        missing.append("read:incidents")
    if missing:
        scoped.update(
            {
                "answer": _restricted_answer(sorted(set(missing))),
                "sql_query": None,
                "table_used": None,
                "row_count": 0,
                "raw_results": [],
                "tools_used": [],
            }
        )
    return scoped


def _top_confidence(result: dict[str, Any], nodes: list[RetrievalSourceNode]) -> float | None:
    explicit = _to_float(result.get("confidence_score"))
    if explicit is not None:
        return explicit
    scores = [node.similarity_score for node in nodes if node.similarity_score is not None]
    return max(scores) if scores else None


def _vector_response(
    request: RetrievalRequest,
    result: dict[str, Any],
    started_at: float,
) -> RetrievalResponse:
    error = result.get("error")
    nodes = _source_nodes(result, request.include_sources)
    citations = _clean_citations(result.get("citations", [])) if request.include_sources else []
    citation_references = _citation_references(citations=citations, nodes=nodes) if request.include_sources else []
    return RetrievalResponse(
        success=error is None,
        mode=request.mode,
        question=request.question,
        answer=_clean_text(result.get("answer") or ""),
        citations=citations,
        citation_references=citation_references,
        source_nodes=nodes,
        chunk_count=int(result.get("chunk_count") or len(nodes)),
        route_decision=str(result.get("route_decision") or "rag"),
        confidence_score=_top_confidence(result, nodes),
        tools_used=_tools_used(result, nodes),
        error=error,
        latency_ms=_latency_ms(started_at),
    )


def _sql_response(
    request: RetrievalRequest,
    result: dict[str, Any],
    started_at: float,
    current_user: AuthenticatedUser,
) -> RetrievalResponse:
    result = _scope_sql_result_to_permissions(result, current_user)
    error = result.get("error")
    raw_results = _json_safe(result.get("raw_results", [])) if request.include_raw_results else []
    structured_result = None
    if result.get("table_used") or result.get("sql_query") or raw_results:
        structured_result = StructuredQueryResult(
            answer=_clean_text(result.get("answer") or ""),
            sql_query=result.get("sql_query"),
            table_used=result.get("table_used"),
            row_count=int(result.get("row_count") or 0),
            raw_results=raw_results,
        )
    return RetrievalResponse(
        success=error is None,
        mode=request.mode,
        question=request.question,
        answer=_clean_text(result.get("answer") or ""),
        structured_result=structured_result,
        route_decision=str(result.get("route_decision") or "sql"),
        tools_used=_tools_used(result, []),
        error=error,
        latency_ms=_latency_ms(started_at),
    )


def _incident_investigation_response(result: dict[str, Any]) -> IncidentInvestigationEvidence | None:
    raw_investigation = result.get("incident_investigation")
    if not raw_investigation:
        return None

    investigation = dict(raw_investigation)
    matched_incidents = []
    for raw_incident in list(investigation.get("matched_incidents") or []):
        incident = dict(raw_incident)
        matched_incidents.append(
            IncidentEvidenceRecord(
                incident_id=incident.get("incident_id"),
                incident_type=_clean_text(incident.get("incident_type") or "") or None,
                severity=_clean_text(incident.get("severity") or "") or None,
                affected_region=_clean_text(incident.get("affected_region") or "") or None,
                start_time=_clean_text(incident.get("start_time") or "") or None,
                end_time=_clean_text(incident.get("end_time") or "") or None,
                resolution_status=_clean_text(incident.get("resolution_status") or "") or None,
                root_cause=_clean_text(incident.get("root_cause") or "") or None,
                escalation_flag=bool(incident.get("escalation_flag")),
                correlation_score=_to_float(incident.get("correlation_score")),
                correlation_reasons=[str(reason) for reason in list(incident.get("correlation_reasons") or [])],
            )
        )

    return IncidentInvestigationEvidence(
        filters_used=_json_safe(dict(investigation.get("filters_used") or {})),
        matched_incidents=matched_incidents,
        active_critical_incident=bool(investigation.get("active_critical_incident")),
        active_critical_incident_correlated=bool(investigation.get("active_critical_incident_correlated")),
        max_correlation_score=_to_float(investigation.get("max_correlation_score")) or 0.0,
        correlation_threshold=_to_float(investigation.get("correlation_threshold")) or 0.65,
        investigation_summary=_clean_text(investigation.get("investigation_summary") or ""),
    )


def _jira_tracking_response(result: dict[str, Any]) -> JiraTrackingEvidence | None:
    raw_jira = result.get("jira_tracking_result")
    if not raw_jira:
        return None
    jira = dict(raw_jira)
    return JiraTrackingEvidence(
        enabled=bool(jira.get("enabled")),
        attempted=bool(jira.get("attempted")),
        should_create=bool(jira.get("should_create")),
        action=_clean_text(jira.get("action") or "") or None,
        reason_code=_clean_text(jira.get("reason_code") or "") or None,
        project_key=_clean_text(jira.get("project_key") or "") or None,
        issue_type=_clean_text(jira.get("issue_type") or "") or None,
        priority=_clean_text(jira.get("priority") or "") or None,
        issue_key=_clean_text(jira.get("issue_key") or "") or None,
        issue_url=_clean_text(jira.get("issue_url") or "") or None,
        duplicate_found=bool(jira.get("duplicate_found")),
        dedupe_jql=_clean_text(jira.get("dedupe_jql") or "") or None,
        comment_added=bool(jira.get("comment_added")),
        triage_transition_attempted=bool(jira.get("triage_transition_attempted")),
        triage_transition_status=_clean_text(jira.get("triage_transition_status") or "") or None,
        status=_clean_text(jira.get("status") or "") or None,
        error=_clean_text(jira.get("error") or "") or None,
    )


def _agent_response(
    request: RetrievalRequest,
    result: dict[str, Any],
    started_at: float,
    current_user: AuthenticatedUser,
) -> RetrievalResponse:
    result = _scope_result_to_permissions(result, current_user)
    error = result.get("error")
    nodes = _source_nodes(result, request.include_sources)
    citations = _clean_citations(result.get("citations", [])) if request.include_sources else []
    citation_references = _citation_references(citations=citations, nodes=nodes) if request.include_sources else []
    structured_result = None
    if result.get("structured_result"):
        raw_structured = dict(result["structured_result"])
        raw_results = _json_safe(raw_structured.get("raw_results", [])) if request.include_raw_results else []
        structured_result = StructuredQueryResult(
            answer=_clean_text(raw_structured.get("answer") or ""),
            sql_query=raw_structured.get("sql_query"),
            table_used=raw_structured.get("table_used"),
            row_count=int(raw_structured.get("row_count") or 0),
            raw_results=raw_results,
        )
    incident_investigation = _incident_investigation_response(result)
    jira_tracking = _jira_tracking_response(result)
    return RetrievalResponse(
        success=error is None,
        mode=request.mode,
        question=request.question,
        answer=_clean_text(result.get("answer") or ""),
        citations=citations,
        citation_references=citation_references,
        source_nodes=nodes,
        chunk_count=int(result.get("chunk_count") or len(nodes)),
        structured_result=structured_result,
        incident_investigation=incident_investigation,
        jira_tracking=jira_tracking,
        intent=result.get("intent"),
        route_decision=result.get("route_decision"),
        severity=result.get("severity"),
        confidence_score=_top_confidence(result, nodes),
        escalation_flag=bool(result.get("escalation_flag")),
        escalation_target=result.get("escalation_target"),
        tools_used=_tools_used(result, nodes),
        agent_trace=list(result.get("agent_trace") or []),
        suggested_questions=[str(item) for item in result.get("suggested_questions", []) or [] if str(item).strip()],
        answer_quality=_json_safe(result.get("answer_quality")),
        final_answer_model=_json_safe(result.get("final_answer_model")),
        error=error,
        latency_ms=_latency_ms(started_at),
    )


def _request_metadata(request: RetrievalRequest) -> dict[str, Any]:
    metadata = dict(request.metadata or {})
    if request.conversation_id:
        metadata["conversation_id"] = request.conversation_id
        metadata.setdefault("thread_id", request.conversation_id)
    if request.user_id:
        metadata["user_id"] = request.user_id
    return metadata


def _cache_attributes(request: RetrievalRequest, route: str) -> dict[str, Any]:
    metadata = dict(request.metadata or {})
    role = str(metadata.get("user_role") or metadata.get("role") or "anonymous").lower()
    attributes: dict[str, Any] = {
        "mode": request.mode.value,
        "cache_route": route,
        "include_sources": request.include_sources,
        "include_raw_results": request.include_raw_results,
        "user_role": role,
    }
    if route == "sql" or request.mode == RetrievalMode.SQL:
        attributes["user_scope"] = request.user_id or metadata.get("user_id") or metadata.get("sub") or "anonymous"
    return attributes


def _cacheable_response_payload(response: RetrievalResponse) -> dict[str, Any]:
    payload = response.model_dump()
    structured = payload.get("structured_result")
    if isinstance(structured, dict):
        structured["raw_results"] = []
    return payload


def _sse_message(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(_json_safe(data), ensure_ascii=False)}\n\n"


def _answer_chunks(answer: str, words_per_chunk: int = 8) -> list[str]:
    words = str(answer or "").split(" ")
    chunks: list[str] = []
    for index in range(0, len(words), words_per_chunk):
        chunk = " ".join(words[index : index + words_per_chunk])
        if chunk:
            chunks.append(chunk + (" " if index + words_per_chunk < len(words) else ""))
    return chunks or [""]


def _stream_progress_message(elapsed_seconds: int) -> str:
    steps = [
        "Validating request and permissions",
        "Classifying intent and selecting route",
        "Collecting RAG and SQL evidence",
        "Checking severity and guardrails",
        "Composing response from verified evidence",
    ]
    return steps[min(elapsed_seconds, len(steps) - 1)]


@router.get(
    "/diagnostics",
    summary="Retrieval diagnostics",
    description="Return non-secret feature flags, SLO thresholds, and cache settings for retrieval.",
)
async def retrieval_diagnostics(
    _current_user: AuthenticatedUser = Depends(validate_token),
) -> dict[str, Any]:
    """Return non-secret retrieval diagnostics for authenticated users."""

    return {
        "slo_thresholds_ms": SLO_THRESHOLDS_MS,
        "features": {
            "semantic_cache": settings.enable_semantic_cache,
            "llm_judge": settings.enable_llm_judge,
            "jira_mcp": settings.enable_jira_mcp,
            "reranking": settings.retrieval_enable_reranking,
            "auth0": settings.enable_auth0,
        },
        "cache": {
            "rag_ttl_seconds": settings.cache_rag_ttl_seconds,
            "sql_ttl_seconds": settings.cache_sql_ttl_seconds,
            "safety_critical_ttl_seconds": settings.cache_safety_critical_ttl_seconds,
            "similarity_threshold": settings.cache_similarity_threshold,
        },
    }


@router.post(
    "/query/stream",
    summary="Stream support query",
    description="Run the ERIS support workflow and stream status, node, delta, final, and error SSE events.",
)
async def query_retrieval_stream(
    request: RetrievalRequest,
    background_tasks: BackgroundTasks,
    current_user: AuthenticatedUser = Depends(require_permissions("ask:support_query")),
) -> StreamingResponse:
    """Stream LangGraph support orchestration events and the final verified answer."""

    async def event_stream():
        started_at = time.perf_counter()
        request_id = str(uuid4())
        answer_streamed = False
        yield _sse_message(
            "status",
            {
                "message": "Request accepted",
                "elapsed_seconds": 0,
            },
        )

        try:
            if request.mode != RetrievalMode.AGENT:
                task = asyncio.create_task(query_retrieval(request, background_tasks, current_user))
                progress_tick = 0
                while not task.done():
                    yield _sse_message(
                        "status",
                        {
                            "message": _stream_progress_message(progress_tick),
                            "elapsed_seconds": round(time.perf_counter() - started_at, 1),
                        },
                    )
                    progress_tick += 1
                    await asyncio.sleep(0.75)

                response = await task
                yield _sse_message(
                    "status",
                    {
                        "message": "Streaming verified answer",
                        "elapsed_seconds": round(time.perf_counter() - started_at, 1),
                    },
                )

                for chunk in _answer_chunks(response.answer):
                    yield _sse_message("delta", {"text": chunk})
                    await asyncio.sleep(0.035)

                yield _sse_message("final", response.model_dump(mode="json"))
                return

            authenticated_request = request.model_copy(
                update={
                    "user_id": current_user.sub,
                    "metadata": {**dict(request.metadata or {}), **auth_metadata(current_user)},
                }
            )
            guardrail_result = await apply_input_guardrails(authenticated_request, request_id)
            guarded_request = authenticated_request.model_copy(
                update={
                    "question": guardrail_result.sanitized_question,
                    "metadata": guardrail_result.metadata,
                }
            )
            request_metadata = _request_metadata(guarded_request)
            semantic_cache = get_semantic_cache()
            cache_route = cache_route_for_query(guarded_request.mode.value, guarded_request.question, request_metadata)
            cache_attributes = _cache_attributes(guarded_request, cache_route.value)
            cache_hit = await semantic_cache.get(guarded_request.question, cache_route, cache_attributes)
            if cache_hit and isinstance(cache_hit.response, dict):
                response = RetrievalResponse.model_validate(cache_hit.response).model_copy(
                    update={
                        "question": guarded_request.question,
                        "latency_ms": _latency_ms(started_at),
                    }
                )
                response = _with_cache_metadata(
                    response,
                    status="hit",
                    route=cache_route.value,
                    strategy=cache_hit.strategy,
                )
                response = apply_output_guardrails(response, request_id)
                response = _with_runtime_timing(response, started_at)
                response = _with_quality_slo_warnings(response)
                yield _sse_message("status", {"message": "Streaming cached answer", "elapsed_seconds": 0})
                for chunk in _answer_chunks(response.answer):
                    yield _sse_message("delta", {"text": chunk})
                    await asyncio.sleep(0.02)
                yield _sse_message("final", response.model_dump(mode="json"))
                return

            service = RetrievalService()
            final_result: dict[str, Any] | None = None
            async for graph_event in service.stream_agent_with_evidence(
                guarded_request.question,
                metadata=request_metadata,
            ):
                node = str(graph_event.get("node") or "")
                result = dict(graph_event.get("result") or {})
                final_result = result
                yield _sse_message(
                    "node",
                    {
                        "node": node,
                        "message": f"{node.replace('_', ' ').title()} completed",
                        "elapsed_seconds": round(time.perf_counter() - started_at, 1),
                        "route_decision": result.get("route_decision"),
                        "severity": result.get("severity"),
                    },
                )
                if node in {"response_composer", "escalation_manager"} and result.get("answer") and not answer_streamed:
                    partial_response = _agent_response(guarded_request, result, started_at, current_user)
                    partial_response = apply_output_guardrails(partial_response, request_id)
                    partial_response = _with_quality_slo_warnings(partial_response)
                    yield _sse_message(
                        "status",
                        {
                            "message": "Streaming verified answer",
                            "elapsed_seconds": round(time.perf_counter() - started_at, 1),
                        },
                    )
                    for chunk in _answer_chunks(partial_response.answer):
                        yield _sse_message("delta", {"text": chunk})
                        await asyncio.sleep(0.035)
                    answer_streamed = True

            if final_result is None:
                raise RuntimeError("LangGraph stream ended without a final result")

            response = _agent_response(guarded_request, final_result, started_at, current_user)
            response = apply_output_guardrails(response, request_id)
            response = _with_runtime_timing(response, started_at)
            response = _with_quality_slo_warnings(response)
            trace_id = str(final_result.get("langfuse_trace_id") or "")
            if trace_id:
                background_tasks.add_task(
                    attach_runtime_slo_scores,
                    trace_id,
                    guarded_request.question,
                    response.model_dump(),
                )
            if response.success and not response.error:
                store_result = await semantic_cache.set(
                    guarded_request.question,
                    _cacheable_response_payload(response),
                    cache_route,
                    cache_attributes,
                )
                if store_result.stored:
                    response = _with_cache_metadata(response, status="stored", route=cache_route.value)
                else:
                    response = _with_cache_metadata(response, status="miss", route=cache_route.value)
            else:
                response = _with_cache_metadata(response, status="bypass", route=cache_route.value)

            yield _sse_message(
                "final",
                response.model_dump(mode="json"),
            )
        except GuardrailViolation as exc:
            yield _sse_message(
                "error",
                {
                    "status_code": exc.status_code,
                    "error": exc.error,
                    "message": exc.detail,
                    "retry_after": exc.retry_after,
                },
            )
        except HTTPException as exc:
            detail = exc.detail
            if isinstance(detail, dict):
                message = detail.get("detail") or detail.get("error") or str(detail)
            else:
                message = str(detail)
            yield _sse_message(
                "error",
                {
                    "status_code": exc.status_code,
                    "message": message,
                },
            )
        except asyncio.CancelledError:
            task.cancel()
            raise
        except Exception as exc:
            logger.exception("streaming retrieval query failed")
            yield _sse_message(
                "error",
                {
                    "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "message": "Unexpected server error while streaming the retrieval response.",
                },
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        background=background_tasks,
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/query",
    summary="Run support query",
    description="Run a support query synchronously through vector, SQL, or agent orchestration.",
    response_model=RetrievalResponse,
    responses={
        401: {"model": RetrievalErrorResponse},
        403: {"model": RetrievalErrorResponse},
        400: {"model": RetrievalErrorResponse},
        503: {"model": RetrievalErrorResponse},
        429: {"model": RetrievalErrorResponse},
        500: {"model": RetrievalErrorResponse},
    },
)
async def query_retrieval(
    request: RetrievalRequest,
    background_tasks: BackgroundTasks,
    current_user: AuthenticatedUser = Depends(require_permissions("ask:support_query")),
) -> RetrievalResponse:
    """Run a support query and return the final retrieval response."""

    started_at = time.perf_counter()
    request_id = str(uuid4())
    logger.info("retrieval query received request_id=%s mode=%s", request_id, request.mode.value)

    try:
        authenticated_request = request.model_copy(
            update={
                "user_id": current_user.sub,
                "metadata": {**dict(request.metadata or {}), **auth_metadata(current_user)},
            }
        )
        guardrail_result = await apply_input_guardrails(authenticated_request, request_id)
        guarded_request = authenticated_request.model_copy(
            update={
                "question": guardrail_result.sanitized_question,
                "metadata": guardrail_result.metadata,
            }
        )
        request_metadata = _request_metadata(guarded_request)
        semantic_cache = get_semantic_cache()
        cache_route = cache_route_for_query(guarded_request.mode.value, guarded_request.question, request_metadata)
        cache_attributes = _cache_attributes(guarded_request, cache_route.value)
        cache_hit = await semantic_cache.get(guarded_request.question, cache_route, cache_attributes)
        if cache_hit and isinstance(cache_hit.response, dict):
            response = RetrievalResponse.model_validate(cache_hit.response).model_copy(
                update={
                    "question": guarded_request.question,
                    "latency_ms": _latency_ms(started_at),
                }
            )
            response = _with_cache_metadata(
                response,
                status="hit",
                route=cache_route.value,
                strategy=cache_hit.strategy,
            )
            response = apply_output_guardrails(response, request_id)
            response = _with_runtime_timing(response, started_at)
            response = _with_quality_slo_warnings(response)
            logger.info(
                "retrieval query served from semantic cache request_id=%s route=%s strategy=%s latency_ms=%s",
                request_id,
                cache_route.value,
                cache_hit.strategy,
                response.latency_ms,
            )
            return response

        service = RetrievalService()
        if guarded_request.mode == RetrievalMode.VECTOR:
            result = await service.query_vector(guarded_request.question)
            response = _vector_response(guarded_request, result, started_at)
        elif guarded_request.mode == RetrievalMode.SQL:
            result = await service.query_sql(guarded_request.question)
            response = _sql_response(guarded_request, result, started_at, current_user)
        else:
            result = await service.query_agent_with_evidence(
                guarded_request.question,
                metadata=request_metadata,
            )
            response = _agent_response(guarded_request, result, started_at, current_user)

        response = apply_output_guardrails(response, request_id)
        response = _with_runtime_timing(response, started_at)
        response = _with_quality_slo_warnings(response)
        trace_id = str(result.get("langfuse_trace_id") or "") if isinstance(result, dict) else ""
        if trace_id:
            background_tasks.add_task(
                attach_runtime_slo_scores,
                trace_id,
                guarded_request.question,
                response.model_dump(),
            )
        if response.success and not response.error:
            store_result = await semantic_cache.set(
                guarded_request.question,
                _cacheable_response_payload(response),
                cache_route,
                cache_attributes,
            )
            if store_result.stored:
                response = _with_cache_metadata(response, status="stored", route=cache_route.value)
                logger.info(
                    "retrieval response stored in semantic cache request_id=%s route=%s",
                    request_id,
                    cache_route.value,
                )
            else:
                response = _with_cache_metadata(response, status="miss", route=cache_route.value)
        else:
            response = _with_cache_metadata(response, status="bypass", route=cache_route.value)

        logger.info(
            "retrieval query completed request_id=%s mode=%s success=%s latency_ms=%s",
            request_id,
            guarded_request.mode.value,
            response.success,
            response.latency_ms,
        )
        return response
    except GuardrailViolation as exc:
        logger.warning("retrieval query blocked by input guardrail request_id=%s error=%s", request_id, exc.error)
        headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after is not None else None
        raise HTTPException(
            status_code=exc.status_code,
            detail=RetrievalErrorResponse(error=exc.error, detail=exc.detail).model_dump(),
            headers=headers,
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=RetrievalErrorResponse(
                error="invalid_retrieval_request",
                detail="The retrieval request is invalid.",
            ).model_dump(),
        ) from exc
    except Exception as exc:
        logger.exception("retrieval query failed request_id=%s mode=%s", request_id, request.mode.value)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=RetrievalErrorResponse(
                error="retrieval_runtime_error",
                detail="Unexpected server error while processing the retrieval request.",
            ).model_dump(),
        ) from exc
