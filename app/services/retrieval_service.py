from typing import Any, Dict

from collections.abc import AsyncIterator

from app.agents.graph import run_support_graph, stream_support_graph
from app.services.tools.sql_tool import execute_nl2sql
from app.services.tools.vector_tool import execute_vector_retrieval


def _enum_value(value: Any) -> str | None:
    if value is None:
        return None
    return getattr(value, "value", str(value))


def _source_nodes_from_state(state: dict[str, Any]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for chunk in state.get("retrieved_chunks", []) or []:
        metadata = dict(chunk.get("metadata") or {})
        nodes.append(
            {
                "chunk_text": chunk.get("chunk_text") or "",
                "source_file": chunk.get("source_file") or metadata.get("source_file") or metadata.get("source"),
                "page_number": chunk.get("page_number") or metadata.get("page_number"),
                "content_type": chunk.get("content_type") or metadata.get("content_type"),
                "category": chunk.get("category") or metadata.get("category"),
                "similarity_score": chunk.get("score"),
                "metadata": metadata,
            }
        )
    return nodes


def _structured_result_from_state(state: dict[str, Any]) -> dict[str, Any] | None:
    sql_results = list(state.get("sql_results") or [])
    if not sql_results:
        return None
    result = dict(sql_results[0])
    return {
        "answer": result.get("answer") or "",
        "sql_query": result.get("sql_query"),
        "table_used": ", ".join(result.get("tables_used", []) or []),
        "row_count": int(result.get("row_count") or 0),
        "raw_results": result.get("raw_results", []),
    }


def _incident_investigation_from_state(state: dict[str, Any]) -> dict[str, Any] | None:
    investigation = dict(state.get("incident_investigation") or {})
    if not investigation:
        return None

    matched_incidents = []
    for incident in list(investigation.get("matched_incidents") or []):
        raw_incident = dict(incident)
        matched_incidents.append(
            {
                "incident_id": raw_incident.get("incident_id"),
                "incident_type": raw_incident.get("incident_type"),
                "severity": raw_incident.get("severity"),
                "affected_region": raw_incident.get("affected_region"),
                "start_time": raw_incident.get("start_time"),
                "end_time": raw_incident.get("end_time"),
                "resolution_status": raw_incident.get("resolution_status"),
                "root_cause": raw_incident.get("root_cause"),
                "escalation_flag": bool(raw_incident.get("escalation_flag")),
                "correlation_score": raw_incident.get("correlation_score"),
                "correlation_reasons": list(raw_incident.get("correlation_reasons") or []),
            }
        )

    return {
        "filters_used": dict(investigation.get("filters_used") or {}),
        "matched_incidents": matched_incidents,
        "active_critical_incident": bool(investigation.get("active_critical_incident")),
        "active_critical_incident_correlated": bool(investigation.get("active_critical_incident_correlated")),
        "max_correlation_score": float(investigation.get("max_correlation_score") or 0.0),
        "correlation_threshold": float(investigation.get("correlation_threshold") or 0.65),
        "investigation_summary": str(investigation.get("investigation_summary") or ""),
    }


def _agent_trace_from_state(state: dict[str, Any]) -> list[dict[str, Any]]:
    trace_events: list[dict[str, Any]] = []
    for event in list(state.get("agent_trace") or [])[-50:]:
        raw_event = dict(event)
        agent_name = str(raw_event.get("agent_name") or "").strip()
        if not agent_name:
            continue
        latency = raw_event.get("latency_ms")
        trace_events.append(
            {
                "agent_name": agent_name,
                "action": str(raw_event.get("action") or "") or None,
                "input_summary": str(raw_event.get("input_summary") or "") or None,
                "output_summary": str(raw_event.get("output_summary") or "") or None,
                "status": str(raw_event.get("status") or "") or None,
                "timestamp": str(raw_event.get("timestamp") or "") or None,
                "latency_ms": int(latency) if isinstance(latency, (int, float)) else None,
            }
        )
    return trace_events


def _graph_state_to_result(state: dict[str, Any]) -> Dict[str, Any]:
    source_nodes = _source_nodes_from_state(state)
    tools_used = []
    if source_nodes:
        tools_used.append("document_retrieval")
    if state.get("sql_results"):
        tools_used.append("sql_agent")
    if state.get("customer_context"):
        tools_used.append("account_validator")
    if state.get("incident_investigation"):
        tools_used.append("incident_investigator")
    if state.get("jira_tracking_result"):
        tools_used.append("jira_mcp_tool")
    if state.get("email_notification_result"):
        tools_used.append("email_mcp_tool")
    errors = list(state.get("errors") or [])
    sql_errors = [
        str(result.get("error"))
        for result in list(state.get("sql_results") or [])
        if result.get("error")
    ]
    answer = str(state.get("final_answer") or "")
    error = "; ".join(sql_errors or [str(error) for error in errors]) or None
    metadata = dict(state.get("metadata") or {})
    return {
        "answer": answer,
        "citations": list(state.get("citations") or []),
        "source_nodes": source_nodes,
        "chunk_count": len(source_nodes),
        "structured_result": _structured_result_from_state(state),
        "incident_investigation": _incident_investigation_from_state(state),
        "jira_tracking_result": dict(state.get("jira_tracking_result") or {}),
        "email_notification_result": dict(state.get("email_notification_result") or {}),
        "route_decision": _enum_value(state.get("route_decision")),
        "intent": _enum_value(state.get("intent")),
        "severity": _enum_value(state.get("severity")),
        "confidence_score": state.get("confidence_score"),
        "escalation_flag": bool(state.get("escalation_flag")),
        "escalation_target": _enum_value(state.get("escalation_target")),
        "tools_used": tools_used,
        "agent_trace": _agent_trace_from_state(state),
        "suggested_questions": list(metadata.get("clarification_suggestions") or []),
        "answer_quality": metadata.get("answer_quality"),
        "final_answer_model": metadata.get("final_answer_model"),
        "error": error if sql_errors or not answer else None,
        "langfuse_trace_id": state.get("langfuse_trace_id"),
    }


class RetrievalService:
    """Facade for agent-orchestrated and direct retrieval paths."""

    async def query(self, question: str, metadata: dict[str, Any] | None = None) -> Dict[str, Any]:
        state = await run_support_graph(question, metadata=metadata)
        return _graph_state_to_result(dict(state))

    async def query_agent_with_evidence(self, question: str, metadata: dict[str, Any] | None = None) -> Dict[str, Any]:
        return await self.query(question, metadata=metadata)

    async def stream_agent_with_evidence(
        self,
        question: str,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        async for event in stream_support_graph(question, metadata=metadata):
            state = dict(event.get("state") or {})
            yield {
                "type": event.get("event"),
                "node": event.get("node"),
                "result": _graph_state_to_result(state),
                "state": state,
            }

    async def query_vector(self, question: str) -> Dict[str, Any]:
        return await execute_vector_retrieval(question)

    async def query_sql(self, question: str) -> Dict[str, Any]:
        return await execute_nl2sql(question)


async def retrieve(question: str) -> Dict[str, Any]:
    return await RetrievalService().query(question)
