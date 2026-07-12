import asyncio
import inspect
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from collections.abc import AsyncIterator
from functools import wraps
from typing import Any
from uuid import uuid4

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, StateGraph
import psycopg

from app.agents.account_validator_agent import validate_account
from app.agents.doc_retrieval_agent import retrieve_documents
from app.agents.escalation_agent import manage_escalation
from app.agents.incident_investigator_agent import investigate_incidents
from app.agents.response_composer_agent import compose_response
from app.agents.response_validator_agent import validate_response
from app.agents.sql_agent import query_structured_data
from app.core.config import settings
from app.core.langfuse import get_langfuse_client, observe, trace_agent_state
from app.core.logging import get_logger
from app.services.agent_service import execute_high_risk_evidence_parallel, execute_hybrid_tools_parallel
from app.orchestration.intent_classifier import classify_intent
from app.orchestration.severity_assessor import assess_severity
from app.orchestration.state import (
    AgentResult,
    EscalationTarget,
    MAX_ORCHESTRATION_ITERATIONS,
    RouteDecision,
    SeverityPriority,
    SupportOrchestrationState,
    VerificationOutcome,
)

_GRAPH = None
_CHECKPOINTER_CONTEXT = None
_CHECKPOINTER = None
_GRAPH_LOCK = asyncio.Lock()
CHECKPOINT_NAMESPACE = "support_orchestration"
CHECKPOINT_ALLOWED_MSGPACK_MODULES = [
    ("app.orchestration.state", "EscalationTarget"),
    ("app.orchestration.state", "RouteDecision"),
    ("app.orchestration.state", "SeverityLevel"),
    ("app.orchestration.state", "SeverityPriority"),
    ("app.orchestration.state", "SupportIntent"),
]
logger = get_logger(__name__)

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_list(state: SupportOrchestrationState, key: str, item: Any) -> None:
    values = list(state.get(key, []))
    values.append(item)
    state[key] = values  # type: ignore[literal-required]


def _route_value(state: SupportOrchestrationState) -> str:
    route = state.get("route_decision")
    if isinstance(route, RouteDecision):
        return route.value
    if route is None:
        return RouteDecision.CLARIFICATION.value
    return str(route)


def _state_log_summary(state: SupportOrchestrationState | Any) -> dict[str, Any]:
    if not isinstance(state, dict):
        return {"state_type": type(state).__name__}
    query = str(state.get("query") or "")
    return {
        "query_length": len(query),
        "route_decision": _route_value(state),
        "intent": getattr(state.get("intent"), "value", state.get("intent")),
        "confidence_score": state.get("confidence_score"),
        "severity": getattr(state.get("severity"), "value", state.get("severity")),
        "severity_priority": getattr(state.get("severity_priority"), "value", state.get("severity_priority")),
        "escalation_flag": bool(state.get("escalation_flag")),
    }


def _estimate_tokens(value: Any) -> int:
    text = str(value or "")
    return max(1, len(text) // 4) if text else 0


def _budget_metadata(state: SupportOrchestrationState) -> dict[str, Any]:
    metadata = dict(state.get("metadata") or {})
    budget = dict(metadata.get("orchestration_budget") or {})
    budget.setdefault("max_agent_calls", settings.orchestration_max_agent_calls)
    budget.setdefault("runtime_budget_ms", settings.orchestration_runtime_budget_ms)
    budget.setdefault("token_budget", settings.orchestration_token_budget)
    budget.setdefault("agent_calls_used", 0)
    budget.setdefault("token_estimate_used", 0)
    budget.setdefault("started_at_perf", time.perf_counter())
    metadata["orchestration_budget"] = budget
    return metadata


def _budget_status(state: SupportOrchestrationState) -> tuple[bool, str | None]:
    metadata = _budget_metadata(state)
    budget = dict(metadata.get("orchestration_budget") or {})
    elapsed_ms = int((time.perf_counter() - float(budget.get("started_at_perf") or time.perf_counter())) * 1000)
    if int(budget.get("agent_calls_used") or 0) >= int(budget.get("max_agent_calls") or 0):
        return False, "Agent call budget exhausted."
    if elapsed_ms >= int(budget.get("runtime_budget_ms") or 0):
        return False, "Runtime budget exhausted."
    if int(budget.get("token_estimate_used") or 0) >= int(budget.get("token_budget") or 0):
        return False, "Token budget estimate exhausted."
    return True, None


def _append_agent_result(
    state: SupportOrchestrationState,
    *,
    agent_name: str,
    status: str,
    summary: str,
    latency_ms: int | None,
    error: str | None = None,
) -> None:
    result: AgentResult = {
        "agent_name": agent_name,
        "status": status,
        "route_decision": _route_value(state),
        "summary": summary,
        "data": _state_log_summary(state),
        "confidence_score": state.get("confidence_score"),
        "latency_ms": latency_ms,
        "token_estimate": _estimate_tokens(summary),
        "error": error,
        "timestamp": _utc_now(),
    }
    _append_list(state, "agent_results", result)


def _mark_budget_exceeded(state: SupportOrchestrationState, node_name: str, reason: str) -> SupportOrchestrationState:
    next_state: SupportOrchestrationState = dict(state)
    metadata = dict(next_state.get("metadata") or {})
    metadata["budget_exceeded"] = True
    metadata["budget_exceeded_reason"] = reason
    metadata["budget_exceeded_at_node"] = node_name
    next_state["metadata"] = metadata
    next_state["escalation_flag"] = True
    next_state["escalation_target"] = EscalationTarget.L2_SUPPORT
    existing_reason = next_state.get("escalation_reason")
    budget_reason = f"{reason} Stopped before {node_name} to avoid runaway orchestration."
    next_state["escalation_reason"] = f"{existing_reason}; {budget_reason}" if existing_reason else budget_reason
    _append_list(next_state, "errors", budget_reason)
    _append_agent_result(
        next_state,
        agent_name=node_name,
        status="skipped_budget_exceeded",
        summary=budget_reason,
        latency_ms=0,
        error=reason,
    )
    return next_state


def _record_budget_usage(state: SupportOrchestrationState, node_name: str, result: SupportOrchestrationState, latency_ms: int) -> None:
    metadata = _budget_metadata(result)
    budget = dict(metadata.get("orchestration_budget") or {})
    budget["agent_calls_used"] = int(budget.get("agent_calls_used") or 0) + 1
    token_estimate = _estimate_tokens(result.get("query")) + _estimate_tokens(_state_log_summary(result))
    budget["token_estimate_used"] = int(budget.get("token_estimate_used") or 0) + token_estimate
    metadata["orchestration_budget"] = budget
    result["metadata"] = metadata
    _append_agent_result(
        result,
        agent_name=node_name,
        status="completed",
        summary=f"{node_name} completed within budget.",
        latency_ms=latency_ms,
    )


def _logged_node(node_name: str, node_func: Any) -> Any:
    """Wrap a LangGraph node with summary logging without changing node behavior."""

    if inspect.iscoroutinefunction(node_func):

        @wraps(node_func)
        async def async_wrapper(state: SupportOrchestrationState) -> SupportOrchestrationState:
            started = time.perf_counter()
            if node_name not in {"response_composer", "escalation_manager"}:
                allowed, reason = _budget_status(state)
                if not allowed and reason:
                    return _mark_budget_exceeded(state, node_name, reason)
            logger.info("agent_node_started", extra={"agent_node": node_name, **_state_log_summary(state)})
            try:
                result = await node_func(state)
            except Exception:
                logger.error(
                    "agent_node_failed",
                    extra={"agent_node": node_name, "duration_ms": int((time.perf_counter() - started) * 1000)},
                    exc_info=True,
                )
                raise
            latency_ms = int((time.perf_counter() - started) * 1000)
            _record_budget_usage(state, node_name, result, latency_ms)
            logger.info(
                "agent_node_completed",
                extra={"agent_node": node_name, "duration_ms": latency_ms, **_state_log_summary(result)},
            )
            return result

        return async_wrapper

    @wraps(node_func)
    def sync_wrapper(state: SupportOrchestrationState) -> SupportOrchestrationState:
        started = time.perf_counter()
        if node_name not in {"response_composer", "escalation_manager"}:
            allowed, reason = _budget_status(state)
            if not allowed and reason:
                return _mark_budget_exceeded(state, node_name, reason)
        logger.info("agent_node_started", extra={"agent_node": node_name, **_state_log_summary(state)})
        try:
            result = node_func(state)
        except Exception:
            logger.error(
                "agent_node_failed",
                extra={"agent_node": node_name, "duration_ms": int((time.perf_counter() - started) * 1000)},
                exc_info=True,
            )
            raise
        latency_ms = int((time.perf_counter() - started) * 1000)
        _record_budget_usage(state, node_name, result, latency_ms)
        logger.info(
            "agent_node_completed",
            extra={"agent_node": node_name, "duration_ms": latency_ms, **_state_log_summary(result)},
        )
        return result

    return sync_wrapper


def _is_escalated(state: SupportOrchestrationState) -> bool:
    return bool(state.get("escalation_flag")) or state.get("severity_priority") == SeverityPriority.P0


def _last_response_validation(state: SupportOrchestrationState) -> VerificationOutcome | None:
    for outcome in reversed(list(state.get("verification_outcomes") or [])):
        if outcome.get("check_name") == "response_validation_complete":
            return outcome
    return None


def _failed_response_checks(outcome: VerificationOutcome | None) -> list[str]:
    if not outcome:
        return ["response validation did not run"]
    metadata = dict(outcome.get("metadata") or {})
    failed = []
    if metadata.get("retrieved_chunks_present") is False:
        failed.append("retrieved_chunks_present")
    if metadata.get("confidence_passed") is False:
        failed.append("confidence_threshold_check")
    if metadata.get("citations_present") is False:
        failed.append("citations_present_og_1")
    return failed or [str(outcome.get("reason") or "response validation failed")]


def _initialize_state(query: str, metadata: dict[str, Any] | None) -> SupportOrchestrationState:
    return {
        "query": query,
        "conversation_id": (metadata or {}).get("conversation_id"),
        "user_id": (metadata or {}).get("user_id"),
        "metadata": dict(metadata or {}),
        "execution_plan": [],
        "progress_updates": [],
        "execution_results": [],
        "agent_results": [],
        "verification_outcomes": [],
        "iteration_count": 1,
        "max_iterations": MAX_ORCHESTRATION_ITERATIONS,
        "iteration_history": [],
        "retrieved_chunks": [],
        "sql_results": [],
        "hybrid_result": None,
        "confidence_score": None,
        "guardrail_flags": [],
        "escalation_flag": False,
        "final_answer": None,
        "citations": [],
        "recommended_actions": [],
        "agent_trace": [],
        "errors": [],
        "latency_ms": None,
        # Explicitly clear per-turn fields so checkpointed conversations do not
        # reuse incident/customer/escalation evidence from a previous query.
        "intent": None,
        "route_decision": None,
        "severity_priority": None,
        "severity": None,
        "severity_reason": None,
        "customer_context": {},
        "incident_investigation": {},
        "escalation_target": EscalationTarget.NONE,
        "escalation_reason": None,
        "jira_tracking_result": {},
        "jira_issue_key": None,
        "jira_issue_url": None,
    }


def _route_after_intent(state: SupportOrchestrationState) -> str:
    route = _route_value(state)
    if route in {
        RouteDecision.RAG.value,
        RouteDecision.SQL.value,
        RouteDecision.HYBRID.value,
        RouteDecision.HIGH_RISK.value,
        RouteDecision.CLARIFICATION.value,
        RouteDecision.CHITCHAT.value,
    }:
        return route
    return RouteDecision.CLARIFICATION.value


def _route_after_account(state: SupportOrchestrationState) -> str:
    route = _route_value(state)
    if route == RouteDecision.HIGH_RISK.value:
        return "high_risk_parallel_tools"
    if route == RouteDecision.HYBRID.value:
        return "hybrid_parallel_tools"
    if route == RouteDecision.SQL.value:
        return "sql_agent"
    return "document_retrieval"


def _route_after_document(state: SupportOrchestrationState) -> str:
    if _route_value(state) == RouteDecision.HYBRID.value:
        return "sql_agent"
    return "severity_assessor"


def _sql_validation_failed(state: SupportOrchestrationState) -> bool:
    for outcome in reversed(list(state.get("verification_outcomes") or [])):
        if outcome.get("check_name") == "sql_query_validated":
            return not bool(outcome.get("passed"))
    return False


def _route_after_sql(state: SupportOrchestrationState) -> str:
    if _sql_validation_failed(state):
        return "severity_assessor"
    if _route_value(state) == RouteDecision.HYBRID.value:
        return "severity_assessor"
    return "severity_assessor"


def _needs_incident_investigation(state: SupportOrchestrationState) -> bool:
    if _route_value(state) == RouteDecision.HIGH_RISK.value:
        return True

    query = str(state.get("query") or "").lower()
    incident_terms = (
        "incident",
        "outage",
        "breach",
        "unauthorized",
        "data loss",
        "production down",
        "critical",
        "p0",
        "p1",
        "sev",
        "security vulnerability",
        "active investigation",
    )
    if any(term in query for term in incident_terms):
        return True

    metadata = dict(state.get("metadata") or {})
    forced_route = str(metadata.get("forced_route_decision") or "").lower()
    if forced_route == RouteDecision.HIGH_RISK.value:
        return True

    customer_context = dict(state.get("customer_context") or {})
    lookup_reason = str(customer_context.get("lookup_reason") or "").lower()
    return any(term in lookup_reason for term in ("incident", "outage", "security", "critical"))


def _route_after_hybrid_parallel(state: SupportOrchestrationState) -> str:
    if _sql_validation_failed(state):
        return "escalation_manager"
    if _needs_incident_investigation(state):
        return "incident_investigator"
    return "severity_assessor"


def _route_after_high_risk_parallel(state: SupportOrchestrationState) -> str:
    if _sql_validation_failed(state):
        return "escalation_manager"
    return "severity_assessor"


def _route_after_incident(state: SupportOrchestrationState) -> str:
    return "severity_assessor"


def _route_after_severity(state: SupportOrchestrationState) -> str:
    if _is_escalated(state):
        return "escalation_manager"
    if _route_value(state) == RouteDecision.SQL.value:
        return "response_composer"
    return "response_validator"


def _reflection_check(state: SupportOrchestrationState) -> SupportOrchestrationState:
    next_state: SupportOrchestrationState = dict(state)
    metadata = dict(next_state.get("metadata") or {})
    validation = _last_response_validation(next_state)

    if validation and validation.get("passed"):
        metadata["_graph_next"] = "response_composer"
        next_state["metadata"] = metadata
        return next_state

    if _is_escalated(next_state):
        metadata["_graph_next"] = "escalation_manager"
        next_state["metadata"] = metadata
        return next_state

    iteration_count = int(next_state.get("iteration_count") or 1)
    max_iterations = int(next_state.get("max_iterations") or MAX_ORCHESTRATION_ITERATIONS)
    failed_checks = _failed_response_checks(validation)

    if iteration_count < max_iterations:
        next_iteration = iteration_count + 1
        route = _route_value(next_state)
        if route == RouteDecision.HYBRID.value:
            retry_target = "hybrid_parallel_tools"
        elif route == RouteDecision.HIGH_RISK.value:
            retry_target = "high_risk_parallel_tools"
        elif route == RouteDecision.SQL.value:
            retry_target = "sql_agent"
        else:
            retry_target = "document_retrieval"
        metadata["_graph_next"] = retry_target
        next_state["metadata"] = metadata
        next_state["iteration_count"] = next_iteration
        _append_list(
            next_state,
            "iteration_history",
            {
                "iteration_number": next_iteration,
                "plan_summary": "Retry evidence collection after response validation failure.",
                "completed_steps": [],
                "failed_steps": failed_checks,
                "verification_outcomes": [validation] if validation else [],
                "correction_summary": f"Retrying via {retry_target}.",
                "should_retry": True,
                "timestamp": _utc_now(),
            },
        )
        return next_state

    next_state["escalation_flag"] = True
    existing_reason = next_state.get("escalation_reason")
    reason = "Response validation failed after max orchestration iterations."
    next_state["escalation_reason"] = f"{existing_reason}; {reason}" if existing_reason else reason
    metadata["_graph_next"] = "escalation_manager"
    next_state["metadata"] = metadata
    _append_list(
        next_state,
        "iteration_history",
        {
            "iteration_number": iteration_count,
            "plan_summary": "Stop reflection loop after max iterations.",
            "completed_steps": [],
            "failed_steps": failed_checks,
            "verification_outcomes": [validation] if validation else [],
            "correction_summary": reason,
            "should_retry": False,
            "timestamp": _utc_now(),
        },
    )
    return next_state


def _route_after_reflection(state: SupportOrchestrationState) -> str:
    metadata = dict(state.get("metadata") or {})
    next_node = str(metadata.get("_graph_next") or "escalation_manager")
    if next_node in {
        "response_composer",
        "escalation_manager",
        "document_retrieval",
        "sql_agent",
        "account_validator",
        "hybrid_parallel_tools",
        "high_risk_parallel_tools",
    }:
        return next_node
    return "escalation_manager"


def build_support_graph(checkpointer: Any | None = None):
    """Build the LangGraph workflow for support orchestration."""

    workflow = StateGraph(SupportOrchestrationState)

    workflow.add_node("intent_classifier", _logged_node("intent_classifier", classify_intent))
    workflow.add_node("severity_assessor", _logged_node("severity_assessor", assess_severity))
    workflow.add_node("document_retrieval", _logged_node("document_retrieval", retrieve_documents))
    workflow.add_node("sql_agent", _logged_node("sql_agent", query_structured_data))
    workflow.add_node("hybrid_parallel_tools", _logged_node("hybrid_parallel_tools", execute_hybrid_tools_parallel))
    workflow.add_node(
        "high_risk_parallel_tools",
        _logged_node("high_risk_parallel_tools", execute_high_risk_evidence_parallel),
    )
    workflow.add_node("account_validator", _logged_node("account_validator", validate_account))
    workflow.add_node("incident_investigator", _logged_node("incident_investigator", investigate_incidents))
    workflow.add_node("response_validator", _logged_node("response_validator", validate_response))
    workflow.add_node("response_composer", _logged_node("response_composer", compose_response))
    workflow.add_node("escalation_manager", _logged_node("escalation_manager", manage_escalation))
    workflow.add_node("reflection_check", _logged_node("reflection_check", _reflection_check))

    workflow.set_entry_point("intent_classifier")

    workflow.add_conditional_edges(
        "intent_classifier",
        _route_after_intent,
        {
            RouteDecision.RAG.value: "document_retrieval",
            RouteDecision.SQL.value: "sql_agent",
            RouteDecision.HYBRID.value: "account_validator",
            RouteDecision.HIGH_RISK.value: "account_validator",
            RouteDecision.CLARIFICATION.value: "response_composer",
            RouteDecision.CHITCHAT.value: "response_composer",
        },
    )
    workflow.add_conditional_edges(
        "account_validator",
        _route_after_account,
        {
            "high_risk_parallel_tools": "high_risk_parallel_tools",
            "hybrid_parallel_tools": "hybrid_parallel_tools",
            "sql_agent": "sql_agent",
            "document_retrieval": "document_retrieval",
        },
    )
    workflow.add_conditional_edges(
        "document_retrieval",
        _route_after_document,
        {
            "sql_agent": "sql_agent",
            "severity_assessor": "severity_assessor",
        },
    )
    workflow.add_conditional_edges(
        "sql_agent",
        _route_after_sql,
        {
            "escalation_manager": "escalation_manager",
            "incident_investigator": "incident_investigator",
            "severity_assessor": "severity_assessor",
        },
    )
    workflow.add_conditional_edges(
        "hybrid_parallel_tools",
        _route_after_hybrid_parallel,
        {
            "escalation_manager": "escalation_manager",
            "incident_investigator": "incident_investigator",
            "severity_assessor": "severity_assessor",
        },
    )
    workflow.add_conditional_edges(
        "high_risk_parallel_tools",
        _route_after_high_risk_parallel,
        {
            "escalation_manager": "escalation_manager",
            "severity_assessor": "severity_assessor",
        },
    )
    workflow.add_conditional_edges(
        "incident_investigator",
        _route_after_incident,
        {
            "escalation_manager": "escalation_manager",
            "severity_assessor": "severity_assessor",
        },
    )
    workflow.add_conditional_edges(
        "severity_assessor",
        _route_after_severity,
        {
            "escalation_manager": "escalation_manager",
            "response_validator": "response_validator",
            "response_composer": "response_composer",
        },
    )
    workflow.add_edge("response_validator", "reflection_check")
    workflow.add_conditional_edges(
        "reflection_check",
        _route_after_reflection,
        {
            "response_composer": "response_composer",
            "escalation_manager": "escalation_manager",
            "document_retrieval": "document_retrieval",
            "sql_agent": "sql_agent",
            "hybrid_parallel_tools": "hybrid_parallel_tools",
            "high_risk_parallel_tools": "high_risk_parallel_tools",
            "account_validator": "account_validator",
        },
    )

    workflow.add_edge("response_composer", END)
    workflow.add_edge("escalation_manager", END)

    return workflow.compile(checkpointer=checkpointer)


async def _get_postgres_checkpointer() -> AsyncPostgresSaver:
    """Create and cache the Neon Postgres-backed async checkpointer."""

    global _CHECKPOINTER_CONTEXT, _CHECKPOINTER

    if _CHECKPOINTER is not None:
        return _CHECKPOINTER

    logger.info("initializing LangGraph AsyncPostgresSaver checkpointer")
    serde = JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_ALLOWED_MSGPACK_MODULES)
    _CHECKPOINTER_CONTEXT = AsyncPostgresSaver.from_conn_string(settings.database_dsn, serde=serde)
    _CHECKPOINTER = await _CHECKPOINTER_CONTEXT.__aenter__()
    await _CHECKPOINTER.setup()
    logger.info("LangGraph AsyncPostgresSaver checkpointer ready")
    return _CHECKPOINTER


async def get_support_graph():
    """Return a cached compiled support graph with Neon Postgres checkpointing."""

    global _GRAPH
    async with _GRAPH_LOCK:
        if _GRAPH is None:
            checkpointer = await _get_postgres_checkpointer()
            _GRAPH = build_support_graph(checkpointer=checkpointer)
    return _GRAPH


async def close_support_graph_checkpointer() -> None:
    """Close the cached checkpointer connection during application shutdown."""

    global _GRAPH, _CHECKPOINTER, _CHECKPOINTER_CONTEXT

    if _CHECKPOINTER_CONTEXT is not None:
        await _CHECKPOINTER_CONTEXT.__aexit__(None, None, None)
    _GRAPH = None
    _CHECKPOINTER = None
    _CHECKPOINTER_CONTEXT = None


async def reset_support_graph_checkpointer() -> None:
    """Drop cached graph/checkpointer so the next request opens a fresh Postgres connection."""

    global _GRAPH, _CHECKPOINTER, _CHECKPOINTER_CONTEXT

    async with _GRAPH_LOCK:
        if _CHECKPOINTER_CONTEXT is not None:
            await _CHECKPOINTER_CONTEXT.__aexit__(None, None, None)
        _GRAPH = None
        _CHECKPOINTER = None
        _CHECKPOINTER_CONTEXT = None


def _is_checkpointer_connection_error(exc: Exception) -> bool:
    if isinstance(exc, psycopg.OperationalError):
        return True
    message = str(exc).lower()
    return any(
        indicator in message
        for indicator in (
            "connection is closed",
            "ssl connection has been closed",
            "could not receive data from server",
            "consuming input failed",
        )
    )


def _thread_id(metadata: dict[str, Any]) -> str:
    return str(
        metadata.get("thread_id")
        or metadata.get("conversation_id")
        or metadata.get("session_id")
        or uuid4()
    )


@observe(name="support_orchestration_graph", as_type="chain", capture_input=False, capture_output=False)
async def run_support_graph(query: str, metadata: dict[str, Any] | None = None) -> SupportOrchestrationState:
    """Run the support orchestration graph for a single query."""

    started = time.perf_counter()
    run_metadata = dict(metadata or {})
    thread_id = _thread_id(run_metadata)
    run_metadata.setdefault("thread_id", thread_id)
    run_metadata.setdefault("conversation_id", thread_id)

    state = _initialize_state(query=query, metadata=run_metadata)
    graph_config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": CHECKPOINT_NAMESPACE,
        }
    }
    try:
        graph = await get_support_graph()
        result = await graph.ainvoke(state, config=graph_config)
    except Exception as exc:
        if not _is_checkpointer_connection_error(exc):
            raise
        logger.warning(
            "LangGraph checkpointer connection failed; rebuilding checkpointer and retrying once: %s",
            exc,
        )
        await reset_support_graph_checkpointer()
        graph = await get_support_graph()
        result = await graph.ainvoke(state, config=graph_config)
    final_state: SupportOrchestrationState = dict(result)
    final_state["latency_ms"] = int((time.perf_counter() - started) * 1000)
    langfuse_client = get_langfuse_client()
    if langfuse_client is not None:
        try:
            final_state["langfuse_trace_id"] = langfuse_client.get_current_trace_id()
        except Exception:
            logger.exception("failed to capture current Langfuse trace id")
    trace_agent_state(
        agent_name="support_orchestration_graph",
        input_state={"query": query, "metadata": run_metadata, "conversation_id": thread_id},
        output_state=final_state,
        started_at=started,
        tool_used="langgraph",
    )
    return final_state


async def stream_support_graph(
    query: str,
    metadata: dict[str, Any] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Stream LangGraph node updates and the final orchestration state."""

    started = time.perf_counter()
    run_metadata = dict(metadata or {})
    thread_id = _thread_id(run_metadata)
    run_metadata.setdefault("thread_id", thread_id)
    run_metadata.setdefault("conversation_id", thread_id)

    state = _initialize_state(query=query, metadata=run_metadata)
    graph_config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": CHECKPOINT_NAMESPACE,
        }
    }

    langfuse_client = get_langfuse_client()
    root_trace_id: str | None = None
    root_observation_context = nullcontext()
    if langfuse_client is not None:
        try:
            root_trace_id = langfuse_client.create_trace_id()
            root_observation_context = langfuse_client.start_as_current_observation(
                name="support_orchestration_graph",
                as_type="chain",
                trace_context={"trace_id": root_trace_id},
                input={"query": query, "metadata": run_metadata, "conversation_id": thread_id},
                metadata={
                    "transport": "stream",
                    "thread_id": thread_id,
                    "conversation_id": thread_id,
                },
            )
        except Exception:
            logger.exception("failed to initialize streaming Langfuse root trace")

    with root_observation_context:
        graph = await get_support_graph()
        final_state: SupportOrchestrationState = dict(state)
        try:
            async for update in graph.astream(state, config=graph_config, stream_mode="updates"):
                if not isinstance(update, dict):
                    continue
                for node_name, node_state in update.items():
                    if not isinstance(node_state, dict):
                        continue
                    final_state.update(node_state)
                    yield {
                        "event": "node",
                        "node": str(node_name),
                        "state": dict(final_state),
                    }
        except Exception as exc:
            if not _is_checkpointer_connection_error(exc):
                raise
            logger.warning(
                "LangGraph streaming checkpointer connection failed; rebuilding checkpointer and retrying once: %s",
                exc,
            )
            await reset_support_graph_checkpointer()
            graph = await get_support_graph()
            final_state = dict(state)
            async for update in graph.astream(state, config=graph_config, stream_mode="updates"):
                if not isinstance(update, dict):
                    continue
                for node_name, node_state in update.items():
                    if not isinstance(node_state, dict):
                        continue
                    final_state.update(node_state)
                    yield {
                        "event": "node",
                        "node": str(node_name),
                        "state": dict(final_state),
                    }

        final_state["latency_ms"] = int((time.perf_counter() - started) * 1000)
        if root_trace_id:
            final_state["langfuse_trace_id"] = root_trace_id
        elif langfuse_client is not None:
            try:
                final_state["langfuse_trace_id"] = langfuse_client.get_current_trace_id()
            except Exception:
                logger.exception("failed to capture current Langfuse trace id")
        trace_agent_state(
            agent_name="support_orchestration_graph",
            input_state={"query": query, "metadata": run_metadata, "conversation_id": thread_id},
            output_state=final_state,
            started_at=started,
            tool_used="langgraph_stream",
        )
        yield {
            "event": "final",
            "node": END,
            "state": final_state,
        }
