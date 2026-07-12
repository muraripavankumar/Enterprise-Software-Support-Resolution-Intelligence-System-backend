import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.azure_openai import AzureOpenAI

from app.agents.doc_retrieval_agent import retrieve_documents
from app.agents.incident_investigator_agent import investigate_incidents
from app.agents.sql_agent import query_structured_data
from app.core.config import settings
from app.orchestration.state import (
    AgentTraceEvent,
    ExecutionResult,
    HybridResult,
    ProgressUpdate,
    RetrievedChunk,
    SQLResult,
    SupportOrchestrationState,
    VerificationOutcome,
)
from app.services.tools.tool_registry import get_tools

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are the retrieval orchestrator for the Enterprise Software Support & Resolution Intelligence System.

Choose tools carefully:
- Use documentation_policy_retrieval for product docs, API guides, troubleshooting, SLA policy, security policy, ITIL, installation, performance, and how-to questions.
- Use structured_support_data_query for customer records, tickets, incidents, subscriptions, SLA tiers, escalation flags, and knowledge article usage.
- Use both tools for hybrid questions requiring documentation plus operational validation.

The documentation_policy_retrieval tool returns evidence chunks and citations only. Read that evidence and generate the final user-facing answer yourself.
Return factual, cited, support-ready answers. Cite source file names or page metadata when available. If tool output is incomplete, say what is missing and recommend escalation when appropriate.
""".strip()

_AGENT: FunctionAgent | None = None
HYBRID_AGENT_NAME = "hybrid_parallel_tool_executor"
MERGED_LIST_KEYS = {
    "progress_updates",
    "execution_results",
    "verification_outcomes",
    "guardrail_flags",
    "agent_trace",
    "errors",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_list(state: SupportOrchestrationState, key: str, item: Any) -> None:
    values = list(state.get(key, []))
    values.append(item)
    state[key] = values  # type: ignore[literal-required]


def _new_items(
    original_state: SupportOrchestrationState,
    updated_state: SupportOrchestrationState,
    key: str,
) -> list[Any]:
    original_count = len(list(original_state.get(key, [])))
    return list(updated_state.get(key, []))[original_count:]


def _progress(step_id: str, status: str, message: str) -> ProgressUpdate:
    return {
        "step_id": step_id,
        "agent_name": HYBRID_AGENT_NAME,
        "status": status,
        "message": message,
        "timestamp": _utc_now(),
    }


def _hybrid_conflicts(retrieved_chunks: list[RetrievedChunk], sql_results: list[SQLResult]) -> list[str]:
    conflicts: list[str] = []
    if not retrieved_chunks:
        conflicts.append("No document evidence was retrieved for the hybrid query.")
    if not sql_results:
        conflicts.append("No structured SQL evidence was retrieved for the hybrid query.")
    for result in sql_results:
        if result.get("error"):
            conflicts.append(f"SQL evidence returned an error: {result['error']}")
    return conflicts


def _hybrid_result(
    retrieved_chunks: list[RetrievedChunk],
    sql_results: list[SQLResult],
    conflicts: list[str],
) -> HybridResult:
    sql_evidence_count = sum(1 for result in sql_results if not result.get("error"))
    recommended_action = (
        "Proceed with answer synthesis using document and structured evidence."
        if not conflicts
        else "Use available evidence cautiously and escalate if missing evidence affects resolution confidence."
    )
    return {
        "combined_summary": (
            f"Hybrid retrieval collected {len(retrieved_chunks)} document chunks and "
            f"{sql_evidence_count} structured result set(s)."
        ),
        "rag_evidence_count": len(retrieved_chunks),
        "sql_evidence_count": sql_evidence_count,
        "conflicts": conflicts,
        "recommended_action": recommended_action,
    }


def _parallel_trace(
    retrieved_chunks: list[RetrievedChunk],
    sql_results: list[SQLResult],
    conflicts: list[str],
    latency_ms: int,
) -> AgentTraceEvent:
    return {
        "agent_name": HYBRID_AGENT_NAME,
        "action": "execute_vector_and_sql_with_asyncio_gather",
        "input_summary": "hybrid route selected",
        "output_summary": (
            f"chunks={len(retrieved_chunks)}; sql_results={len(sql_results)}; conflicts={len(conflicts)}"
        ),
        "status": "completed" if not conflicts else "completed_with_warnings",
        "timestamp": _utc_now(),
        "latency_ms": latency_ms,
    }


def _parallel_execution_result(
    retrieved_chunks: list[RetrievedChunk],
    sql_results: list[SQLResult],
    conflicts: list[str],
    latency_ms: int,
) -> ExecutionResult:
    return {
        "step_id": "hybrid-parallel-tools",
        "agent_name": HYBRID_AGENT_NAME,
        "result_type": "hybrid_parallel_retrieval",
        "summary": (
            f"parallel vector+sql completed; chunks={len(retrieved_chunks)}; "
            f"sql_results={len(sql_results)}; latency_ms={latency_ms}"
        ),
        "data": {
            "rag_evidence_count": len(retrieved_chunks),
            "sql_evidence_count": len(sql_results),
            "conflicts": conflicts,
            "parallel_execution": True,
        },
        "error": "; ".join(conflicts) if conflicts else None,
        "timestamp": _utc_now(),
    }


def _parallel_verification(
    retrieved_chunks: list[RetrievedChunk],
    sql_results: list[SQLResult],
    conflicts: list[str],
) -> VerificationOutcome:
    passed = bool(retrieved_chunks) and bool(sql_results) and not conflicts
    return {
        "check_name": "hybrid_parallel_tools_complete",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": "Vector and SQL evidence completed concurrently and returned usable evidence."
        if passed
        else "Hybrid parallel execution completed with missing or conflicting evidence.",
        "corrective_action": None if passed else "Validate confidence and escalate if response quality remains below threshold.",
        "metadata": {
            "rag_evidence_count": len(retrieved_chunks),
            "sql_evidence_count": len(sql_results),
            "conflicts": conflicts,
        },
    }


def _merge_parallel_state(
    original_state: SupportOrchestrationState,
    vector_state: SupportOrchestrationState,
    sql_state: SupportOrchestrationState,
    latency_ms: int,
    incident_state: SupportOrchestrationState | None = None,
) -> SupportOrchestrationState:
    merged_state: SupportOrchestrationState = dict(original_state)

    for key in MERGED_LIST_KEYS:
        merged_values = list(original_state.get(key, []))
        merged_values.extend(_new_items(original_state, vector_state, key))
        merged_values.extend(_new_items(original_state, sql_state, key))
        if incident_state is not None:
            merged_values.extend(_new_items(original_state, incident_state, key))
        merged_state[key] = merged_values  # type: ignore[literal-required]

    retrieved_chunks = list(vector_state.get("retrieved_chunks") or [])
    sql_results = list(sql_state.get("sql_results") or [])
    citations = sorted(
        {
            str(citation)
            for citation in list(vector_state.get("citations") or [])
            if str(citation).strip()
        }
    )

    merged_state["retrieved_chunks"] = retrieved_chunks
    merged_state["sql_results"] = sql_results
    merged_state["citations"] = citations
    if incident_state is not None:
        merged_state["incident_investigation"] = dict(incident_state.get("incident_investigation") or {})

    confidence_candidates = [
        value
        for value in [
            original_state.get("confidence_score"),
            vector_state.get("confidence_score"),
            sql_state.get("confidence_score"),
        ]
        if value is not None
    ]
    if confidence_candidates:
        merged_state["confidence_score"] = max(float(value) for value in confidence_candidates)

    conflicts = _hybrid_conflicts(retrieved_chunks, sql_results)
    merged_state["hybrid_result"] = _hybrid_result(retrieved_chunks, sql_results, conflicts)

    _append_list(merged_state, "execution_results", _parallel_execution_result(retrieved_chunks, sql_results, conflicts, latency_ms))
    _append_list(merged_state, "verification_outcomes", _parallel_verification(retrieved_chunks, sql_results, conflicts))
    _append_list(merged_state, "agent_trace", _parallel_trace(retrieved_chunks, sql_results, conflicts, latency_ms))
    _append_list(
        merged_state,
        "progress_updates",
        _progress(
            "hybrid-parallel-tools",
            "completed",
            (
            f"Hybrid evidence tools completed concurrently; "
            f"chunks={len(retrieved_chunks)}; sql_results={len(sql_results)}; conflicts={len(conflicts)}"
            ),
        ),
    )
    return merged_state


def _build_llm() -> AzureOpenAI:
    return AzureOpenAI(
        model=settings.azure_openai_chat_deployment,
        deployment_name=settings.azure_openai_chat_deployment,
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        temperature=0.1,
    )


def get_agent() -> FunctionAgent:
    global _AGENT
    if _AGENT is not None:
        return _AGENT
    _AGENT = FunctionAgent(
        name="enterprise_support_retrieval_agent",
        description="Routes support questions to vector retrieval, SQL retrieval, or both.",
        tools=get_tools(),
        llm=_build_llm(),
        system_prompt=SYSTEM_PROMPT,
        verbose=True,
        timeout=120,
    )
    return _AGENT


async def ask_retrieval_agent(question: str) -> Dict[str, Any]:
    try:
        response = await get_agent().run(user_msg=question)
        return {"answer": str(response), "error": None}
    except Exception as exc:
        logger.exception("Retrieval agent failed")
        return {"answer": "Retrieval agent failed.", "error": str(exc)}


async def execute_hybrid_tools_parallel(state: SupportOrchestrationState) -> SupportOrchestrationState:
    """Run document retrieval and SQL retrieval concurrently for hybrid support queries."""

    started = time.perf_counter()
    original_state: SupportOrchestrationState = dict(state)
    _append_list(
        original_state,
        "progress_updates",
        _progress("hybrid-parallel-tools", "started", "Running vector and SQL tools concurrently."),
    )

    vector_input: SupportOrchestrationState = dict(original_state)
    sql_input: SupportOrchestrationState = dict(original_state)

    try:
        vector_state, sql_state = await asyncio.gather(
            retrieve_documents(vector_input),
            query_structured_data(sql_input),
        )
    except Exception as exc:
        logger.exception("Hybrid parallel tool execution failed")
        failed_state: SupportOrchestrationState = dict(original_state)
        _append_list(failed_state, "errors", f"Hybrid parallel tool execution failed: {exc}")
        _append_list(
            failed_state,
            "verification_outcomes",
            {
                "check_name": "hybrid_parallel_tools_complete",
                "passed": False,
                "score": 0.0,
                "reason": str(exc),
                "corrective_action": "Retry route or escalate if evidence cannot be gathered.",
                "metadata": {"parallel_execution": True},
            },
        )
        return failed_state

    latency_ms = int((time.perf_counter() - started) * 1000)
    return _merge_parallel_state(original_state, vector_state, sql_state, latency_ms)


async def execute_high_risk_evidence_parallel(state: SupportOrchestrationState) -> SupportOrchestrationState:
    """Run document, SQL, and incident evidence collection concurrently for high-risk routes."""

    started = time.perf_counter()
    original_state: SupportOrchestrationState = dict(state)
    _append_list(
        original_state,
        "progress_updates",
        _progress("high-risk-parallel-tools", "started", "Running document, SQL, and incident searches concurrently."),
    )

    vector_input: SupportOrchestrationState = dict(original_state)
    sql_input: SupportOrchestrationState = dict(original_state)
    incident_input: SupportOrchestrationState = dict(original_state)

    try:
        vector_state, sql_state, incident_state = await asyncio.gather(
            retrieve_documents(vector_input),
            query_structured_data(sql_input),
            investigate_incidents(incident_input),
        )
    except Exception as exc:
        logger.exception("High-risk parallel evidence execution failed")
        failed_state: SupportOrchestrationState = dict(original_state)
        _append_list(failed_state, "errors", f"High-risk parallel evidence execution failed: {exc}")
        _append_list(
            failed_state,
            "verification_outcomes",
            {
                "check_name": "high_risk_parallel_tools_complete",
                "passed": False,
                "score": 0.0,
                "reason": str(exc),
                "corrective_action": "Escalate if high-risk evidence cannot be gathered.",
                "metadata": {"parallel_execution": True},
            },
        )
        return failed_state

    latency_ms = int((time.perf_counter() - started) * 1000)
    merged_state = _merge_parallel_state(original_state, vector_state, sql_state, latency_ms, incident_state)
    _append_list(
        merged_state,
        "verification_outcomes",
        {
            "check_name": "high_risk_parallel_tools_complete",
            "passed": True,
            "score": 1.0,
            "reason": "Document, SQL, and incident evidence completed concurrently.",
            "corrective_action": None,
            "metadata": {"parallel_execution": True, "latency_ms": latency_ms},
        },
    )
    return merged_state
