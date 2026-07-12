import logging
import time
from datetime import datetime, timezone
from typing import Any

from app.core.langfuse import observe, trace_agent_state, trace_guardrail_event
from app.orchestration.state import (
    AgentTraceEvent,
    EscalationTarget,
    ExecutionResult,
    GuardrailFlag,
    ProgressUpdate,
    RetrievedChunk,
    RouteDecision,
    SeverityLevel,
    SupportOrchestrationState,
    VerificationOutcome,
)

logger = logging.getLogger(__name__)

AGENT_NAME = "response_validation_agent"
CONFIDENCE_THRESHOLD = 0.70
STRONG_ESCALATION_TARGETS = {
    EscalationTarget.SECURITY_TEAM,
    EscalationTarget.INCIDENT_RESPONSE,
    EscalationTarget.ENGINEERING,
    EscalationTarget.L3_SUPPORT,
}


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


def _existing_target(state: SupportOrchestrationState) -> EscalationTarget:
    target = state.get("escalation_target")
    if isinstance(target, EscalationTarget):
        return target
    try:
        return EscalationTarget(str(target))
    except (TypeError, ValueError):
        return EscalationTarget.NONE


def _append_escalation_reason(state: SupportOrchestrationState, reason: str) -> None:
    existing_reason = state.get("escalation_reason")
    state["escalation_reason"] = f"{existing_reason}; {reason}" if existing_reason else reason


def _route_value(state: SupportOrchestrationState) -> str:
    route = state.get("route_decision")
    if isinstance(route, RouteDecision):
        return route.value
    return str(route or "")


def _requires_document_evidence(state: SupportOrchestrationState) -> bool:
    return _route_value(state) in {RouteDecision.RAG.value, RouteDecision.HYBRID.value}


def _has_non_document_evidence(state: SupportOrchestrationState) -> bool:
    if state.get("sql_results") or state.get("customer_context") or state.get("incident_investigation"):
        return True
    return _route_value(state) in {RouteDecision.HIGH_RISK.value, RouteDecision.SQL.value}


def _derive_citations(chunks: list[RetrievedChunk]) -> list[str]:
    citations = sorted({str(chunk.get("source_file")) for chunk in chunks if chunk.get("source_file")})
    return citations


def _guardrail(name: str, passed: bool, reason: str, severity: SeverityLevel, metadata: dict[str, Any]) -> GuardrailFlag:
    return {
        "name": name,
        "passed": passed,
        "severity": severity,
        "reason": reason,
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
    evidence_present: bool,
    confidence_passed: bool,
    citations_present: bool,
    confidence_score: float,
    document_evidence_required: bool,
) -> ExecutionResult:
    return {
        "step_id": "response-validation",
        "agent_name": AGENT_NAME,
        "result_type": "response_validation",
        "summary": (
            f"evidence_present={evidence_present}; confidence={confidence_score:.2f}; "
            f"citations_present={citations_present}"
        ),
        "data": {
            "retrieved_chunks_present": evidence_present,
            "document_evidence_required": document_evidence_required,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "confidence_score": confidence_score,
            "confidence_passed": confidence_passed,
            "citations_present": citations_present,
        },
        "error": None,
        "timestamp": _utc_now(),
    }


def _verification(
    evidence_present: bool,
    confidence_passed: bool,
    citations_present: bool,
    confidence_score: float,
    document_evidence_required: bool,
) -> VerificationOutcome:
    passed = evidence_present and confidence_passed and citations_present
    failures = []
    if not evidence_present:
        failures.append("required evidence is missing")
    if not confidence_passed:
        failures.append(f"confidence below {CONFIDENCE_THRESHOLD:.2f}")
    if not citations_present:
        failures.append("citations are missing")

    return {
        "check_name": "response_validation_complete",
        "passed": passed,
        "score": confidence_score,
        "reason": "Response validation passed." if passed else "Response validation failed: " + ", ".join(failures),
        "corrective_action": None if passed else "Escalate, retrieve more evidence, or ask for clarification.",
        "metadata": {
            "retrieved_chunks_present": evidence_present,
            "document_evidence_required": document_evidence_required,
            "confidence_passed": confidence_passed,
            "citations_present": citations_present,
        },
    }


def _set_low_confidence_escalation(state: SupportOrchestrationState, reason: str) -> None:
    state["escalation_flag"] = True
    existing_target = _existing_target(state)
    if existing_target not in STRONG_ESCALATION_TARGETS:
        state["escalation_target"] = EscalationTarget.L2_SUPPORT
    else:
        state["escalation_target"] = existing_target
    _append_escalation_reason(state, reason)


@observe(name=AGENT_NAME, as_type="agent", capture_input=False, capture_output=False)
def validate_response(state: SupportOrchestrationState) -> SupportOrchestrationState:
    """Validate evidence, confidence, and citations before response delivery."""

    started = time.perf_counter()
    next_state: SupportOrchestrationState = dict(state)

    _append_list(next_state, "progress_updates", _progress("response-validation", "started", "Response validation started."))

    chunks = list(next_state.get("retrieved_chunks") or [])
    chunks_present = bool(chunks)
    document_evidence_required = _requires_document_evidence(next_state)
    evidence_present = chunks_present if document_evidence_required else _has_non_document_evidence(next_state)

    confidence_score = _clamp_score(next_state.get("confidence_score"))
    confidence_passed = confidence_score >= CONFIDENCE_THRESHOLD

    citations = [str(citation) for citation in next_state.get("citations", []) if citation]
    if not citations and chunks:
        citations = _derive_citations(chunks)
        if citations:
            next_state["citations"] = citations
    citations_present = bool(citations) if document_evidence_required else True

    if not evidence_present:
        _append_list(next_state, "errors", "Response validation failed: required evidence is missing.")
    if not confidence_passed:
        reason = f"MG-2 triggered: confidence_score {confidence_score:.2f} is below {CONFIDENCE_THRESHOLD:.2f}."
        _append_list(next_state, "errors", reason)
        _set_low_confidence_escalation(next_state, reason)
    if document_evidence_required and not citations_present:
        _append_list(next_state, "errors", "OG-1 failed: response citations are missing.")

    _append_list(
        next_state,
        "guardrail_flags",
        _guardrail(
            name="retrieved_chunks_present",
            passed=evidence_present,
            reason=(
                "Document evidence is present."
                if chunks_present
                else "Structured/non-document route has sufficient evidence."
                if evidence_present
                else "Required evidence is missing."
            ),
            severity=SeverityLevel.LOW if evidence_present else SeverityLevel.HIGH,
            metadata={
                "chunk_count": len(chunks),
                "document_evidence_required": document_evidence_required,
                "route_decision": _route_value(next_state),
            },
        ),
    )
    _append_list(
        next_state,
        "guardrail_flags",
        _guardrail(
            name="confidence_threshold_check",
            passed=confidence_passed,
            reason=f"Confidence score is {confidence_score:.2f}; threshold is {CONFIDENCE_THRESHOLD:.2f}.",
            severity=SeverityLevel.LOW if confidence_passed else SeverityLevel.HIGH,
            metadata={"confidence_score": confidence_score, "threshold": CONFIDENCE_THRESHOLD},
        ),
    )
    _append_list(
        next_state,
        "guardrail_flags",
        _guardrail(
            name="citations_present_og_1",
            passed=citations_present,
            reason=(
                "Citations are present."
                if citations
                else "Citations are not required for this non-document route."
                if citations_present
                else "Citations are missing for document-backed response."
            ),
            severity=SeverityLevel.LOW if citations_present else SeverityLevel.HIGH,
            metadata={
                "citation_count": len(citations),
                "document_evidence_required": document_evidence_required,
                "route_decision": _route_value(next_state),
            },
        ),
    )
    for guardrail in list(next_state.get("guardrail_flags", []))[-3:]:
        trace_guardrail_event(
            name=str(guardrail.get("name")),
            passed=bool(guardrail.get("passed")),
            reason=str(guardrail.get("reason")),
            metadata=dict(guardrail.get("metadata") or {}),
        )

    latency_ms = int((time.perf_counter() - started) * 1000)
    status = "completed" if evidence_present and confidence_passed and citations_present else "failed"
    output_summary = (
        f"evidence_present={evidence_present}; document_required={document_evidence_required}; "
        f"confidence={confidence_score:.2f}; citations_present={citations_present}; status={status}"
    )

    _append_list(
        next_state,
        "execution_results",
        _execution_result(
            evidence_present,
            confidence_passed,
            citations_present,
            confidence_score,
            document_evidence_required,
        ),
    )
    _append_list(
        next_state,
        "verification_outcomes",
        _verification(
            evidence_present,
            confidence_passed,
            citations_present,
            confidence_score,
            document_evidence_required,
        ),
    )
    _append_list(
        next_state,
        "agent_trace",
        _trace(
            action="validate_response",
            status=status,
            input_summary=f"chunks={len(chunks)}; citations={len(citations)}",
            output_summary=output_summary,
            latency_ms=latency_ms,
        ),
    )
    _append_list(next_state, "progress_updates", _progress("response-validation", "completed", output_summary))

    trace_agent_state(
        agent_name=AGENT_NAME,
        input_state=state,
        output_state=next_state,
        started_at=started,
        tool_used="response_guardrails",
    )

    return next_state
