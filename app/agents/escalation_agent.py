import time
from datetime import datetime, timezone
from typing import Any

from app.core.langfuse import observe, trace_agent_state
from app.services.tools.email_mcp_tool import send_escalation_email
from app.services.tools.jira_mcp_tool import track_escalation_in_jira
from app.orchestration.state import (
    AgentTraceEvent,
    EscalationTarget,
    ExecutionResult,
    ProgressUpdate,
    SeverityLevel,
    SeverityPriority,
    SupportIntent,
    SupportOrchestrationState,
    VerificationOutcome,
)

AGENT_NAME = "escalation_manager_agent"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_list(state: SupportOrchestrationState, key: str, item: Any) -> None:
    values = list(state.get(key, []))
    values.append(item)
    state[key] = values  # type: ignore[literal-required]


def _enum_value(value: Any) -> str | None:
    if value is None:
        return None
    return getattr(value, "value", str(value))


def _target_from_state(state: SupportOrchestrationState) -> EscalationTarget:
    existing = state.get("escalation_target")
    if isinstance(existing, EscalationTarget) and existing != EscalationTarget.NONE:
        return existing
    try:
        parsed = EscalationTarget(str(existing))
        if parsed != EscalationTarget.NONE:
            return parsed
    except (TypeError, ValueError):
        pass

    if state.get("intent") == SupportIntent.SECURITY:
        return EscalationTarget.SECURITY_TEAM
    if state.get("severity_priority") == SeverityPriority.P0 or state.get("severity") == SeverityLevel.CRITICAL:
        return EscalationTarget.INCIDENT_RESPONSE

    incident_result = dict(state.get("incident_investigation") or {})
    if incident_result.get("active_critical_incident_correlated"):
        return EscalationTarget.INCIDENT_RESPONSE

    return EscalationTarget.L2_SUPPORT


def _reason_from_state(state: SupportOrchestrationState, target: EscalationTarget) -> str:
    if state.get("escalation_reason"):
        return str(state["escalation_reason"])
    if state.get("severity_reason"):
        return str(state["severity_reason"])
    if state.get("errors"):
        return "; ".join(str(error) for error in state.get("errors", [])[-3:])
    return f"Escalation required for {target.value} review."


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


def _execution_result(target: EscalationTarget, reason: str, package: dict[str, Any]) -> ExecutionResult:
    return {
        "step_id": "escalation-management",
        "agent_name": AGENT_NAME,
        "result_type": "escalation_management",
        "summary": f"escalation_target={target.value}; reason={reason[:120]}",
        "data": package,
        "error": None,
        "timestamp": _utc_now(),
    }


def _verification(target: EscalationTarget, reason: str) -> VerificationOutcome:
    passed = target != EscalationTarget.NONE and bool(reason.strip())
    return {
        "check_name": "escalation_package_complete",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": "Escalation package was finalized." if passed else "Escalation target or reason is missing.",
        "corrective_action": None if passed else "Assign an escalation owner and reason.",
        "metadata": {"escalation_target": target.value},
    }


def _build_package(state: SupportOrchestrationState, target: EscalationTarget, reason: str) -> dict[str, Any]:
    return {
        "target": target.value,
        "reason": reason,
        "query": state.get("query"),
        "intent": _enum_value(state.get("intent")),
        "route_decision": _enum_value(state.get("route_decision")),
        "severity_priority": _enum_value(state.get("severity_priority")),
        "severity": _enum_value(state.get("severity")),
        "confidence_score": state.get("confidence_score"),
        "citations": list(state.get("citations") or []),
        "retrieved_chunk_count": len(state.get("retrieved_chunks") or []),
        "sql_result_count": len(state.get("sql_results") or []),
        "customer_context": dict(state.get("customer_context") or {}),
        "incident_investigation": dict(state.get("incident_investigation") or {}),
        "retrieved_chunks": list(state.get("retrieved_chunks") or [])[:10],
        "sql_results": list(state.get("sql_results") or [])[:5],
        "guardrail_flags": list(state.get("guardrail_flags") or []),
        "agent_trace": list(state.get("agent_trace") or []),
        "errors": list(state.get("errors") or []),
    }


def _display_label(value: Any) -> str:
    text = _enum_value(value) or "unknown"
    return text.replace("_", " ").replace("-", " ").title()


def _risk_control_summary(reason: str) -> str:
    lowered = reason.lower()
    if "data loss" in lowered:
        return (
            "ERIS did not mark the incident as resolved. Data-loss cases require a human incident owner "
            "to confirm recovery, data integrity, customer impact, and closure evidence before status changes."
        )
    if any(keyword in lowered for keyword in ["breach", "vulnerability", "unauthorized", "credential", "security"]):
        return (
            "ERIS did not perform an automated remediation or closure action. Security-sensitive cases require "
            "validated containment, remediation evidence, and owner approval before resolution."
        )
    if any(keyword in lowered for keyword in ["outage", "production", "service down", "critical incident"]):
        return (
            "ERIS did not close the incident automatically. Production-impacting incidents require human "
            "validation that service health, customer impact, and mitigation evidence are complete."
        )
    return (
        "ERIS paused automated resolution and prepared a human handoff package because this request crossed "
        "the configured escalation policy."
    )


def _immediate_actions(reason: str, target: EscalationTarget) -> list[str]:
    lowered = reason.lower()
    if "data loss" in lowered:
        return [
            "Incident Response validates affected scope, recovery status, and data-integrity checks.",
            "Confirm backups, reconciliation, or compensating recovery steps are complete and documented.",
            "Only the incident owner should move the incident to Resolved after closure evidence is attached.",
            "Record customer/stakeholder communication requirements in the Jira issue.",
        ]
    if any(keyword in lowered for keyword in ["breach", "vulnerability", "unauthorized", "credential", "security"]):
        return [
            "Security owner validates containment, affected assets, and exposure window.",
            "Confirm remediation, patching, credential rotation, and evidence collection are complete.",
            "Keep the case open until security sign-off and customer communication requirements are clear.",
        ]
    if target == EscalationTarget.INCIDENT_RESPONSE:
        return [
            "Incident Response reviews active incident state, ownership, impact, and mitigation status.",
            "Confirm service health and customer impact before any closure or resolution update.",
            "Attach operational evidence and next-owner notes to the Jira issue.",
        ]
    return [
        f"{_display_label(target)} reviews the evidence package and confirms the next owner.",
        "Do not apply irreversible changes until the assigned owner approves the action.",
        "Keep the Jira issue updated with decision, evidence, and follow-up tasks.",
    ]


def _final_answer(
    target: EscalationTarget,
    reason: str,
    jira_result: dict[str, Any] | None = None,
    email_result: dict[str, Any] | None = None,
) -> str:
    lines = [
        "## Escalation required",
        "",
        _risk_control_summary(reason),
        "",
        "### Handoff summary",
        f"- Target team: {_display_label(target)}",
        f"- Severity policy: Human approval required before resolution or closure.",
        f"- Escalation reason: {reason}",
        "",
        "### Immediate actions",
    ]
    lines.extend(f"- {action}" for action in _immediate_actions(reason, target))

    if jira_result and jira_result.get("issue_key"):
        issue_text = str(jira_result.get("issue_key"))
        if jira_result.get("issue_url"):
            issue_text = f"[{issue_text}]({jira_result.get('issue_url')})"
        lines.extend(["", "### Tracking", f"- Jira engineering handoff: {issue_text}"])
    elif jira_result and jira_result.get("status") == "failed":
        lines.extend(
            [
                "",
                "### Tracking",
                "- Jira tracking could not be created automatically. The escalation package remains available in the orchestration trace.",
            ]
        )
    if email_result and email_result.get("sent"):
        if "### Tracking" not in lines:
            lines.append("")
            lines.append("### Tracking")
        recipients = ", ".join(str(item) for item in email_result.get("recipients", []))
        lines.append(f"- Support notification email sent to: {recipients}")
    elif email_result and email_result.get("enabled") and email_result.get("error"):
        if "### Tracking" not in lines:
            lines.append("")
            lines.append("### Tracking")
        lines.append("- Support notification email could not be sent automatically; use the Jira issue for handoff.")
    return "\n".join(lines)


@observe(name=AGENT_NAME, as_type="agent", capture_input=False, capture_output=False)
async def manage_escalation(state: SupportOrchestrationState) -> SupportOrchestrationState:
    """Finalize the escalation decision and handoff package."""

    started = time.perf_counter()
    next_state: SupportOrchestrationState = dict(state)

    _append_list(next_state, "progress_updates", _progress("escalation-management", "started", "Escalation management started."))

    target = _target_from_state(next_state)
    reason = _reason_from_state(next_state, target)
    package = _build_package(next_state, target, reason)
    metadata = dict(next_state.get("metadata") or {})

    next_state["escalation_flag"] = True
    next_state["escalation_target"] = target
    next_state["escalation_reason"] = reason

    jira_result = await track_escalation_in_jira(next_state, package)
    package["jira_tracking_result"] = jira_result
    email_result = await send_escalation_email(next_state, package, jira_result)
    package["email_notification_result"] = email_result
    metadata["escalation_package"] = package
    metadata["jira_tracking_result"] = jira_result
    metadata["email_notification_result"] = email_result

    next_state["metadata"] = metadata
    next_state["jira_tracking_result"] = jira_result
    next_state["email_notification_result"] = email_result  # type: ignore[typeddict-unknown-key]
    next_state["jira_issue_key"] = jira_result.get("issue_key")
    next_state["jira_issue_url"] = jira_result.get("issue_url")
    next_state["final_answer"] = _final_answer(target, reason, jira_result, email_result)
    next_state["recommended_actions"] = _immediate_actions(reason, target)
    if jira_result.get("issue_key"):
        next_state["recommended_actions"].append(f"Track engineering work in Jira issue {jira_result['issue_key']}.")
    elif jira_result.get("status") == "failed":
        _append_list(next_state, "errors", f"jira_tracking_failed: {jira_result.get('error')}")
    if email_result.get("sent"):
        next_state["recommended_actions"].append("Support team notification email was sent with the Jira handoff details.")
    elif email_result.get("enabled") and email_result.get("error"):
        _append_list(next_state, "errors", f"email_notification_failed: {email_result.get('error')}")

    latency_ms = int((time.perf_counter() - started) * 1000)
    output_summary = (
        f"target={target.value}; package_keys={len(package)}; "
        f"jira_action={jira_result.get('action')}; jira_issue={jira_result.get('issue_key')}; "
        f"email_status={email_result.get('status')}"
    )

    _append_list(next_state, "execution_results", _execution_result(target, reason, package))
    _append_list(next_state, "verification_outcomes", _verification(target, reason))
    _append_list(
        next_state,
        "agent_trace",
        _trace(
            action="manage_escalation",
            status="completed",
            input_summary=f"severity={_enum_value(next_state.get('severity'))}; errors={len(next_state.get('errors', []))}",
            output_summary=output_summary,
            latency_ms=latency_ms,
        ),
    )
    _append_list(next_state, "progress_updates", _progress("escalation-management", "completed", output_summary))

    trace_agent_state(
        agent_name=AGENT_NAME,
        input_state=state,
        output_state=next_state,
        started_at=started,
        tool_used="escalation_packager",
    )

    return next_state
