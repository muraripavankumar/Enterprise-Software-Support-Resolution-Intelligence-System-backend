import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.core.langfuse import observe, trace_agent_state
from app.orchestration.state import (
    AgentTraceEvent,
    EscalationTarget,
    ExecutionResult,
    ProgressUpdate,
    RouteDecision,
    SeverityLevel,
    SeverityPriority,
    SupportIntent,
    SupportOrchestrationState,
    VerificationOutcome,
)

logger = logging.getLogger(__name__)

AGENT_NAME = "severity_assessment_agent"

P0_KEYWORDS = {
    "outage",
    "production outage",
    "data loss",
    "security vulnerability",
    "vulnerability",
    "production down",
    "service down",
    "system down",
    "unresolved critical alert",
    "unresolved critical",
    "critical alert",
    "premium customer outage",
}

SECURITY_ESCALATION_KEYWORDS = {
    "data breach",
    "security breach",
    "credential breach",
    "credentials breached",
    "security vulnerability",
    "vulnerability",
    "unauthorized access",
    "api key exposed",
    "customer data exposed",
    "token leak",
    "secret leak",
    "exposure",
}

P1_KEYWORDS = {
    "high severity",
    "severity high",
    "p1",
    "sev1",
    "urgent",
    "high impact",
    "degraded production",
    "production degradation",
    "degraded service",
    "repeated failures",
    "active incident",
    "severe latency",
    "major latency",
    "payment failure",
    "account suspension",
    "suspended account",
    "suspended accounts",
    "suspended during incident",
    "multiple users",
    "critical severity ticket",
    "critical severity tickets",
    "critical ticket",
    "critical tickets",
    "open incident",
    "open incidents",
    "active incident",
    "active incidents",
}

P2_KEYWORDS = {
    "401",
    "403",
    "authentication",
    "authorization",
    "api",
    "configuration",
    "config",
    "single customer",
    "one customer",
    "integration",
    "failed request",
}

FAILURE_KEYWORDS = {
    "error",
    "errors",
    "failed",
    "failure",
    "fails",
    "cannot",
    "can't",
    "unable",
    "broken",
    "not working",
    "issue",
    "problem",
}

P3_KEYWORDS = {
    "how",
    "what",
    "configure",
    "setup",
    "install",
    "documentation",
    "docs",
    "guide",
    "use",
    "usage",
    "general",
    "process",
    "procedure",
    "workflow",
    "lifecycle",
    "management",
    "ownership",
    "owner",
    "responsibility",
    "responsibilities",
    "role",
    "roles",
    "handoff",
    "support model",
    "ownership model",
    "recommended",
    "best practice",
    "explain",
}

PREMIUM_TIERS = {"premium", "enterprise", "platinum", "gold", "critical"}

SLA_BREACH_PATTERNS = [
    re.compile(pattern)
    for pattern in [
        r"\b(sla|service level|response sla|resolution sla|4-hour|four-hour|priority customer|ticket)\b.{0,80}\bbreach(?:ed|es)?\b",
        r"\bbreach(?:ed|es)?\b.{0,80}\b(sla|service level|response sla|resolution sla|4-hour|four-hour|priority customer|ticket)\b",
    ]
]

SECURITY_BREACH_PATTERNS = [
    re.compile(pattern)
    for pattern in [
        r"\b(data|security|credential|credentials|privacy|customer data|api key|token|secret)\s+breach(?:ed|es)?\b",
        r"\bbreach(?:ed|es)?\s+(of\s+)?(data|security|credentials|privacy|customer data|api key|token|secret)\b",
        r"\bunauthorized\s+access\b",
        r"\b(customer\s+data|api\s+key|token|secret)\s+(exposed|leaked|compromised)\b",
    ]
]

INCIDENT_PROCESS_PATTERNS = [
    re.compile(pattern)
    for pattern in [
        r"\b(recommended|standard|best|process|procedure|lifecycle|workflow|phase|phases|steps)\b.{0,100}\b(incident|itil)\b",
        r"\b(incident|itil)\b.{0,100}\b(lifecycle|management|process|procedure|workflow|phase|phases|steps|best practice)\b",
        r"\b(support\s+)?(ownership|owner|responsibility|responsibilities|roles?|handoff|operating model|support model|ownership model)\b.{0,100}\b(incident|resolution)\b",
        r"\b(incident|resolution)\b.{0,100}\b(ownership|owner|responsibility|responsibilities|roles?|handoff|operating model|support model|ownership model)\b",
    ]
]

LIVE_INCIDENT_TERMS = {
    "active",
    "open",
    "current",
    "ongoing",
    "unresolved",
    "outage",
    "production",
    "down",
    "affected",
    "impacting",
    "customer",
    "customers",
    "ticket",
    "tickets",
    "region",
    "log",
    "logs",
    "status",
    "critical alert",
    "breach",
    "data loss",
    "vulnerability",
    "list",
    "show",
    "which",
    "check",
    "lookup",
    "more than",
}
CRITICAL_LIVE_INCIDENT_TERMS = {
    "active incident",
    "open incident",
    "current incident",
    "ongoing incident",
    "unresolved incident",
    "active critical",
    "unresolved critical",
    "critical alert",
    "outage",
    "production outage",
    "production down",
    "service down",
    "system down",
    "unavailable",
    "affected region",
    "impacting users",
    "impacting customers",
    "production impact",
    "breach",
    "data loss",
    "vulnerability",
}

UNSAFE_DATA_ACCESS_PATTERNS = [
    re.compile(pattern)
    for pattern in [
        r"\b(show|list|display|find|get|reveal|dump)\b.{0,80}\b(password|passwords|api[_\s-]?key|secret|secrets|token|credential|credentials)\b",
        r"\b(password|passwords|api[_\s-]?key|secret|secrets|token|credential|credentials)\b.{0,80}\b(internal_users|internal|table|database)\b",
    ]
]

FALSE_POSITIVE_RISK_EXPLANATION_PATTERNS = [
    re.compile(pattern)
    for pattern in [
        r"\b(word|term|phrase)\s+breach\b",
        r"\bbreach\b.{0,80}\b(policy|example|examples|explain|not\s+a\s+security|not\s+security)\b",
        r"\b(policy|example|examples|explain)\b.{0,80}\bbreach\b",
    ]
]


@dataclass(frozen=True)
class SeverityAssessmentResult:
    severity_priority: SeverityPriority
    severity: SeverityLevel
    severity_reason: str
    escalation_flag: bool
    escalation_target: EscalationTarget
    escalation_reason: str | None
    matched_indicators: list[str]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_list(state: SupportOrchestrationState, key: str, item: Any) -> None:
    values = list(state.get(key, []))
    values.append(item)
    state[key] = values  # type: ignore[literal-required]


def _normalize_query(query: str) -> str:
    return " ".join(query.lower().strip().split())


def _keyword_hits(query: str, keywords: set[str]) -> list[str]:
    return sorted({keyword for keyword in keywords if keyword in query})


def _pattern_hits(query: str, patterns: list[re.Pattern[str]], label: str) -> list[str]:
    return sorted({label for pattern in patterns if pattern.search(query)})


def _is_sla_breach_query(query: str) -> bool:
    return any(pattern.search(query) for pattern in SLA_BREACH_PATTERNS)


def _is_incident_process_documentation_query(query: str) -> bool:
    if not any(pattern.search(query) for pattern in INCIDENT_PROCESS_PATTERNS):
        return False
    return not any(term in query for term in LIVE_INCIDENT_TERMS)


def _has_live_incident_context(query: str) -> bool:
    return any(term in query for term in CRITICAL_LIVE_INCIDENT_TERMS)


def _is_unsafe_data_access_request(query: str) -> bool:
    return any(pattern.search(query) for pattern in UNSAFE_DATA_ACCESS_PATTERNS)


def _is_false_positive_risk_explanation(query: str) -> bool:
    return any(pattern.search(query) for pattern in FALSE_POSITIVE_RISK_EXPLANATION_PATTERNS)


def _metadata_customer_context(state: SupportOrchestrationState) -> dict[str, Any]:
    metadata = dict(state.get("metadata") or {})
    customer_context = metadata.get("customer_context") or {}
    return dict(customer_context) if isinstance(customer_context, dict) else {}


def _human_handoff_metadata(state: SupportOrchestrationState) -> dict[str, Any]:
    metadata = dict(state.get("metadata") or {})
    handoff = metadata.get("human_handoff") or {}
    return dict(handoff) if isinstance(handoff, dict) else {}


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "y", "1", "active", "open"}
    return bool(value)


def _is_premium_context(customer_context: dict[str, Any]) -> bool:
    tier_values = [
        customer_context.get("customer_tier"),
        customer_context.get("sla_tier"),
        customer_context.get("plan"),
        customer_context.get("subscription_plan"),
    ]
    return any(str(value).strip().lower() in PREMIUM_TIERS for value in tier_values if value is not None)


def _affected_users(customer_context: dict[str, Any]) -> int:
    value = customer_context.get("affected_users") or customer_context.get("impacted_users") or 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _map_priority(priority: SeverityPriority) -> SeverityLevel:
    return {
        SeverityPriority.P0: SeverityLevel.CRITICAL,
        SeverityPriority.P1: SeverityLevel.HIGH,
        SeverityPriority.P2: SeverityLevel.MEDIUM,
        SeverityPriority.P3: SeverityLevel.LOW,
    }[priority]


def _target_for_p0(query: str, intent: SupportIntent | None) -> EscalationTarget:
    security_hits = _keyword_hits(query, SECURITY_ESCALATION_KEYWORDS)
    security_hits.extend(_pattern_hits(query, SECURITY_BREACH_PATTERNS, "security breach context"))
    if security_hits or intent == SupportIntent.SECURITY:
        return EscalationTarget.SECURITY_TEAM
    return EscalationTarget.INCIDENT_RESPONSE


def _assess_local(state: SupportOrchestrationState) -> SeverityAssessmentResult:
    query = _normalize_query(str(state.get("query") or ""))
    intent = state.get("intent")
    route_decision = state.get("route_decision")
    customer_context = _metadata_customer_context(state)
    incident_investigation = dict(state.get("incident_investigation") or {})
    human_handoff = _human_handoff_metadata(state)

    sla_breach_query = _is_sla_breach_query(query)
    p0_hits = _keyword_hits(query, P0_KEYWORDS)
    if not sla_breach_query:
        p0_hits.extend(_pattern_hits(query, SECURITY_BREACH_PATTERNS, "security breach context"))
    p1_hits = _keyword_hits(query, P1_KEYWORDS)
    p2_hits = _keyword_hits(query, P2_KEYWORDS)
    p3_hits = _keyword_hits(query, P3_KEYWORDS)
    failure_hits = _keyword_hits(query, FAILURE_KEYWORDS)

    premium_context = _is_premium_context(customer_context)
    active_incident = _is_truthy(customer_context.get("active_incident"))
    open_critical_alert = _is_truthy(customer_context.get("open_critical_alert"))
    impacted_users = _affected_users(customer_context)
    documentation_process_query = _is_incident_process_documentation_query(query)
    risk_term_explanation = _is_false_positive_risk_explanation(query)
    live_incident_context = _has_live_incident_context(query)

    if _is_unsafe_data_access_request(query):
        reason = "P3 policy escalation: request asks for secrets, credentials, or non-allowlisted sensitive data."
        return SeverityAssessmentResult(
            severity_priority=SeverityPriority.P3,
            severity=_map_priority(SeverityPriority.P3),
            severity_reason=reason,
            escalation_flag=True,
            escalation_target=EscalationTarget.SECURITY_TEAM,
            escalation_reason=reason,
            matched_indicators=["unsafe sensitive-data access request"],
        )

    if _is_truthy(human_handoff.get("needs_engineering")):
        confirmed = str(human_handoff.get("confirmed_severity") or "High").strip().lower()
        priority = {
            "critical": SeverityPriority.P0,
            "p0": SeverityPriority.P0,
            "high": SeverityPriority.P1,
            "p1": SeverityPriority.P1,
            "medium": SeverityPriority.P2,
            "p2": SeverityPriority.P2,
            "low": SeverityPriority.P3,
            "p3": SeverityPriority.P3,
        }.get(confirmed, SeverityPriority.P1)
        reason = "Human handoff confirmed this case needs engineering ownership."
        return SeverityAssessmentResult(
            severity_priority=priority,
            severity=_map_priority(priority),
            severity_reason=reason,
            escalation_flag=True,
            escalation_target=EscalationTarget.ENGINEERING,
            escalation_reason=reason,
            matched_indicators=["human_handoff_needs_engineering"],
        )

    if premium_context and not documentation_process_query and (
        live_incident_context or active_incident or open_critical_alert
    ):
        p0_hits.append("premium/enterprise customer impact")

    if open_critical_alert:
        p0_hits.append("open critical alert")
    active_critical_correlated = bool(incident_investigation.get("active_critical_incident_correlated"))
    if active_critical_correlated and not documentation_process_query:
        p0_hits.append("correlated active critical incident investigation")
    elif incident_investigation.get("active_critical_incident") and not documentation_process_query and live_incident_context:
        p1_hits.append("active Critical incident exists but did not meet correlation threshold for P0")

    if risk_term_explanation:
        p0_hits = []
        p1_hits = []

    if p0_hits:
        target = _target_for_p0(query, intent)
        reason = "P0 detected due to critical indicators: " + ", ".join(sorted(set(p0_hits)))
        return SeverityAssessmentResult(
            severity_priority=SeverityPriority.P0,
            severity=_map_priority(SeverityPriority.P0),
            severity_reason=reason,
            escalation_flag=True,
            escalation_target=target,
            escalation_reason=reason,
            matched_indicators=sorted(set(p0_hits)),
        )

    if route_decision == RouteDecision.HIGH_RISK:
        p1_hits.append("high-risk route requires human review but no concrete P0 indicator was detected")

    if active_incident:
        p1_hits.append("active incident")
    if impacted_users >= 10:
        p1_hits.append("multiple affected users")
    if intent == SupportIntent.PERFORMANCE and ("severe" in query or "production" in query):
        p1_hits.append("severe production performance issue")

    if sla_breach_query and not risk_term_explanation:
        p1_hits.append("sla breach validation request")

    if re.search(r"\b(open|active|current|ongoing|unresolved)\b.{0,50}\bincident(s)?\b", query) or re.search(
        r"\bincident(s)?\b.{0,50}\b(open|active|current|ongoing|unresolved|more than)\b",
        query,
    ):
        p1_hits.append("active/open incident lookup")

    if p1_hits:
        reason = "P1 detected due to high-impact indicators: " + ", ".join(sorted(set(p1_hits)))
        return SeverityAssessmentResult(
            severity_priority=SeverityPriority.P1,
            severity=_map_priority(SeverityPriority.P1),
            severity_reason=reason,
            escalation_flag=False,
            escalation_target=EscalationTarget.NONE,
            escalation_reason=None,
            matched_indicators=sorted(set(p1_hits)),
        )

    how_to_only = bool(p3_hits) and not failure_hits and not any(code in query for code in ("401", "403", "500", "503"))

    if (p2_hits and not how_to_only) or intent == SupportIntent.PERFORMANCE:
        indicators = sorted(set(p2_hits or [str(intent.value if isinstance(intent, SupportIntent) else intent)]))
        reason = "P2 detected due to functional support indicators: " + ", ".join(indicators)
        return SeverityAssessmentResult(
            severity_priority=SeverityPriority.P2,
            severity=_map_priority(SeverityPriority.P2),
            severity_reason=reason,
            escalation_flag=False,
            escalation_target=EscalationTarget.NONE,
            escalation_reason=None,
            matched_indicators=indicators,
        )

    indicators = sorted(set(p3_hits or ["general support request"]))
    reason = "P3 detected due to low-risk/general support indicators: " + ", ".join(indicators)
    return SeverityAssessmentResult(
        severity_priority=SeverityPriority.P3,
        severity=_map_priority(SeverityPriority.P3),
        severity_reason=reason,
        escalation_flag=False,
        escalation_target=EscalationTarget.NONE,
        escalation_reason=None,
        matched_indicators=indicators,
    )


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


def _execution_result(result: SeverityAssessmentResult) -> ExecutionResult:
    return {
        "step_id": "severity-assessment",
        "agent_name": AGENT_NAME,
        "result_type": "severity_assessment",
        "summary": f"{result.severity_priority.value} -> {result.severity.value}",
        "data": {
            "severity_priority": result.severity_priority.value,
            "severity": result.severity.value,
            "severity_reason": result.severity_reason,
            "escalation_flag": result.escalation_flag,
            "escalation_target": result.escalation_target.value,
            "escalation_reason": result.escalation_reason,
            "matched_indicators": result.matched_indicators,
        },
        "error": None,
        "timestamp": _utc_now(),
    }


def _verification(result: SeverityAssessmentResult) -> VerificationOutcome:
    passed = bool(result.severity_priority and result.severity and result.severity_reason)
    return {
        "check_name": "severity_assessment_complete",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": "Severity priority, normalized severity, and severity reason were produced."
        if passed
        else "Severity assessment output was incomplete.",
        "corrective_action": None if passed else "Retry severity assessment.",
        "metadata": {
            "severity_priority": result.severity_priority.value,
            "severity": result.severity.value,
            "escalation_flag": result.escalation_flag,
        },
    }


@observe(name=AGENT_NAME, as_type="agent", capture_input=False, capture_output=False)
def assess_severity(state: SupportOrchestrationState) -> SupportOrchestrationState:
    """Assess operational severity and escalation requirement."""

    started = time.perf_counter()
    next_state: SupportOrchestrationState = dict(state)

    _append_list(next_state, "progress_updates", _progress("severity-assessment", "started", "Severity assessment started."))

    result = _assess_local(next_state)
    next_state["severity_priority"] = result.severity_priority
    next_state["severity"] = result.severity
    next_state["severity_reason"] = result.severity_reason
    existing_escalation_flag = bool(next_state.get("escalation_flag"))
    existing_escalation_target = next_state.get("escalation_target")
    existing_escalation_reason = next_state.get("escalation_reason")
    if result.escalation_flag:
        next_state["escalation_flag"] = True
        next_state["escalation_target"] = result.escalation_target
        next_state["escalation_reason"] = result.escalation_reason
    elif existing_escalation_flag:
        next_state["escalation_flag"] = True
        next_state["escalation_target"] = existing_escalation_target
        next_state["escalation_reason"] = existing_escalation_reason
    else:
        next_state["escalation_flag"] = False
        next_state["escalation_target"] = result.escalation_target
        next_state["escalation_reason"] = result.escalation_reason

    latency_ms = int((time.perf_counter() - started) * 1000)
    output_summary = (
        f"priority={result.severity_priority.value}; severity={result.severity.value}; "
        f"escalation={result.escalation_flag}"
    )

    _append_list(next_state, "execution_results", _execution_result(result))
    _append_list(next_state, "verification_outcomes", _verification(result))
    _append_list(
        next_state,
        "agent_trace",
        _trace(
            action="assess_severity",
            status="completed",
            input_summary=(
                f"query_length={len(str(next_state.get('query') or ''))}; "
                f"intent={getattr(next_state.get('intent'), 'value', next_state.get('intent'))}; "
                f"route={getattr(next_state.get('route_decision'), 'value', next_state.get('route_decision'))}"
            ),
            output_summary=output_summary,
            latency_ms=latency_ms,
        ),
    )
    _append_list(next_state, "progress_updates", _progress("severity-assessment", "completed", output_summary))

    trace_agent_state(
        agent_name=AGENT_NAME,
        input_state=state,
        output_state=next_state,
        started_at=started,
        tool_used="local_severity_rules",
    )

    return next_state
