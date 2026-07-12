import logging
import re
import time
from datetime import datetime, timezone
from typing import Any

import psycopg
from psycopg.rows import dict_row

from app.core.config import settings
from app.core.langfuse import observe, trace_agent_state, trace_guardrail_event
from app.orchestration.state import (
    AgentTraceEvent,
    EscalationTarget,
    ExecutionResult,
    GuardrailFlag,
    IncidentInvestigationResult,
    IncidentRecord,
    ProgressUpdate,
    SeverityLevel,
    SeverityPriority,
    SupportOrchestrationState,
    VerificationOutcome,
)

logger = logging.getLogger(__name__)

AGENT_NAME = "incident_investigator_agent"
ACTIVE_STATUSES = ["Open", "Investigating", "In Progress", "Escalated"]
KNOWN_REGIONS = {"US", "EU", "APAC", "MEA"}
KNOWN_SEVERITIES = {"Critical", "High", "Medium", "Low"}
HISTORY_KEYWORDS = {"resolved", "history", "historical", "past", "closed"}
INCIDENT_PROCESS_PATTERNS = [
    re.compile(pattern)
    for pattern in [
        r"\b(recommended|standard|best|process|procedure|lifecycle|workflow|phase|phases|steps)\b.{0,100}\b(incident|itil)\b",
        r"\b(incident|itil)\b.{0,100}\b(lifecycle|management|process|procedure|workflow|phase|phases|steps|best practice)\b",
        r"\b(support\s+)?(ownership|owner|responsibility|responsibilities|roles?|handoff|operating model|support model|ownership model)\b.{0,100}\b(incident|resolution)\b",
        r"\b(incident|resolution)\b.{0,100}\b(ownership|owner|responsibility|responsibilities|roles?|handoff|operating model|support model|ownership model)\b",
    ]
]
OPERATIONAL_INVESTIGATION_TERMS = {
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
ACTIVE_CRITICAL_CORRELATION_THRESHOLD = 0.65
ERROR_CODE_PATTERN = re.compile(r"\b[1-5][0-9]{2}\b")
CORRELATION_STOPWORDS = {
    "account",
    "accounts",
    "check",
    "customer",
    "customers",
    "error",
    "errors",
    "fix",
    "getting",
    "happening",
    "issue",
    "issues",
    "please",
    "suggest",
    "their",
    "there",
    "ticket",
    "tickets",
    "with",
}
DOMAIN_CORRELATION_GROUPS = {
    "authentication": {"401", "403", "auth", "authentication", "authorization", "oauth", "token"},
    "api": {"api", "endpoint", "integration", "request", "webhook"},
    "database": {"database", "db", "postgres", "sql", "query"},
    "billing": {"billing", "payment", "invoice", "subscription"},
    "latency": {"latency", "performance", "slow", "timeout", "timeouts"},
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_list(state: SupportOrchestrationState, key: str, item: Any) -> None:
    values = list(state.get(key, []))
    values.append(item)
    state[key] = values  # type: ignore[literal-required]


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _normalize(value: Any) -> str:
    return str(value or "").strip().lower()


def _stringify_time(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _customer_context(state: SupportOrchestrationState) -> dict[str, Any]:
    if state.get("customer_context"):
        return dict(state["customer_context"])
    metadata = dict(state.get("metadata") or {})
    context = metadata.get("customer_context") or {}
    return dict(context) if isinstance(context, dict) else {}


def _query_tokens(query: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9]+", query.lower())
        if (len(token) >= 3 or token.isdigit()) and token not in CORRELATION_STOPWORDS
    }


def _customer_name_candidates(state: SupportOrchestrationState) -> list[str]:
    context = _customer_context(state)
    metadata = dict(state.get("metadata") or {})
    query = str(state.get("query") or "")
    candidates: list[str] = []

    for source in (context, metadata):
        for key in ("company_name", "customer_name", "company", "account_name"):
            value = _clean_text(source.get(key))
            if value:
                candidates.append(value)

    match = re.search(
        r"\b(?:customer|company|account)\s+([a-zA-Z0-9&_. -]+?)(?:\s+(?:is|has|having|gets|getting|reports|reported|with|in|on|for|:)|[.?,]|$)",
        query,
        re.IGNORECASE,
    )
    if match:
        candidates.append(match.group(1))

    normalized: list[str] = []
    for candidate in candidates:
        cleaned = " ".join(candidate.strip(" .,:;!?").split())
        if len(cleaned) >= 3 and cleaned.lower() not in {value.lower() for value in normalized}:
            normalized.append(cleaned)
    return normalized


def _searchable_incident_text(row: dict[str, Any]) -> str:
    fields = [
        row.get("incident_type"),
        row.get("root_cause"),
        row.get("affected_region"),
        row.get("severity"),
        row.get("resolution_status"),
    ]
    return " ".join(str(field or "").lower() for field in fields)


def _domain_correlation_reasons(query: str, searchable: str) -> list[str]:
    query_tokens = _query_tokens(query)
    reasons: list[str] = []
    for group_name, terms in DOMAIN_CORRELATION_GROUPS.items():
        query_has_group = bool(query_tokens & terms)
        incident_has_group = any(term in searchable for term in terms)
        if query_has_group and incident_has_group:
            reasons.append(f"{group_name} domain match")
    return reasons


def _region_from_query(query: str) -> str | None:
    upper_query = query.upper()
    for region in KNOWN_REGIONS:
        if re.search(rf"\b{re.escape(region)}\b", upper_query):
            return region
    return None


def _region_filter(state: SupportOrchestrationState) -> str | None:
    context = _customer_context(state)
    return _clean_text(context.get("region")) or _region_from_query(str(state.get("query") or ""))


def _severity_filter(state: SupportOrchestrationState) -> str | None:
    query = _normalize(state.get("query"))
    state_severity = state.get("severity")
    severity_priority = state.get("severity_priority")

    if "critical" in query or severity_priority == SeverityPriority.P0 or state_severity == SeverityLevel.CRITICAL:
        return "Critical"
    if "high" in query or severity_priority == SeverityPriority.P1 or state_severity == SeverityLevel.HIGH:
        return "High"
    if "medium" in query or severity_priority == SeverityPriority.P2 or state_severity == SeverityLevel.MEDIUM:
        return "Medium"
    if "low" in query or severity_priority == SeverityPriority.P3 or state_severity == SeverityLevel.LOW:
        return "Low"
    return None


def _include_resolved(query: str) -> bool:
    normalized = _normalize(query)
    return any(keyword in normalized for keyword in HISTORY_KEYWORDS)


def _is_incident_process_documentation_query(query: str) -> bool:
    normalized = _normalize(query)
    if not any(pattern.search(normalized) for pattern in INCIDENT_PROCESS_PATTERNS):
        return False
    return not any(term in normalized for term in OPERATIONAL_INVESTIGATION_TERMS)


def _status_filter(query: str) -> list[str] | None:
    if _include_resolved(query):
        return None
    return ACTIVE_STATUSES


def _correlation_details(
    row: dict[str, Any],
    query: str,
    region: str | None,
    customer_names: list[str],
) -> tuple[float, list[str]]:
    tokens = _query_tokens(query)
    searchable = _searchable_incident_text(row)
    matched_tokens = sorted(token for token in tokens if token in searchable)
    score = min(0.35, len(matched_tokens) * 0.08)
    reasons: list[str] = []

    if matched_tokens:
        reasons.append("matched query terms: " + ", ".join(matched_tokens[:8]))

    matched_codes = sorted({code for code in ERROR_CODE_PATTERN.findall(query) if code in searchable})
    if matched_codes:
        score += 0.30
        reasons.append("matched error code(s): " + ", ".join(matched_codes))

    for customer_name in customer_names:
        if customer_name.lower() in searchable:
            score += 0.35
            reasons.append(f"matched customer/account name: {customer_name}")
            break

    if region and _normalize(row.get("affected_region")) == region.lower():
        score += 0.25
        reasons.append(f"matched affected region: {region}")

    domain_reasons = _domain_correlation_reasons(query, searchable)
    if domain_reasons:
        score += min(0.25, len(domain_reasons) * 0.15)
        reasons.extend(domain_reasons)

    if _normalize(row.get("severity")) == "critical":
        score += 0.10
    if _normalize(row.get("resolution_status")) in {status.lower() for status in ACTIVE_STATUSES}:
        score += 0.10

    return max(0.0, min(1.0, score)), reasons


def _is_active_critical(row: dict[str, Any]) -> bool:
    return _normalize(row.get("severity")) == "critical" and _normalize(row.get("resolution_status")) in {
        status.lower() for status in ACTIVE_STATUSES
    }


def _incident_record(
    row: dict[str, Any],
    query: str,
    region: str | None,
    customer_names: list[str],
) -> IncidentRecord:
    correlation_score, correlation_reasons = _correlation_details(row, query, region, customer_names)
    return {
        "incident_id": int(row["incident_id"]) if row.get("incident_id") is not None else None,
        "incident_type": _clean_text(row.get("incident_type")),
        "severity": _clean_text(row.get("severity")),
        "affected_region": _clean_text(row.get("affected_region")),
        "start_time": _stringify_time(row.get("start_time")),
        "end_time": _stringify_time(row.get("end_time")),
        "resolution_status": _clean_text(row.get("resolution_status")),
        "root_cause": _clean_text(row.get("root_cause")),
        "escalation_flag": bool(row.get("escalation_flag")),
        "correlation_score": correlation_score,
        "correlation_reasons": correlation_reasons,
    }


def _fetch_incidents(region: str | None, severity: str | None, statuses: list[str] | None) -> list[dict[str, Any]]:
    where_clauses = []
    params: list[Any] = []

    if region:
        where_clauses.append("affected_region = %s")
        params.append(region)
    if severity:
        where_clauses.append("severity = %s")
        params.append(severity)
    if statuses:
        placeholders = ", ".join(["%s"] * len(statuses))
        where_clauses.append(f"resolution_status IN ({placeholders})")
        params.extend(statuses)

    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
    query = f"""
        SELECT incident_id, incident_type, severity, affected_region, start_time, end_time,
               resolution_status, root_cause, escalation_flag
        FROM incident_logs
        {where_sql}
        ORDER BY start_time DESC NULLS LAST
        LIMIT 25
    """

    with psycopg.connect(settings.database_dsn, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            return [dict(row) for row in cur.fetchall()]


def _empty_result(filters: dict[str, Any], summary: str) -> IncidentInvestigationResult:
    return {
        "filters_used": filters,
        "matched_incidents": [],
        "active_critical_incident": False,
        "active_critical_incident_correlated": False,
        "max_correlation_score": 0.0,
        "correlation_threshold": ACTIVE_CRITICAL_CORRELATION_THRESHOLD,
        "investigation_summary": summary,
    }


def _build_result(state: SupportOrchestrationState) -> IncidentInvestigationResult:
    query = str(state.get("query") or "")
    region = _region_filter(state)
    severity = _severity_filter(state)
    statuses = _status_filter(query)
    customer_names = _customer_name_candidates(state)
    filters = {
        "region": region,
        "severity": severity,
        "resolution_status": statuses or "all",
        "include_resolved": statuses is None,
        "customer_name_candidates": customer_names,
    }

    rows = _fetch_incidents(region, severity, statuses)
    records = [_incident_record(row, query, region, customer_names) for row in rows]
    records = sorted(
        records,
        key=lambda record: (
            _normalize(record.get("severity")) == "critical"
            and _normalize(record.get("resolution_status")) in {status.lower() for status in ACTIVE_STATUSES},
            float(record.get("correlation_score") or 0.0),
            record.get("start_time") or "",
        ),
        reverse=True,
    )[:5]

    active_critical_records = [record for record in records if _is_active_critical(record)]
    active_critical = bool(active_critical_records)
    max_correlation_score = max(
        [float(record.get("correlation_score") or 0.0) for record in active_critical_records],
        default=0.0,
    )
    active_critical_correlated = any(
        float(record.get("correlation_score") or 0.0) >= ACTIVE_CRITICAL_CORRELATION_THRESHOLD
        for record in active_critical_records
    )
    if not records:
        return _empty_result(filters, "No incidents matched the investigation filters.")

    summary = (
        f"Found {len(records)} incident(s); active critical incident present: {active_critical}; "
        f"correlated active critical incident: {active_critical_correlated}; "
        f"max active-critical correlation: {max_correlation_score:.2f}."
    )
    return {
        "filters_used": filters,
        "matched_incidents": records,
        "active_critical_incident": active_critical,
        "active_critical_incident_correlated": active_critical_correlated,
        "max_correlation_score": max_correlation_score,
        "correlation_threshold": ACTIVE_CRITICAL_CORRELATION_THRESHOLD,
        "investigation_summary": summary,
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


def _guardrail(result: IncidentInvestigationResult) -> GuardrailFlag:
    active_critical = bool(result.get("active_critical_incident"))
    correlated = bool(result.get("active_critical_incident_correlated"))
    if correlated:
        reason = "Correlated active Critical incident exists."
        severity = SeverityLevel.CRITICAL
    elif active_critical:
        reason = "Active Critical incident exists, but it is not sufficiently correlated to this request."
        severity = SeverityLevel.MEDIUM
    else:
        reason = "No active Critical incident found."
        severity = SeverityLevel.LOW
    return {
        "name": "active_critical_incident_check",
        "passed": not correlated,
        "severity": severity,
        "reason": reason,
        "metadata": {
            "filters_used": result.get("filters_used", {}),
            "incident_count": len(result.get("matched_incidents", [])),
            "active_critical_incident": active_critical,
            "active_critical_incident_correlated": correlated,
            "max_correlation_score": result.get("max_correlation_score", 0.0),
            "correlation_threshold": result.get("correlation_threshold", ACTIVE_CRITICAL_CORRELATION_THRESHOLD),
        },
    }


def _execution_result(result: IncidentInvestigationResult, error: str | None = None) -> ExecutionResult:
    return {
        "step_id": "incident-investigation",
        "agent_name": AGENT_NAME,
        "result_type": "incident_investigation",
        "summary": result.get("investigation_summary", "Incident investigation completed."),
        "data": dict(result),
        "error": error,
        "timestamp": _utc_now(),
    }


def _verification(result: IncidentInvestigationResult, error: str | None = None) -> VerificationOutcome:
    passed = error is None
    return {
        "check_name": "incident_investigation_complete",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": error or result.get("investigation_summary", "Incident investigation completed."),
        "corrective_action": None if passed else "Retry incident investigation or escalate for manual review.",
        "metadata": {
            "active_critical_incident": result.get("active_critical_incident", False),
            "active_critical_incident_correlated": result.get("active_critical_incident_correlated", False),
            "max_correlation_score": result.get("max_correlation_score", 0.0),
            "incident_count": len(result.get("matched_incidents", [])),
        },
    }


@observe(name=AGENT_NAME, as_type="agent", capture_input=False, capture_output=False)
async def investigate_incidents(state: SupportOrchestrationState) -> SupportOrchestrationState:
    """Investigate structured incident logs and flag active Critical incidents."""

    started = time.perf_counter()
    next_state: SupportOrchestrationState = dict(state)

    _append_list(next_state, "progress_updates", _progress("incident-investigation", "started", "Incident investigation started."))

    error: str | None = None
    status = "completed"
    try:
        query = str(next_state.get("query") or "")
        if _is_incident_process_documentation_query(query):
            status = "skipped"
            result = _empty_result(
                {"skipped": True, "reason": "documentation_process_query"},
                "Incident investigation skipped because the query asks for lifecycle/process documentation, not live incident validation.",
            )
        else:
            result = _build_result(next_state)
    except Exception as exc:
        logger.exception("Incident investigation failed")
        error = str(exc)
        status = "failed"
        result = _empty_result({}, f"Incident investigation failed: {error}")
        _append_list(next_state, "errors", result["investigation_summary"])

    next_state["incident_investigation"] = result

    if result.get("active_critical_incident_correlated"):
        reason = "Correlated active Critical incident exists in incident_logs."
        existing_reason = next_state.get("escalation_reason")
        next_state["escalation_flag"] = True
        next_state["escalation_target"] = EscalationTarget.INCIDENT_RESPONSE
        next_state["escalation_reason"] = f"{existing_reason}; {reason}" if existing_reason else reason

    latency_ms = int((time.perf_counter() - started) * 1000)
    output_summary = (
        f"incidents={len(result.get('matched_incidents', []))}; "
        f"active_critical={result.get('active_critical_incident', False)}; "
        f"correlated={result.get('active_critical_incident_correlated', False)}; "
        f"max_score={float(result.get('max_correlation_score') or 0.0):.2f}; status={status}"
    )

    guardrail = _guardrail(result)
    _append_list(next_state, "guardrail_flags", guardrail)
    trace_guardrail_event(
        name=str(guardrail["name"]),
        passed=bool(guardrail["passed"]),
        reason=str(guardrail["reason"]),
        metadata=dict(guardrail.get("metadata") or {}),
    )
    _append_list(next_state, "execution_results", _execution_result(result, error))
    _append_list(next_state, "verification_outcomes", _verification(result, error))
    _append_list(
        next_state,
        "agent_trace",
        _trace(
            action="investigate_incidents",
            status=status,
            input_summary=str(result.get("filters_used", {})),
            output_summary=output_summary,
            latency_ms=latency_ms,
        ),
    )
    _append_list(next_state, "progress_updates", _progress("incident-investigation", "completed", output_summary))

    trace_agent_state(
        agent_name=AGENT_NAME,
        input_state=state,
        output_state=next_state,
        started_at=started,
        tool_used="postgres_incident_logs_table",
    )

    return next_state
