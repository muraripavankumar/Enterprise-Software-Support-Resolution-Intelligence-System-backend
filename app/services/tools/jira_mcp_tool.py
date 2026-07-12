import asyncio
import hashlib
import json
import os
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.core.config import ConfigurationError, settings
from app.core.langfuse import observe, trace_tool_result
from app.core.logging import get_logger
from app.orchestration.state import (
    EscalationTarget,
    JiraTrackingResult,
    SeverityLevel,
    SeverityPriority,
    SupportOrchestrationState,
)

logger = get_logger(__name__)

SECURITY_REASON_CODES = {"security_breach", "data_breach", "unauthorized_access"}
OUTAGE_REASON_CODES = {"production_outage", "service_unresponsive", "active_critical_incident", "systemic_failure"}
DATA_LOSS_REASON_CODES = {"data_loss"}
ENGINEERING_TARGETS = {
    EscalationTarget.SECURITY_TEAM.value,
    EscalationTarget.ENGINEERING.value,
    EscalationTarget.INCIDENT_RESPONSE.value,
}


@dataclass(frozen=True)
class JiraIssueMapping:
    project_key: str
    issue_type: str
    priority: str
    reason_code: str
    dedupe_key: str
    dedupe_label: str
    labels: list[str]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _enum_value(value: Any) -> str | None:
    if value is None:
        return None
    return getattr(value, "value", str(value))


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _sanitize_label(value: str) -> str:
    label = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip().lower()).strip("-")
    return label or "unknown"


def _normalize_dedupe_text(value: Any, max_chars: int = 180) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    text = re.sub(r"\s+", " ", text)
    return text[:max_chars].strip()


def _collect_field_values(value: Any, field_names: set[str]) -> list[str]:
    values: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if key in field_names and child not in (None, "", "unknown"):
                    values.append(str(child))
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return sorted(set(_normalize_dedupe_text(item, 80) for item in values if _normalize_dedupe_text(item, 80)))


def _dedupe_components(state: SupportOrchestrationState, reason_code: str) -> dict[str, Any]:
    customer_context = _as_dict(state.get("customer_context"))
    incident_result = _as_dict(state.get("incident_investigation"))
    sql_results = list(state.get("sql_results") or [])
    metadata = _as_dict(state.get("metadata"))

    incident_ids = _collect_field_values(incident_result.get("matched_incidents") or [], {"incident_id", "id"})
    support_ticket_ids = _collect_field_values(sql_results, {"ticket_id", "support_ticket_id"})
    customer_ids = _collect_field_values(
        {
            "customer_context": customer_context,
            "sql_results": sql_results,
            "metadata": metadata,
        },
        {"customer_id", "account_id", "company_name", "customer_name"},
    )

    stable_identifiers = incident_ids or support_ticket_ids or customer_ids
    components: dict[str, Any] = {
        "reason_code": _sanitize_label(reason_code),
        "target": _sanitize_label(_target_value(state) or "unknown"),
        "severity_priority": _sanitize_label(_enum_value(state.get("severity_priority")) or "unknown"),
        "severity": _sanitize_label(_enum_value(state.get("severity")) or "unknown"),
        "incident_ids": incident_ids,
        "support_ticket_ids": support_ticket_ids,
        "customer_ids": customer_ids,
    }

    # Use the normalized user request only when ERIS has no stable operational
    # identifier. This lets repeats of the same free-form issue dedupe while
    # preventing broad reason-code matches from linking unrelated incidents.
    if not stable_identifiers:
        components["query"] = _normalize_dedupe_text(state.get("query"), 220)
    return components


def _dedupe_key(state: SupportOrchestrationState, reason_code: str) -> str:
    payload = json.dumps(_dedupe_components(state, reason_code), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _jql_quote(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _safe_text(value: Any, max_chars: int = 3000) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _display_value(value: Any, fallback: str = "Not available") -> str:
    if value is None:
        return fallback
    text = _safe_text(value, 500)
    if not text or text.lower() in {"none", "null", "unknown"}:
        return fallback
    return text


def _wiki_bullets(items: list[str]) -> str:
    cleaned = [_safe_text(item, 1200) for item in items if _safe_text(item, 1200)]
    return "\n".join(f"* {item}" for item in cleaned) if cleaned else "* Not available"


def _compact_json(value: Any, max_chars: int = 2500) -> str:
    text = json.dumps(value, indent=2, default=str)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _humanize_value(value: Any, fallback: str = "Not available") -> str:
    text = _display_value(value, fallback)
    if text == fallback:
        return text
    if re.fullmatch(r"p\d", text, flags=re.IGNORECASE):
        return text.upper()
    return text.replace("_", " ").title()


def _reason_source(state: SupportOrchestrationState) -> str:
    parts = [
        state.get("escalation_reason"),
        state.get("severity_reason"),
        state.get("query"),
    ]
    incident_result = _as_dict(state.get("incident_investigation"))
    if incident_result:
        parts.append(incident_result.get("investigation_summary"))
        for incident in list(incident_result.get("matched_incidents") or [])[:3]:
            raw_incident = _as_dict(incident)
            parts.extend(
                [
                    raw_incident.get("incident_type"),
                    raw_incident.get("root_cause"),
                    raw_incident.get("resolution_status"),
                    raw_incident.get("severity"),
                ]
            )
    return " ".join(str(part) for part in parts if part).lower()


def _reason_code(state: SupportOrchestrationState) -> str:
    source = _reason_source(state)
    incident_result = _as_dict(state.get("incident_investigation"))
    if re.search(r"\b(data breach|breached data|customer data exposure)\b", source):
        return "data_breach"
    if re.search(r"\b(unauthorized access|permission bypass|privilege escalation)\b", source):
        return "unauthorized_access"
    if re.search(r"\b(security breach|security vulnerability|vulnerability exposure|exploit)\b", source):
        return "security_breach"
    if re.search(r"\b(data loss|lost data|missing records|deleted data)\b", source):
        return "data_loss"
    if re.search(r"\b(production outage|prod outage|outage|production down|service down)\b", source):
        return "production_outage"
    if re.search(r"\b(service unresponsive|unresponsive|timeouts?|5xx|http 500|http 503)\b", source):
        return "service_unresponsive"
    if re.search(r"\b(systemic failure|multiple tickets|pattern|many customers|widespread)\b", source):
        return "systemic_failure"
    if re.search(r"\b(active critical incident|critical incident|unresolved critical alert)\b", source) and (
        not incident_result or bool(incident_result.get("active_critical_incident_correlated"))
    ):
        return "active_critical_incident"
    if re.search(r"\b(sla breach|breached the .*sla|response sla|sla miss|sla-miss)\b", source):
        return "sla_miss"
    return "engineering_handoff"


def _human_handoff(state: SupportOrchestrationState) -> dict[str, Any]:
    metadata = _as_dict(state.get("metadata"))
    for key in ("human_handoff", "human_handoff_result", "human_resolution", "interrupt_result"):
        candidate = _as_dict(metadata.get(key))
        if candidate:
            return candidate
    return metadata


def _human_needs_engineering(state: SupportOrchestrationState) -> bool:
    handoff = _human_handoff(state)
    return str(handoff.get("needs_engineering", "")).lower() in {"1", "true", "yes", "y"} or str(
        handoff.get("resolution", "")
    ).lower() in {"needs_engineering", "engineering_required", "create_jira"}


def _target_value(state: SupportOrchestrationState) -> str:
    return _enum_value(state.get("escalation_target")) or ""


def should_create_jira_issue(state: SupportOrchestrationState) -> tuple[bool, str]:
    """Return whether escalation should create an engineering-bound Jira issue."""

    reason_code = _reason_code(state)
    if reason_code == "sla_miss":
        return False, reason_code

    if _human_needs_engineering(state):
        return True, reason_code

    severity_priority = _enum_value(state.get("severity_priority"))
    severity = _enum_value(state.get("severity"))
    is_p0 = severity_priority == SeverityPriority.P0.value or severity == SeverityLevel.CRITICAL.value
    if not is_p0:
        return False, reason_code

    target = _target_value(state)
    if target in ENGINEERING_TARGETS:
        return True, reason_code

    if target == EscalationTarget.L2_SUPPORT.value and reason_code in OUTAGE_REASON_CODES | DATA_LOSS_REASON_CODES:
        return True, reason_code

    return False, reason_code


def _confirmed_human_issue_type(state: SupportOrchestrationState) -> str | None:
    handoff = _human_handoff(state)
    issue_type = str(handoff.get("jira_issue_type") or handoff.get("issue_type") or "").strip()
    return issue_type or None


def _confirmed_human_priority(state: SupportOrchestrationState) -> str | None:
    handoff = _human_handoff(state)
    priority = str(handoff.get("jira_priority") or handoff.get("priority") or "").strip()
    if priority:
        return priority
    severity = str(handoff.get("confirmed_severity") or handoff.get("severity") or "").lower()
    if severity in {"p0", "critical"}:
        return settings.jira_critical_priority
    if severity in {"p1", "high"}:
        return settings.jira_high_priority
    if severity in {"p2", "medium"}:
        return settings.jira_medium_priority
    return None


def _issue_mapping(state: SupportOrchestrationState, reason_code: str) -> JiraIssueMapping:
    if reason_code in SECURITY_REASON_CODES:
        project_key = settings.jira_security_project_key or settings.jira_project_key or ""
        issue_type = settings.jira_critical_issue_type
        priority = settings.jira_critical_priority
    elif reason_code in OUTAGE_REASON_CODES:
        project_key = settings.jira_project_key or ""
        issue_type = settings.jira_critical_issue_type
        priority = settings.jira_critical_priority
    elif reason_code in DATA_LOSS_REASON_CODES:
        project_key = settings.jira_project_key or ""
        issue_type = settings.jira_critical_issue_type
        priority = settings.jira_high_priority
    elif _human_needs_engineering(state):
        project_key = settings.jira_project_key or ""
        issue_type = _confirmed_human_issue_type(state) or settings.jira_human_engineering_issue_type
        priority = _confirmed_human_priority(state) or settings.jira_high_priority
    else:
        project_key = settings.jira_project_key or ""
        issue_type = settings.jira_default_issue_type
        priority = settings.jira_high_priority

    dedupe_key = _dedupe_key(state, reason_code)
    dedupe_label = f"eris-dedupe-{dedupe_key}"
    labels = [
        _sanitize_label(settings.jira_default_label),
        "eris-autocreated",
        f"eris-{_sanitize_label(reason_code)}",
        dedupe_label,
    ]
    return JiraIssueMapping(
        project_key=project_key,
        issue_type=issue_type,
        priority=priority,
        reason_code=reason_code,
        dedupe_key=dedupe_key,
        dedupe_label=dedupe_label,
        labels=labels,
    )


def _dedupe_jql(mapping: JiraIssueMapping) -> str:
    window_days = max(int(settings.jira_dedupe_window_days or 7), 1)
    return (
        f'project = "{_jql_quote(mapping.project_key)}" '
        f"AND resolution = Unresolved "
        f"AND updated >= -{window_days}d "
        f'AND labels = "{_jql_quote(mapping.dedupe_label)}" '
        "ORDER BY updated DESC"
    )


def _ticket_id(state: SupportOrchestrationState) -> str:
    metadata = _as_dict(state.get("metadata"))
    for key in ("ticket_id", "support_ticket_id", "request_id", "conversation_id", "thread_id"):
        value = metadata.get(key) or state.get(key)  # type: ignore[literal-required]
        if value:
            return str(value)
    return "unknown"


def _audit_reference(state: SupportOrchestrationState) -> str:
    metadata = _as_dict(state.get("metadata"))
    if metadata.get("audit_log_url"):
        return str(metadata["audit_log_url"])
    if metadata.get("langfuse_trace_url"):
        return str(metadata["langfuse_trace_url"])
    trace_id = state.get("langfuse_trace_id") or metadata.get("langfuse_trace_id")
    if trace_id and settings.audit_log_base_url:
        return f"{settings.audit_log_base_url.rstrip('/')}/{trace_id}"
    if trace_id:
        return f"Langfuse trace id: {trace_id}"
    return "Agent trace is included in the escalation package metadata."


def _evidence_summary(state: SupportOrchestrationState) -> str:
    sections: list[str] = []
    chunks = list(state.get("retrieved_chunks") or [])[:5]
    if chunks:
        lines: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            metadata = _as_dict(chunk.get("metadata"))
            source = chunk.get("source_file") or metadata.get("source_file") or metadata.get("source") or "unknown source"
            page = chunk.get("page_number") or metadata.get("page_number")
            source_text = f"{source}" + (f", page {page}" if page else "")
            lines.append(f"* Evidence {index}: {source_text} - {_safe_text(chunk.get('chunk_text'), 500)}")
        sections.append("h3. Document evidence\n" + "\n".join(lines))

    sql_results = list(state.get("sql_results") or [])[:3]
    if sql_results:
        lines = []
        for index, result in enumerate(sql_results, start=1):
            lines.append(
                "* Structured result "
                f"{index}: tables={', '.join(result.get('tables_used') or []) or 'unknown'}; "
                f"rows={result.get('row_count', 'unknown')}; "
                f"summary={_safe_text(result.get('answer'), 500)}"
            )
        sections.append("h3. Structured data evidence\n" + "\n".join(lines))

    customer_context = _as_dict(state.get("customer_context"))
    if customer_context:
        account_lines = [
            f"* Customer: {_display_value(customer_context.get('company_name'))}",
            f"* Customer ID: {_display_value(customer_context.get('customer_id'))}",
            f"* Account status: {_display_value(customer_context.get('account_status'))}",
            f"* Subscription/SLA: {_display_value(customer_context.get('subscription_tier'))} / {_display_value(customer_context.get('sla_level'))}",
            f"* Region: {_display_value(customer_context.get('region'))}",
            f"* Lookup status: {_display_value(customer_context.get('lookup_status'))}",
        ]
        sections.append("h3. Account context\n" + "\n".join(account_lines))

    incident_result = _as_dict(state.get("incident_investigation"))
    if incident_result:
        lines = [
            f"* Summary: {_display_value(incident_result.get('investigation_summary'))}",
            f"* Active critical incident present: {_display_value(incident_result.get('active_critical_incident_correlated'))}",
            f"* Max correlation score: {_display_value(incident_result.get('max_correlation_score'))}",
        ]
        matched = list(incident_result.get("matched_incidents") or [])[:5]
        if matched:
            for index, raw_incident in enumerate(matched, start=1):
                incident = _as_dict(raw_incident)
                lines.append(
                    "* Matched incident "
                    f"{index}: id={_display_value(incident.get('incident_id'))}; "
                    f"type={_display_value(incident.get('incident_type'))}; "
                    f"severity={_display_value(incident.get('severity'))}; "
                    f"region={_display_value(incident.get('affected_region'))}; "
                    f"status={_display_value(incident.get('resolution_status'))}; "
                    f"root cause={_display_value(incident.get('root_cause'))}; "
                    f"correlation={_display_value(incident.get('correlation_score'))}"
                )
        sections.append("h3. Incident investigation\n" + "\n".join(lines))

    return "\n\n".join(sections) if sections else "No structured evidence was available before escalation."


def _impact_summary(state: SupportOrchestrationState, mapping: JiraIssueMapping) -> list[str]:
    incident_result = _as_dict(state.get("incident_investigation"))
    active_critical = incident_result.get("active_critical_incident_correlated")
    return [
        f"Priority: {mapping.priority}",
        f"Severity: {_humanize_value(_enum_value(state.get('severity_priority')))} / {_humanize_value(_enum_value(state.get('severity')))}",
        f"Escalation target: {_humanize_value(_target_value(state))}",
        f"Reason code: {_humanize_value(mapping.reason_code)}",
        f"Active critical incident correlated: {_display_value(active_critical)}",
    ]


def _recommended_engineering_actions(state: SupportOrchestrationState, mapping: JiraIssueMapping) -> list[str]:
    actions = [
        "Assign an owning engineer or incident commander.",
        "Validate the correlated incident and confirm customer impact.",
        "Review retrieved evidence and structured operational data before remediation.",
        "Post status updates in this Jira issue until mitigation or closure.",
    ]
    if mapping.reason_code in SECURITY_REASON_CODES:
        actions.insert(1, "Confirm exposure scope, affected systems, and patch/remediation deadline.")
    if mapping.reason_code in OUTAGE_REASON_CODES:
        actions.insert(1, "Confirm live service health, affected regions, and rollback/mitigation options.")
    return actions


def _agent_trace_summary(state: SupportOrchestrationState) -> str:
    trace_rows = []
    for event in list(state.get("agent_trace") or [])[-10:]:
        row = _as_dict(event)
        trace_rows.append(
            "* "
            f"{_display_value(row.get('agent_name'))}: "
            f"{_display_value(row.get('status'))}; "
            f"{_display_value(row.get('output_summary') or row.get('action'))}; "
            f"latency={_display_value(row.get('latency_ms'), '0')}ms"
        )
    return "\n".join(trace_rows) if trace_rows else "* No agent trace events were captured."


def _description(state: SupportOrchestrationState, mapping: JiraIssueMapping) -> str:
    reason = _safe_text(state.get("escalation_reason") or state.get("severity_reason"), 1500)
    return "\n\n".join(
        [
            "h2. ERIS Engineering Escalation",
            "h3. Executive summary",
            _wiki_bullets(
                [
                    f"Original request: {_safe_text(state.get('query'), 1000)}",
                    f"Why this was escalated: {reason}",
                    f"Audit reference: {_audit_reference(state)}",
                ]
            ),
            "h3. Impact and routing",
            _wiki_bullets(_impact_summary(state, mapping)),
            "h3. Dedupe identity",
            _wiki_bullets(
                [
                    f"Dedupe key: {mapping.dedupe_key}",
                    f"Dedupe label: {mapping.dedupe_label}",
                    "ERIS updates this Jira issue only when a future escalation has the same issue fingerprint.",
                ]
            ),
            "h3. Immediate next actions",
            _wiki_bullets(_recommended_engineering_actions(state, mapping)),
            "h3. Evidence collected by ERIS",
            _evidence_summary(state),
            "h3. Agent trace summary",
            _agent_trace_summary(state),
            "h3. Technical appendix",
            "{code:json}\n"
            + _compact_json(
                {
                    "ticket_id": _ticket_id(state),
                    "confidence_score": state.get("confidence_score"),
                    "labels": mapping.labels,
                    "metadata": _as_dict(state.get("metadata")),
                }
            )
            + "\n{code}",
        ]
    )


def _summary(state: SupportOrchestrationState, mapping: JiraIssueMapping) -> str:
    reason = mapping.reason_code.replace("_", " ").title()
    severity = _humanize_value(_enum_value(state.get("severity_priority")), "P?")
    target = _target_value(state).replace("_", " ").title() or "Engineering"
    query = _safe_text(state.get("query"), 75)
    return f"[ERIS][{severity}] {reason} -> {target}: {query}"


def _comment_body(state: SupportOrchestrationState, duplicate: bool = False) -> str:
    action = "Linked duplicate escalation to existing Jira issue." if duplicate else "ERIS escalation update."
    reason_code = _reason_code(state)
    dedupe_key = _dedupe_key(state, reason_code)
    return "\n\n".join(
        [
            f"h3. {action}",
            _wiki_bullets(
                [
                    f"Ticket ID: {_ticket_id(state)}",
                    f"Dedupe key: {dedupe_key}",
                    f"Reason code: {_humanize_value(reason_code)}",
                    f"Original query: {_safe_text(state.get('query'), 1000)}",
                    f"Escalation reason: {_safe_text(state.get('escalation_reason') or state.get('severity_reason'), 1000)}",
                    f"Audit reference: {_audit_reference(state)}",
                ]
            ),
        ]
    )


def _extract_text_content(result: Any) -> str:
    pieces: list[str] = []
    structured = getattr(result, "structuredContent", None)
    if structured:
        pieces.append(json.dumps(structured, default=str))
    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", None)
        if text:
            pieces.append(str(text))
    if not pieces:
        try:
            pieces.append(json.dumps(result.model_dump(), default=str))
        except Exception:
            pieces.append(str(result))
    return "\n".join(pieces)


def _extract_issue_keys(payload: Any, project_key: str) -> list[str]:
    keys: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            maybe_key = value.get("key") or value.get("issue_key")
            if isinstance(maybe_key, str) and maybe_key.startswith(f"{project_key}-"):
                keys.append(maybe_key)
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, str):
            keys.extend(re.findall(rf"\b{re.escape(project_key)}-\d+\b", value))

    if isinstance(payload, str):
        try:
            visit(json.loads(payload))
        except json.JSONDecodeError:
            visit(payload)
    else:
        visit(payload)

    deduped: list[str] = []
    for key in keys:
        if key not in deduped:
            deduped.append(key)
    return deduped


def _issue_url(issue_key: str) -> str:
    return f"{str(settings.jira_url).rstrip('/')}/browse/{issue_key}"


def _root_error_message(exc: BaseException) -> str:
    if isinstance(exc, BaseExceptionGroup):
        parts = [_root_error_message(item) for item in exc.exceptions]
        return "; ".join(part for part in parts if part) or str(exc)
    return str(exc) or exc.__class__.__name__


@asynccontextmanager
async def _jira_mcp_session() -> AsyncIterator[ClientSession]:
    env = {
        "JIRA_URL": str(settings.jira_url or ""),
        "JIRA_USERNAME": str(settings.jira_username or ""),
        "JIRA_API_TOKEN": str(settings.jira_api_token or ""),
        "JIRA_PROJECTS_FILTER": str(settings.jira_project_key or ""),
    }
    if settings.jira_security_project_key:
        env["JIRA_PROJECTS_FILTER"] = f"{settings.jira_project_key},{settings.jira_security_project_key}"

    params = StdioServerParameters(
        command=settings.jira_mcp_command,
        args=settings.jira_mcp_arg_list,
        env=env,
        encoding="utf-8",
        encoding_error_handler="replace",
    )
    with open(os.devnull, "w", encoding="utf-8") as errlog:
        async with stdio_client(params, errlog=errlog) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await asyncio.wait_for(session.initialize(), timeout=settings.jira_mcp_timeout_seconds)
                yield session


async def _call_tool(session: ClientSession, name: str, arguments: dict[str, Any]) -> Any:
    started = time.perf_counter()
    try:
        result = await asyncio.wait_for(
            session.call_tool(name, arguments=arguments),
            timeout=settings.jira_mcp_timeout_seconds,
        )
        logger.info(
            "jira_mcp_call_succeeded",
            extra={"tool_name": name, "duration_ms": int((time.perf_counter() - started) * 1000)},
        )
        return result
    except Exception:
        logger.error(
            "jira_mcp_call_failed",
            extra={"tool_name": name, "duration_ms": int((time.perf_counter() - started) * 1000)},
            exc_info=True,
        )
        raise


async def _search_existing(session: ClientSession, mapping: JiraIssueMapping) -> tuple[str | None, str]:
    jql = _dedupe_jql(mapping)
    result = await _call_tool(
        session,
        "jira_search",
        {
            "jql": jql,
            "fields": "key,summary,status,priority,labels,updated",
            "limit": 5,
            "projects_filter": mapping.project_key,
        },
    )
    text = _extract_text_content(result)
    keys = _extract_issue_keys(text, mapping.project_key)
    return (keys[0] if keys else None), jql


async def _add_comment(session: ClientSession, issue_key: str, body: str) -> None:
    await _call_tool(session, "jira_add_comment", {"issue_key": issue_key, "body": body})


async def _create_issue(session: ClientSession, state: SupportOrchestrationState, mapping: JiraIssueMapping) -> str:
    additional_fields: dict[str, Any] = {
        "priority": {"name": mapping.priority},
        "labels": mapping.labels,
    }
    if settings.jira_component_name:
        additional_fields["components"] = [{"name": settings.jira_component_name}]

    arguments: dict[str, Any] = {
        "project_key": mapping.project_key,
        "summary": _summary(state, mapping),
        "issue_type": mapping.issue_type,
        "description": _description(state, mapping),
        "additional_fields": json.dumps(additional_fields),
    }
    if settings.jira_component_name:
        arguments["components"] = settings.jira_component_name

    result = await _call_tool(session, "jira_create_issue", arguments)
    text = _extract_text_content(result)
    keys = _extract_issue_keys(text, mapping.project_key)
    if not keys:
        raise RuntimeError(f"Jira issue creation completed but no issue key was returned: {_safe_text(text, 500)}")
    return keys[0]


async def _transition_to_triage(session: ClientSession, issue_key: str) -> tuple[bool, str | None]:
    if not settings.jira_triage_status:
        return False, None

    if settings.jira_triage_transition_id:
        await _call_tool(
            session,
            "jira_transition_issue",
            {"issue_key": issue_key, "transition_id": settings.jira_triage_transition_id},
        )
        return True, await _verified_status(session, issue_key)

    try:
        transitions = await _call_tool(session, "jira_get_transitions", {"issue_key": issue_key})
        text = _extract_text_content(transitions)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = text
        transition_id = _find_transition_id(payload, settings.jira_triage_status)
        if transition_id:
            await _call_tool(session, "jira_transition_issue", {"issue_key": issue_key, "transition_id": transition_id})
            return True, await _verified_status(session, issue_key)
    except Exception as exc:
        logger.warning("failed to resolve Jira triage transition for %s: %s", issue_key, exc)

    try:
        await _call_tool(
            session,
            "jira_transition_issue",
            {"issue_key": issue_key, "transition_name": settings.jira_triage_status},
        )
        return True, await _verified_status(session, issue_key)
    except Exception as exc:
        logger.warning("failed to transition Jira issue %s to %s: %s", issue_key, settings.jira_triage_status, exc)
        return True, f"transition_failed:{settings.jira_triage_status}"


async def _verified_status(session: ClientSession, issue_key: str) -> str | None:
    try:
        issue = await _call_tool(session, "jira_get_issue", {"issue_key": issue_key, "fields": "status"})
        status_name = _extract_status_name(_extract_text_content(issue))
        if status_name:
            return status_name
    except Exception as exc:
        logger.warning("failed to verify Jira issue %s status: %s", issue_key, exc)
    return settings.jira_triage_status


def _extract_status_name(value: Any) -> str | None:
    if isinstance(value, str):
        try:
            return _extract_status_name(json.loads(value))
        except json.JSONDecodeError:
            match = re.search(r'"name"\s*:\s*"([^"]+)"', value)
            return match.group(1) if match else None
    if isinstance(value, dict):
        if "result" in value:
            found = _extract_status_name(value["result"])
            if found:
                return found
        status = value.get("status")
        if isinstance(status, dict) and status.get("name"):
            return str(status["name"])
        for item in value.values():
            found = _extract_status_name(item)
            if found:
                return found
    if isinstance(value, list):
        for item in value:
            found = _extract_status_name(item)
            if found:
                return found
    return None


def _find_transition_id(payload: Any, desired_status: str) -> str | None:
    desired = desired_status.lower()

    def visit(value: Any) -> str | None:
        if isinstance(value, str):
            try:
                return visit(json.loads(value))
            except json.JSONDecodeError:
                return None
        if isinstance(value, dict):
            name = str(value.get("name") or value.get("to") or value.get("status") or "").lower()
            if desired in name and value.get("id"):
                return str(value["id"])
            for item in value.values():
                found = visit(item)
                if found:
                    return found
        elif isinstance(value, list):
            for item in value:
                found = visit(item)
                if found:
                    return found
        return None

    return visit(payload)


def _disabled_result(reason_code: str | None, action: str, error: str | None = None) -> JiraTrackingResult:
    return {
        "enabled": bool(settings.enable_jira_mcp),
        "attempted": False,
        "should_create": False,
        "action": action,
        "reason_code": reason_code,
        "project_key": settings.jira_project_key,
        "issue_type": None,
        "priority": None,
        "issue_key": None,
        "issue_url": None,
        "duplicate_found": False,
        "dedupe_jql": None,
        "comment_added": False,
        "triage_transition_attempted": False,
        "triage_transition_status": None,
        "status": "skipped" if not error else "failed",
        "error": error,
        "metadata": {"timestamp": _utc_now()},
    }


def _rest_base_url() -> str:
    return str(settings.jira_url or "").rstrip("/")


async def _rest_request(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    async with httpx.AsyncClient(
        base_url=_rest_base_url(),
        auth=(str(settings.jira_username or ""), str(settings.jira_api_token or "")),
        timeout=settings.jira_mcp_timeout_seconds,
    ) as client:
        try:
            response = await client.request(method, path, params=params, json=json_payload)
            response.raise_for_status()
            logger.info(
                "jira_rest_call_succeeded",
                extra={
                    "method": method,
                    "path": path,
                    "status_code": response.status_code,
                    "duration_ms": int((time.perf_counter() - started) * 1000),
                },
            )
            if not response.content:
                return {}
            return response.json()
        except Exception:
            logger.error(
                "jira_rest_call_failed",
                extra={"method": method, "path": path, "duration_ms": int((time.perf_counter() - started) * 1000)},
                exc_info=True,
            )
            raise


async def _rest_search_existing(mapping: JiraIssueMapping) -> tuple[str | None, str]:
    jql = _dedupe_jql(mapping)
    payload = await _rest_request(
        "GET",
        "/rest/api/3/search/jql",
        params={
            "jql": jql,
            "fields": "key,summary,status,priority,labels,updated",
            "maxResults": 5,
        },
    )
    issues = list(payload.get("issues") or [])
    for issue in issues:
        key = str(dict(issue).get("key") or "")
        if key.startswith(f"{mapping.project_key}-"):
            return key, jql
    return None, jql


async def _rest_add_comment(issue_key: str, body: str) -> None:
    await _rest_request("POST", f"/rest/api/2/issue/{issue_key}/comment", json_payload={"body": body})


async def _rest_create_issue(state: SupportOrchestrationState, mapping: JiraIssueMapping) -> str:
    fields: dict[str, Any] = {
        "project": {"key": mapping.project_key},
        "summary": _summary(state, mapping),
        "issuetype": {"name": mapping.issue_type},
        "description": _description(state, mapping),
        "priority": {"name": mapping.priority},
        "labels": mapping.labels,
    }
    if settings.jira_component_name:
        fields["components"] = [{"name": settings.jira_component_name}]
    payload = await _rest_request("POST", "/rest/api/2/issue", json_payload={"fields": fields})
    issue_key = str(payload.get("key") or "")
    if not issue_key:
        raise RuntimeError(f"Jira REST issue creation completed but no key was returned: {_safe_text(payload, 500)}")
    return issue_key


async def _rest_transition_to_triage(issue_key: str) -> tuple[bool, str | None]:
    if not settings.jira_triage_status:
        return False, None

    transition_id = settings.jira_triage_transition_id
    if not transition_id:
        transitions = await _rest_request("GET", f"/rest/api/2/issue/{issue_key}/transitions")
        transition_id = _find_transition_id(transitions, settings.jira_triage_status)

    if not transition_id:
        return True, f"transition_not_found:{settings.jira_triage_status}"

    await _rest_request(
        "POST",
        f"/rest/api/2/issue/{issue_key}/transitions",
        json_payload={"transition": {"id": transition_id}},
    )
    issue = await _rest_request("GET", f"/rest/api/2/issue/{issue_key}", params={"fields": "status"})
    return True, _extract_status_name(issue) or settings.jira_triage_status


async def _track_escalation_via_rest(
    state: SupportOrchestrationState,
    escalation_package: dict[str, Any],
    mapping: JiraIssueMapping,
    existing_result: JiraTrackingResult,
) -> JiraTrackingResult:
    result: JiraTrackingResult = dict(existing_result)
    result["metadata"] = {
        **dict(result.get("metadata") or {}),
        "fallback": "jira_rest",
        "escalation_package_keys": sorted(escalation_package.keys()),
    }
    existing_key, jql = await _rest_search_existing(mapping)
    result["dedupe_jql"] = jql
    if existing_key:
        await _rest_add_comment(existing_key, _comment_body(state, duplicate=True))
        result.update(
            {
                "action": "linked_existing_rest_fallback",
                "issue_key": existing_key,
                "issue_url": _issue_url(existing_key),
                "duplicate_found": True,
                "comment_added": True,
                "status": "linked",
                "error": None,
            }
        )
        return result

    issue_key = await _rest_create_issue(state, mapping)
    await _rest_add_comment(issue_key, _comment_body(state, duplicate=False))
    triage_attempted, triage_status = await _rest_transition_to_triage(issue_key)
    result.update(
        {
            "action": "created_rest_fallback",
            "issue_key": issue_key,
            "issue_url": _issue_url(issue_key),
            "duplicate_found": False,
            "comment_added": True,
            "triage_transition_attempted": triage_attempted,
            "triage_transition_status": triage_status,
            "status": "created",
            "error": None,
        }
    )
    return result


@observe(name="jira_mcp_tool", as_type="tool", capture_input=False, capture_output=False)
async def track_escalation_in_jira(
    state: SupportOrchestrationState,
    escalation_package: dict[str, Any],
) -> JiraTrackingResult:
    """Create or link a Jira engineering ticket through mcp-atlassian."""

    started = time.perf_counter()
    should_create, reason_code = should_create_jira_issue(state)
    if not should_create:
        result = _disabled_result(reason_code, "not_engineering_bound")
        trace_tool_result(tool_name="jira_mcp_tool", question=str(state.get("query") or ""), result=result, started_at=started)
        return result
    if not settings.enable_jira_mcp:
        result = _disabled_result(reason_code, "disabled")
        trace_tool_result(tool_name="jira_mcp_tool", question=str(state.get("query") or ""), result=result, started_at=started)
        return result

    try:
        settings.validate_for_jira_mcp()
    except ConfigurationError as exc:
        result = _disabled_result(reason_code, "configuration_error", str(exc))
        trace_tool_result(tool_name="jira_mcp_tool", question=str(state.get("query") or ""), result=result, started_at=started)
        return result

    mapping = _issue_mapping(state, reason_code)
    result: JiraTrackingResult = {
        "enabled": True,
        "attempted": True,
        "should_create": True,
        "action": "create_or_link",
        "reason_code": reason_code,
        "project_key": mapping.project_key,
        "issue_type": mapping.issue_type,
        "priority": mapping.priority,
        "issue_key": None,
        "issue_url": None,
        "duplicate_found": False,
        "dedupe_jql": None,
        "comment_added": False,
        "triage_transition_attempted": False,
        "triage_transition_status": None,
        "status": "started",
        "error": None,
        "metadata": {
            "labels": mapping.labels,
            "dedupe_key": mapping.dedupe_key,
            "dedupe_label": mapping.dedupe_label,
            "dedupe_components": _dedupe_components(state, reason_code),
            "ticket_id": _ticket_id(state),
            "timestamp": _utc_now(),
            "escalation_package_keys": sorted(escalation_package.keys()),
        },
    }

    try:
        async with _jira_mcp_session() as session:
            existing_key, jql = await _search_existing(session, mapping)
            result["dedupe_jql"] = jql
            if existing_key:
                await _add_comment(session, existing_key, _comment_body(state, duplicate=True))
                result.update(
                    {
                        "action": "linked_existing",
                        "issue_key": existing_key,
                        "issue_url": _issue_url(existing_key),
                        "duplicate_found": True,
                        "comment_added": True,
                        "status": "linked",
                    }
                )
                trace_tool_result(tool_name="jira_mcp_tool", question=str(state.get("query") or ""), result=result, started_at=started)
                return result

            issue_key = await _create_issue(session, state, mapping)
            await _add_comment(session, issue_key, _comment_body(state, duplicate=False))
            triage_attempted, triage_status = await _transition_to_triage(session, issue_key)
            result.update(
                {
                    "action": "created",
                    "issue_key": issue_key,
                    "issue_url": _issue_url(issue_key),
                    "comment_added": True,
                    "triage_transition_attempted": triage_attempted,
                    "triage_transition_status": triage_status,
                    "status": "created",
                }
            )
    except Exception as exc:
        mcp_error = _root_error_message(exc)
        logger.warning("Jira MCP tracking failed; trying Jira REST fallback: %s", mcp_error)
        try:
            result = await _track_escalation_via_rest(state, escalation_package, mapping, result)
            result["metadata"] = {
                **dict(result.get("metadata") or {}),
                "mcp_error": mcp_error,
            }
        except Exception as fallback_exc:
            fallback_error = _root_error_message(fallback_exc)
            logger.exception("Jira REST fallback failed")
            result["status"] = "failed"
            result["error"] = f"MCP failed: {mcp_error}; REST fallback failed: {fallback_error}"

    trace_tool_result(tool_name="jira_mcp_tool", question=str(state.get("query") or ""), result=result, started_at=started)
    return result
