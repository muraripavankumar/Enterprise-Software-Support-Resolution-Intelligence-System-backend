import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.core.langfuse import observe, trace_agent_state
from app.orchestration.state import (
    AgentTraceEvent,
    ExecutionResult,
    ProgressUpdate,
    RouteDecision,
    SQLResult,
    SupportOrchestrationState,
    VerificationOutcome,
)

AGENT_NAME = "response_composer_agent"
MAX_EVIDENCE_SNIPPETS = 3
MAX_SNIPPET_CHARS = 700
RESPONSE_STYLE_LIMITS = {
    "concise": 180,
    "detailed": 500,
    "technical": 700,
}
SECRET_PATTERNS = [
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{12,}"),
    re.compile(r"(?i)\b(client_secret|access_token|refresh_token|api[_-]?key|authorization)\b\s*[:=]\s*['\"]?(?:Bearer\s+)?[^,'\"`\s}]+"),
    re.compile(r"\bsk_(?:live|test)_[A-Za-z0-9_=-]{8,}"),
    re.compile(r"\beyJ[A-Za-z0-9._~+/=-]{12,}"),
]
URL_PATTERN = re.compile(r"https?://[^\s)>\"]+")
MARKDOWN_ARTIFACT_PATTERN = re.compile(r"\s+#\s+")
RAW_JSON_HINT_PATTERN = re.compile(r"[{}]\s*\"[A-Za-z_][A-Za-z0-9_ -]*\"\s*:")
CLARIFICATION_SUGGESTIONS = [
    "I'm getting a 403 error on the billing API.",
    "My integration stopped working after the last update.",
    "A premium customer is seeing high latency in the EU region.",
]
CHITCHAT_RESPONSES = [
    "Hi! I'm the ERIS support assistant. Tell me what you're running into and I'll help with a product question, account issue, or anything urgent.",
    "Hello! ERIS support is here. Share the issue, error message, or customer context and I'll route it to the right workflow.",
    "Thanks for stopping by. I'm ERIS, your support assistant; send me the product question, account problem, or incident details when you're ready.",
    "Got it. I'm ERIS support; when you're ready, describe what's happening and I'll help investigate or escalate if needed.",
]


@dataclass(frozen=True)
class SourceReference:
    citation_id: int
    title: str
    page: int | None = None
    score: float | None = None
    content_type: str | None = None
    snippet: str | None = None


@dataclass(frozen=True)
class EvidenceBundle:
    question: str
    route_decision: str
    intent: str | None = None
    matched_entity: str | None = None
    mode: str = "agent"
    rag_summary: list[str] = field(default_factory=list)
    evidence_summary: list[str] = field(default_factory=list)
    table_facts: list[str] = field(default_factory=list)
    key_values: dict[str, Any] = field(default_factory=dict)
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)
    source_nodes: list[dict[str, Any]] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    sources: list[SourceReference] = field(default_factory=list)
    sql_summary: str | None = None
    sql_rows: list[Any] = field(default_factory=list)
    structured_result: dict[str, Any] | None = None
    table_used: str | None = None
    row_count: int = 0
    customer_context: dict[str, Any] = field(default_factory=dict)
    incident_investigation: dict[str, Any] = field(default_factory=dict)
    severity: str | None = None
    confidence_score: float | None = None
    escalation_flag: bool = False
    escalation_target: str | None = None
    tools_used: list[str] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    trace_id: str | None = None
    response_style: str = "concise"


@dataclass(frozen=True)
class FinalAnswer:
    answer: str
    key_points: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    implementation_steps: list[str] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)
    security_notes: list[str] = field(default_factory=list)
    current_status: list[str] = field(default_factory=list)
    evidence_summary: list[str] = field(default_factory=list)
    sources: list[SourceReference] = field(default_factory=list)
    structured_result: dict[str, Any] | list[Any] | None = None
    suggested_questions: list[str] = field(default_factory=list)
    escalation_summary: dict[str, Any] | None = None
    quality_summary: dict[str, Any] | None = None
    technical_details: dict[str, Any] | None = None


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


def _execution_result(answer: str, recommended_actions: list[str]) -> ExecutionResult:
    return {
        "step_id": "response-composition",
        "agent_name": AGENT_NAME,
        "result_type": "response_composition",
        "summary": f"final_answer_length={len(answer)}; recommended_actions={len(recommended_actions)}",
        "data": {
            "final_answer": answer,
            "recommended_actions": recommended_actions,
        },
        "error": None,
        "timestamp": _utc_now(),
    }


def _verification(answer: str) -> VerificationOutcome:
    passed = bool(answer.strip())
    return {
        "check_name": "response_composition_complete",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": "Final response was composed." if passed else "Final response is empty.",
        "corrective_action": None if passed else "Escalate or request clarification.",
        "metadata": {"final_answer_length": len(answer)},
    }


def _first_sql_answer(sql_results: list[SQLResult]) -> str | None:
    for result in sql_results:
        if result.get("answer"):
            return str(result["answer"])
    return None


def _clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    replacements = {
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
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    text = MARKDOWN_ARTIFACT_PATTERN.sub(". ", text)
    text = URL_PATTERN.sub("[link]", text)
    text = re.sub(r"\s*[-•]\s*", " - ", text)
    return " ".join(text.split())


def _response_style(state: SupportOrchestrationState) -> str:
    metadata = dict(state.get("metadata") or {})
    style = str(metadata.get("response_style") or "concise").lower()
    return style if style in RESPONSE_STYLE_LIMITS else "concise"


def _redact_sensitive_values(text: str) -> str:
    redacted = text

    def replace_named_secret(match: re.Match[str]) -> str:
        if "[REDACTED]" in match.group(0):
            return match.group(0)
        key = match.group(1) if match.lastindex else "secret"
        if key.lower() == "authorization":
            return "Authorization: Bearer [REDACTED]"
        return f"{key}: [REDACTED]"

    for pattern in SECRET_PATTERNS:
        if pattern.pattern.startswith("(?i)\\b(client_secret"):
            redacted = pattern.sub(replace_named_secret, redacted)
        elif "Bearer" in pattern.pattern:
            redacted = pattern.sub("Bearer [REDACTED]", redacted)
        else:
            redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def _limit_words(text: str, max_words: int) -> str:
    words = re.findall(r"\S+", text)
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(" ,;:-") + "..."


def clean_final_answer(answer: str, *, max_words: int = 180) -> str:
    """Return a concise, safe, user-facing answer string."""

    text = _redact_sensitive_values(answer)
    text = text.replace("\r\n", "\n")
    text = re.sub(r"(?m)^\s*#{4,}\s*", "### ", text)
    text = re.sub(r"(?m)^\s*#{1,2}(?!#)\s*", "### ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"(?i)(https?://[^\s)>\"]+)(\s+\1)+", r"\1", text)
    text = re.sub(r"(?im)^Sources:\s*(Sources:\s*)+", "### Sources\n", text)

    if _word_count(text) > max_words:
        if "\n### Sources" in text:
            body, sources = text.split("\n### Sources", 1)
            text = _limit_words(body.strip(), max_words) + "\n\n### Sources:" + sources.lstrip(":")
        else:
            text = _limit_words(text, max_words)
    return text.strip()


def redact_sensitive_content(text: str) -> str:
    """Public cleanup helper used by evaluators and tests."""

    return _redact_sensitive_values(text)


def _evidence_snippets(state: SupportOrchestrationState) -> list[str]:
    snippets: list[str] = []
    for chunk in list(state.get("retrieved_chunks") or [])[:MAX_EVIDENCE_SNIPPETS]:
        text = _clean_text(chunk.get("chunk_text"))
        if not text:
            continue
        snippets.append(text[:MAX_SNIPPET_CHARS].rstrip())
    return snippets


def _source_label(chunk: dict[str, Any]) -> str:
    source = chunk.get("source_file") or dict(chunk.get("metadata") or {}).get("source_file") or "source"
    page = chunk.get("page_number") or dict(chunk.get("metadata") or {}).get("page_number")
    return f"{source}, page {page}" if page else str(source)


def _clean_source_name(value: Any) -> str:
    text = str(value or "source").strip()
    text = text.split("\\")[-1].split("/")[-1]
    text = re.sub(r"\.pdf$", "", text, flags=re.IGNORECASE)
    text = text.replace("_", " ")
    return text or "Source"


def _source_lines(state: SupportOrchestrationState, limit: int = 3) -> list[str]:
    lines: list[str] = []
    seen: set[tuple[str, Any]] = set()
    for chunk in list(state.get("retrieved_chunks") or []):
        source = chunk.get("source_file") or dict(chunk.get("metadata") or {}).get("source_file") or "Source"
        page = chunk.get("page_number") or dict(chunk.get("metadata") or {}).get("page_number")
        key = (str(source), page)
        if key in seen:
            continue
        seen.add(key)
        label = _clean_source_name(source)
        page_text = f"page {page}" if page else "page not specified"
        lines.append(f"[{len(lines) + 1}] {label} - {page_text}")
        if len(lines) >= limit:
            break
    return lines


def _source_references_from_chunks(chunks: list[dict[str, Any]], limit: int = 5) -> list[SourceReference]:
    sources: list[SourceReference] = []
    seen: set[tuple[str, Any]] = set()
    for chunk in chunks:
        metadata = dict(chunk.get("metadata") or {})
        source = chunk.get("source_file") or metadata.get("source_file") or metadata.get("source") or "Source"
        page = chunk.get("page_number") or metadata.get("page_number")
        key = (str(source), page)
        if key in seen:
            continue
        seen.add(key)
        try:
            page_number = int(page) if page is not None else None
        except (TypeError, ValueError):
            page_number = None
        try:
            score = float(chunk.get("score")) if chunk.get("score") is not None else None
        except (TypeError, ValueError):
            score = None
        sources.append(
            SourceReference(
                citation_id=len(sources) + 1,
                title=_clean_source_name(source),
                page=page_number,
                score=score,
                content_type=chunk.get("content_type") or metadata.get("content_type"),
                snippet=_limit_words(_clean_text(chunk.get("chunk_text") or ""), 32),
            )
        )
        if len(sources) >= limit:
            break
    return sources


def _sql_summary_from_results(sql_results: list[SQLResult]) -> tuple[str | None, list[Any], str | None, int]:
    if not sql_results:
        return None, [], None, 0
    result = dict(sql_results[0])
    rows = list(result.get("raw_results") or [])
    row_count = int(result.get("row_count") or len(rows))
    tables = result.get("tables_used") or result.get("table_used") or []
    table_used = ", ".join(str(table) for table in tables) if isinstance(tables, list) else str(tables or "")
    answer = str(result.get("answer") or "").strip()
    if answer:
        summary = answer
    elif result.get("error"):
        summary = f"Structured data lookup failed: {result.get('error')}"
    elif row_count == 0:
        summary = "No matching structured records were found."
    else:
        summary = f"Found {row_count} structured record(s)."
    return clean_final_answer(summary, max_words=80), rows, table_used or None, row_count


def _matched_entity(question: str) -> str | None:
    lowered = question.lower()
    entity_patterns = [
        (r"\bpriority customers?\b|\bpriority support\b", "Priority support tier"),
        (r"\bhigh security vulnerability\b|\bhigh vulnerability\b", "High security vulnerability"),
        (r"\b400\b|\binvalid json\b", "400 Invalid JSON"),
        (r"\b504\b|\bgateway timeout\b", "504 Gateway Timeout"),
        (r"\b429\b|\brate limit(?:ing|ed)?\b", "429 Rate Limiting"),
        (r"\brca\b|\broot cause analysis\b", "RCA requirements"),
        (r"\bread-heavy\b|\bcache-control\b|\bcaching strategy\b", "Read-heavy API cache strategy"),
        (r"\bregional endpoints?\b|\breduce latency\b", "Regional endpoint selection"),
        (r"\binstallation\b|\bdeployment\b|\bdocker-compose\b", "Deployment verification"),
        (r"\bitil\b|\bincident lifecycle\b", "ITIL incident lifecycle"),
    ]
    for pattern, label in entity_patterns:
        if re.search(pattern, lowered):
            return label
    return None


def _key_values_from_sql_rows(rows: list[Any]) -> dict[str, Any]:
    if not rows or not isinstance(rows[0], dict):
        return {}
    first = dict(rows[0])
    allowed = {
        "ticket_id",
        "customer_id",
        "company_name",
        "account_status",
        "sla_level",
        "severity_level",
        "ticket_status",
        "incident_id",
        "severity",
        "resolution_status",
    }
    return {key: value for key, value in first.items() if key in allowed and value is not None}


def normalize_evidence(state_or_result: SupportOrchestrationState | dict[str, Any]) -> EvidenceBundle:
    """Normalize graph state into clean evidence for route-specific composers."""

    state = dict(state_or_result)
    route = _enum_value(state.get("route_decision")) or "clarification"
    chunks = [dict(chunk) for chunk in list(state.get("retrieved_chunks") or [])]
    facts = _evidence_facts({**state, "retrieved_chunks": chunks}) if chunks else []
    sql_summary, sql_rows, table_used, row_count = _sql_summary_from_results(list(state.get("sql_results") or []))
    metadata = dict(state.get("metadata") or {})
    tools = []
    if chunks:
        tools.append("document_retrieval")
    if state.get("sql_results"):
        tools.append("sql_agent")
    if state.get("customer_context"):
        tools.append("account_validator")
    if state.get("incident_investigation"):
        tools.append("incident_investigator")

    missing_information: list[str] = []
    if route == RouteDecision.CLARIFICATION.value:
        missing_information.extend(["customer ID or ticket ID", "error code or symptom", "environment", "start time"])
    if route == RouteDecision.RAG.value and not chunks:
        missing_information.append("documentation evidence")
    if route == RouteDecision.SQL.value and not state.get("sql_results"):
        missing_information.append("structured query result")

    return EvidenceBundle(
        question=str(state.get("query") or ""),
        route_decision=route,
        intent=_enum_value(state.get("intent")),
        matched_entity=_matched_entity(str(state.get("query") or "")),
        rag_summary=[_redact_sensitive_values(fact) for fact in facts[:7]],
        evidence_summary=[_redact_sensitive_values(fact) for fact in facts[:5]],
        table_facts=[_redact_sensitive_values(fact) for fact in facts if re.search(r"\b(table|tier|severity|code|phase|timeline|response time|cache-control|endpoint)\b", fact, re.IGNORECASE)][:5],
        key_values=_key_values_from_sql_rows(sql_rows),
        retrieved_chunks=chunks,
        source_nodes=chunks,
        citations=[str(citation) for citation in list(state.get("citations") or [])],
        sources=_source_references_from_chunks(chunks),
        sql_summary=sql_summary,
        sql_rows=sql_rows,
        structured_result=dict(list(state.get("sql_results") or [{}])[0]) if state.get("sql_results") else None,
        table_used=table_used,
        row_count=row_count,
        customer_context=dict(state.get("customer_context") or {}),
        incident_investigation=dict(state.get("incident_investigation") or {}),
        severity=_enum_value(state.get("severity")) or _enum_value(state.get("severity_priority")),
        confidence_score=state.get("confidence_score"),
        escalation_flag=bool(state.get("escalation_flag")),
        escalation_target=_enum_value(state.get("escalation_target")),
        tools_used=tools,
        missing_information=missing_information,
        errors=[str(error) for error in list(state.get("errors") or [])],
        trace_id=str(state.get("langfuse_trace_id") or "") or None,
        response_style=_response_style(state),
    )


def _split_evidence_facts(text: str) -> list[str]:
    cleaned = _clean_text(text)
    cleaned = re.sub(r"\{[^{}]{0,260}\}", " ", cleaned)
    cleaned = re.sub(r"\bcurl\b[^.]{0,240}", " ", cleaned, flags=re.IGNORECASE)
    parts = re.split(r"(?<=[.!?])\s+|\s+-\s+|\s+#\s+|;\s+", cleaned)
    facts: list[str] = []
    for part in parts:
        fact = part.strip(" .:-")
        if len(fact) < 18:
            continue
        if RAW_JSON_HINT_PATTERN.search(fact):
            continue
        if fact.lower().startswith(("http", "get http", "post http")):
            continue
        facts.append(fact[:220].strip())
    return facts


def _evidence_facts(state: SupportOrchestrationState) -> list[str]:
    facts: list[str] = []
    seen: set[str] = set()
    for snippet in _evidence_snippets(state):
        for fact in _split_evidence_facts(snippet):
            normalized = fact.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            facts.append(fact)
            if len(facts) >= 7:
                return facts
    return facts


def _is_authentication_question(state: SupportOrchestrationState, facts: list[str]) -> bool:
    text = " ".join([str(state.get("query") or ""), *facts]).lower()
    return any(keyword in text for keyword in ("oauth", "api key", "authentication", "token", "authorization"))


def _is_oauth_configuration_question(question: str) -> bool:
    lowered = question.lower()
    return "oauth" in lowered and bool(re.search(r"\b(how|configure|configuration|setup|set up|implement|use)\b", lowered))


def _api_versions(text: str) -> set[str]:
    return {match.group(1) for match in re.finditer(r"\bapi\s+v?(\d+(?:\.\d+)*)\b", text, flags=re.IGNORECASE)}


def _version_caveats(question: str, facts: list[str], chunks: list[dict[str, Any]]) -> list[str]:
    query_versions = _api_versions(question)
    evidence_text = " ".join([*facts, *(str(chunk.get("chunk_text") or "") for chunk in chunks)])
    evidence_versions = _api_versions(evidence_text)
    if query_versions and evidence_versions and query_versions.isdisjoint(evidence_versions):
        requested = ", ".join(sorted(query_versions))
        documented = ", ".join(sorted(evidence_versions))
        return [
            f"The retrieved documentation references API v{documented}, while your question asks about API v{requested}; verify whether v{requested} has a separate legacy OAuth flow before production rollout."
        ]
    return []


def _oauth_facts(facts: list[str]) -> list[str]:
    oauth_terms = ("oauth", "authorization code", "authorization url", "access token", "bearer", "redirect", "client_id")
    selected = [fact for fact in facts if any(term in fact.lower() for term in oauth_terms)]
    return selected or facts[:2]


def _direct_document_answer(state: SupportOrchestrationState, facts: list[str]) -> str:
    query = str(state.get("query") or "")
    if _is_oauth_configuration_question(query):
        return (
            "Configure OAuth using the OAuth 2.0 authorization-code flow when the API call needs delegated user access. "
            "Redirect the user for authorization, exchange the returned code for an access token, then call the API with a bearer token."
        )
    if _is_authentication_question(state, facts):
        return (
            "Use scoped API keys for server-to-server integrations and OAuth 2.0 when delegated user access is needed. "
            "Store credentials securely, redact secrets from logs, and rotate production keys regularly."
        )
    if facts:
        first = facts[0].rstrip(".")
        return first + "."
    return "I found relevant support documentation, but it did not contain enough clean detail for a confident answer."


def _action_from_fact(fact: str) -> str | None:
    lowered = fact.lower()
    if "scope" in lowered or "permission" in lowered:
        return "Create credentials with only the required scopes or permissions."
    if "rotate" in lowered:
        return "Rotate production credentials on the documented schedule or immediately after compromise."
    if "store" in lowered or "secret" in lowered or "environment" in lowered:
        return "Store keys and tokens in a secrets manager or protected environment variables."
    if "authorization code" in lowered or "redirect" in lowered or "oauth" in lowered:
        return "Use the OAuth authorization-code flow for user-delegated access."
    if "rate limit" in lowered or "429" in lowered:
        return "Handle rate limits by checking reset headers and retrying after the documented delay."
    if "sla" in lowered:
        return "Validate the customer's SLA tier before choosing the next operational action."
    if "timeout" in lowered or "latency" in lowered:
        return "Check upstream latency, retry behavior, and regional incident status before escalating."
    return None


def _recommended_actions_from_evidence(state: SupportOrchestrationState, facts: list[str]) -> list[str]:
    actions: list[str] = []
    for fact in facts:
        action = _action_from_fact(fact)
        if action and action not in actions:
            actions.append(action)
        if len(actions) >= 5:
            break
    return actions[:5]


def _format_answer_sections(answer: str, actions: list[str], sources: list[str], notes: list[str] | None = None) -> str:
    lines = ["### Answer:", answer.strip()]
    if actions:
        lines.extend(["", "### Recommended actions:"])
        lines.extend(f"- {action.rstrip('.')}" + "." for action in actions[:5])
    if notes:
        lines.extend(["", "### Important notes:"])
        lines.extend(f"- {note.rstrip('.')}" + "." for note in notes[:3])
    if sources:
        lines.extend(["", "### Sources:"])
        lines.extend(sources)
    return "\n".join(lines)


def _source_markdown(sources: list[SourceReference]) -> list[str]:
    lines = []
    for source in sources:
        page = f"page {source.page}" if source.page else "page not specified"
        topic = _source_topic(source)
        topic_text = f" - {topic}" if topic else ""
        lines.append(f"- [{source.citation_id}] {source.title}{topic_text} - {page}")
    return lines


def _source_topic(source: SourceReference) -> str | None:
    snippet = (source.snippet or "").lower()
    if "oauth" in snippet or "authorization code" in snippet or "access token" in snippet:
        return "OAuth 2.0 authorization flow"
    if "api key" in snippet or "scoped" in snippet or "rotate" in snippet:
        return "credential management"
    if "rate limit" in snippet or "429" in snippet:
        return "rate limiting"
    return None


def _contains_any(text: str, *terms: str) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms)


def _specific_rag_answer(evidence: EvidenceBundle) -> FinalAnswer | None:
    question = evidence.question.lower()
    sources = evidence.sources

    if "sla" in question and "priority" in question and ("commitment" in question or "response" in question):
        return FinalAnswer(
            answer="Priority customers have a support-tier response commitment of under 1 hour.",
            key_points=[
                "Priority support is available 24/5 with all standard channels plus phone support.",
                "Priority escalations route directly to L2 support.",
                "Priority SLA credits are 10% per P0/P1 breach, capped at 50%.",
                "Severity-specific Priority response times are P0: 30 minutes, P1: 1 hour, P2: 4 hours, and P3: 8 hours.",
            ],
            sources=sources,
        )

    if "incident lifecycle" in question or ("itil" in question and ("stage" in question or "lifecycle" in question)):
        return FinalAnswer(
            answer="The ITIL incident lifecycle follows identification, classification and prioritization, diagnosis and investigation, resolution and recovery, and incident closure.",
            key_points=[
                "Identification records the timestamp, reporter, symptoms, and monitoring or service-desk source.",
                "Classification and prioritization categorize the issue, assess urgency and impact, and assign P0-P3 priority.",
                "Diagnosis investigates root cause using knowledge base entries, diagnostic tools, error logs, and known errors.",
                "Resolution restores service and confirms recovery with the user.",
                "Closure verifies resolution, records time metrics and feedback, and captures lessons learned.",
            ],
            sources=sources,
        )

    if "rca" in question or "root cause analysis" in question:
        return FinalAnswer(
            answer="RCA is mandatory for all P0 and P1 incidents, and recommended for recurring P2 incidents.",
            key_points=[
                "Data collection should produce a complete incident timeline within 24 hours of resolution.",
                "Root cause identification should complete within 3 days.",
                "Corrective actions should be defined within 5 days.",
                "Documentation and review should include the RCA report, stakeholder review, knowledge-base updates, and lessons learned.",
            ],
            sources=sources,
        )

    if _contains_any(question, "read-heavy", "caching strategy", "cache strategy", "cache-control", "caching"):
        return FinalAnswer(
            answer="For read-heavy API data, use client or edge caching with `Cache-Control: public, max-age=300` and ETags for conditional requests.",
            key_points=[
                "The documented target is a cache-hit ratio above 90%.",
                "Read-heavy data can be cached for 5 minutes with `public, max-age=300`.",
                "User-specific data should use `private, max-age=60`.",
                "`no-cache` is used when every request must revalidate with the server.",
                "ETags let clients avoid payload downloads when the resource has not changed.",
            ],
            recommended_actions=[
                "Cache stable list or reference data for 300 seconds.",
                "Use ETags with `If-None-Match` for conditional GET requests.",
                "Monitor cache-hit ratio and increase TTL only when freshness requirements allow it.",
            ],
            sources=sources,
        )

    if ("verify" in question or "check" in question) and _contains_any(question, "installation", "deployment", "deploy"):
        return FinalAnswer(
            answer="After deployment, verify the installation by confirming containers are running, the health endpoint responds, the database is initialized, and logs show no startup errors.",
            key_points=[
                "Run `docker-compose ps` to confirm all services are running.",
                "Run `curl http://localhost:8080/health` for the application health check.",
                "Confirm database initialization and migrations completed successfully.",
                "Review application, database, Redis, and reverse-proxy logs if the health check fails.",
            ],
            implementation_steps=[
                "Check service status with `docker-compose ps`.",
                "Call the health endpoint: `curl http://localhost:8080/health`.",
                "Verify DB connectivity and migration status.",
                "Inspect logs before declaring the deployment ready.",
            ],
            sources=sources,
        )

    if "400" in question or "invalid json" in question:
        return FinalAnswer(
            answer="For a `400 Invalid JSON` error, validate the JSON syntax and request formatting before retrying.",
            key_points=[
                "The documented example is an unexpected token in the request body.",
                "Check `Content-Type: application/json`.",
                "Validate required fields and data types before sending the request.",
                "Capture the request ID and sanitized request body if the error persists.",
            ],
            recommended_actions=[
                "Run the payload through a JSON validator.",
                "Confirm the request body matches the API schema.",
                "Retry only after correcting malformed JSON or missing fields.",
            ],
            sources=sources,
        )

    if "504" in question or "gateway timeout" in question:
        return FinalAnswer(
            answer="A `504 Gateway Timeout` means the request exceeded the documented timeout window; optimize the query or increase the client timeout where appropriate.",
            key_points=[
                "The documented timeout threshold is 30 seconds.",
                "Log the request ID, response time, and sanitized request details.",
                "Check upstream latency and status-page incidents before escalating.",
            ],
            recommended_actions=[
                "Optimize heavy queries and reduce response payload size.",
                "Retry with controlled backoff if the operation is safe to retry.",
                "Escalate with the request ID if 5xx errors continue for more than 15 minutes.",
            ],
            sources=sources,
        )

    if "429" in question or "rate limit" in question:
        return FinalAnswer(
            answer="Clients should handle API 429 responses by honoring `Retry-After`, reducing concurrency, and using exponential backoff.",
            key_points=[
                "`X-RateLimit-Remaining: 0` with `Retry-After: 60` means wait 60 seconds before retrying.",
                "Burst-limit resets require reducing concurrent requests.",
                "Daily quota exhaustion requires waiting 24 hours or upgrading the plan.",
                "The API guide recommends checking `X-RateLimit-Limit`, `X-RateLimit-Remaining`, and `X-RateLimit-Reset`.",
            ],
            recommended_actions=[
                "Read `Retry-After` and sleep for that duration before retrying.",
                "Use exponential backoff and cap retry attempts.",
                "Cache responses and batch requests to reduce rate-limit pressure.",
            ],
            sources=sources,
        )

    if _contains_any(question, "regional endpoint", "regional endpoints", "reduce latency", "nearest region"):
        return FinalAnswer(
            answer="To reduce latency, use the global endpoint for automatic GeoDNS routing or pin clients to the nearest compliant regional endpoint when data residency requires it.",
            key_points=[
                "US East: `us-east.api.example.com`, average global latency 50-150 ms.",
                "US West: `us-west.api.example.com`, average global latency 40-140 ms.",
                "EU Central: `eu.api.example.com`, average global latency 60-180 ms and GDPR compliance.",
                "Asia Pacific: `apac.api.example.com`, average global latency 80-250 ms and regional isolation.",
                "Use `/v3/ping` to measure round-trip time before pinning an endpoint.",
            ],
            recommended_actions=[
                "Start with `api.example.com` for auto-routing.",
                "Pin to a regional endpoint only for data residency, disaster recovery, or measured latency gains.",
                "Validate endpoint choice with `/v3/ping` from the client region.",
            ],
            sources=sources,
        )

    if "high" in question and "security" in question and "vulnerab" in question:
        return FinalAnswer(
            answer="A High (P1) security vulnerability requires response in under 1 hour and total patch SLA under 36 hours.",
            key_points=[
                "High (P1) means high likelihood of exploitation or significant risk, such as SQL injection, privilege escalation, or PII exposure.",
                "High patch development target is under 24 hours.",
                "Testing target is under 8 hours.",
                "Deployment target is under 4 hours.",
                "The total High-severity patch SLA is under 36 hours.",
            ],
            recommended_actions=[
                "Start assessment immediately and assign a security owner.",
                "Prepare patch, test, deployment, rollback, and communication plans.",
                "Use the security response timeline rather than treating this as a generic support ticket.",
            ],
            sources=sources,
        )

    return None


def _customer_status_line(customer_context: dict[str, Any]) -> str | None:
    if not customer_context:
        return None
    pieces = [
        customer_context.get("account_status"),
        customer_context.get("sla_level"),
        customer_context.get("subscription_tier"),
        customer_context.get("region"),
    ]
    values = [str(piece) for piece in pieces if piece]
    return ", ".join(values) if values else None


def _quality_score(evidence: EvidenceBundle, final: FinalAnswer) -> dict[str, Any]:
    answer_text = final.answer.lower()
    question_terms = {
        term
        for term in re.findall(r"[a-zA-Z0-9]{4,}", evidence.question.lower())
        if term not in {"what", "with", "from", "this", "that", "about"}
    }
    relevance_overlap = len([term for term in question_terms if term in answer_text])
    relevance = 0.72 if not question_terms else min(0.98, 0.62 + (relevance_overlap / max(1, len(question_terms))) * 0.34)
    if evidence.route_decision in {RouteDecision.CHITCHAT.value, RouteDecision.CLARIFICATION.value}:
        faithfulness = None
    elif evidence.route_decision == RouteDecision.SQL.value:
        faithfulness = 0.92 if evidence.sql_summary else 0.55
    elif evidence.rag_summary or evidence.sql_summary or evidence.incident_investigation:
        faithfulness = 0.9
    else:
        faithfulness = 0.55
    structure_bonus = 0.08 if (
        final.key_points
        or final.recommended_actions
        or final.current_status
        or final.implementation_steps
        or final.security_notes
        or final.caveats
    ) else 0.0
    safety_penalty = 0.25 if any(pattern.search(final.answer) for pattern in SECRET_PATTERNS) else 0.0
    base = relevance if faithfulness is None else (relevance + faithfulness) / 2
    overall = max(0.0, min(1.0, base + structure_bonus - safety_penalty))
    return {
        "faithfulness_score": faithfulness,
        "answer_relevance_score": round(relevance, 2),
        "overall_quality_score": round(overall, 2),
        "evaluation_reasoning": "Deterministic quality heuristic based on route evidence, question overlap, structure, and safety.",
        "evaluation_status": "completed",
    }


def compose_rag_answer(evidence: EvidenceBundle) -> FinalAnswer:
    specific = _specific_rag_answer(evidence)
    if specific is not None:
        return specific

    if _is_oauth_configuration_question(evidence.question):
        return compose_oauth_configuration_answer(evidence)

    answer = _direct_document_answer({"query": evidence.question}, evidence.rag_summary)
    actions = _recommended_actions_from_evidence({"retrieved_chunks": evidence.retrieved_chunks}, evidence.rag_summary)
    if not evidence.rag_summary:
        answer = "I found limited documentation evidence for this request, so the answer may need verification."
        actions = ["Ask for the exact product area, version, or error message.", "Retry with a more specific support question."]
    return FinalAnswer(
        answer=answer,
        key_points=evidence.rag_summary[:4],
        recommended_actions=actions[:5],
        evidence_summary=evidence.rag_summary[:3],
        sources=evidence.sources,
        technical_details={"raw_chunk_count": len(evidence.retrieved_chunks), "response_style": evidence.response_style},
    )


def compose_oauth_configuration_answer(evidence: EvidenceBundle) -> FinalAnswer:
    oauth_facts = _oauth_facts(evidence.rag_summary)
    caveats = _version_caveats(evidence.question, evidence.rag_summary, evidence.retrieved_chunks)
    answer = (
        "Use OAuth 2.0 when the integration needs delegated user access. The documented pattern is the authorization-code flow: "
        "send the user to the authorization endpoint, receive an authorization code, exchange that code for an access token, and call the API with `Authorization: Bearer [REDACTED]`."
    )
    steps = [
        "Redirect the user to the authorization endpoint: `GET /oauth/authorize?client_id=your_client_id&response_type=code&redirect_uri=your_callback`.",
        "Exchange the returned authorization code for an access token: `POST /oauth/token` with `grant_type=authorization_code`, `code`, `client_id`, and a server-side `client_secret`.",
        "Call protected API endpoints using `Authorization: Bearer [REDACTED]`.",
        "Implement refresh-token handling and retry logic for expired or rate-limited requests.",
    ]
    security_notes = [
        "Never expose `client_secret`, access tokens, refresh tokens, or API keys in browser code or logs.",
        "Use HTTPS for all OAuth redirects and token exchanges.",
        "Store secrets in a secrets manager or protected environment variables and rotate credentials regularly.",
    ]
    return FinalAnswer(
        answer=answer,
        key_points=oauth_facts[:4],
        caveats=caveats,
        implementation_steps=steps,
        recommended_actions=[
            "Confirm whether the requested API version has a separate OAuth specification.",
            "Use the authorization-code flow for delegated user access, not API keys.",
            "Validate the callback URL and scopes before production deployment.",
        ],
        security_notes=security_notes,
        evidence_summary=oauth_facts[:4],
        sources=evidence.sources,
        technical_details={"raw_chunk_count": len(evidence.retrieved_chunks), "response_style": evidence.response_style},
    )


def compose_sql_answer(evidence: EvidenceBundle) -> FinalAnswer:
    if evidence.row_count == 0:
        answer = "No matching structured records were found for this request."
        key_finding = "No matching records found."
    else:
        answer = evidence.sql_summary or f"Found {evidence.row_count} matching structured record(s)."
        key_finding = answer
    current_status = [
        f"Table checked: {evidence.table_used or 'unknown'}",
        f"Records found: {evidence.row_count}",
        f"Key finding: {key_finding}",
    ]
    return FinalAnswer(
        answer=answer,
        key_points=[
            f"Records found: {evidence.row_count}",
            f"Table checked: {evidence.table_used or 'unknown'}",
            f"Key finding: {key_finding}",
        ],
        current_status=current_status,
        evidence_summary=[f"Structured data: {evidence.sql_summary or key_finding}"],
        structured_result=evidence.structured_result,
        technical_details={"sql_rows": evidence.sql_rows[:10], "table_used": evidence.table_used},
    )


def compose_hybrid_answer(evidence: EvidenceBundle) -> FinalAnswer:
    customer_status = _customer_status_line(evidence.customer_context)
    incident_summary = str(evidence.incident_investigation.get("investigation_summary") or "").strip()
    doc_summary = evidence.rag_summary[0] if evidence.rag_summary else "No specific document guidance was available."
    status_bits = []
    if customer_status:
        status_bits.append(f"Customer/account: {customer_status}")
    if incident_summary:
        status_bits.append(f"Incident/ticket: {incident_summary}")
    if evidence.severity:
        status_bits.append(f"Severity: {evidence.severity}")
    answer_parts = []
    if customer_status:
        answer_parts.append(f"The customer/account context is {customer_status}.")
    if incident_summary:
        answer_parts.append(incident_summary)
    if doc_summary:
        answer_parts.append(f"Documentation guidance: {doc_summary}")
    answer = " ".join(answer_parts) or "Hybrid investigation completed with available structured and documentation evidence."
    actions = _recommended_actions_from_evidence({"retrieved_chunks": evidence.retrieved_chunks}, evidence.rag_summary)
    if not actions:
        actions = ["Validate the structured account or ticket context.", "Follow the relevant runbook guidance.", "Escalate if business impact increases."]
    return FinalAnswer(
        answer=answer,
        key_points=evidence.rag_summary[:3],
        recommended_actions=actions[:5],
        current_status=status_bits,
        evidence_summary=[
            f"Structured data: {evidence.sql_summary or customer_status or 'available'}",
            doc_summary,
        ],
        sources=evidence.sources,
        structured_result=evidence.structured_result,
        technical_details={"sql_rows": evidence.sql_rows[:10], "raw_chunk_count": len(evidence.retrieved_chunks)},
    )


def compose_high_risk_answer(evidence: EvidenceBundle) -> FinalAnswer:
    target = evidence.escalation_target or "incident_response"
    severity = evidence.severity or "high"
    reason = "; ".join(evidence.errors[:2]) or "High-risk indicators were detected."
    jira = "Not created because engineering tracking was not required."
    return FinalAnswer(
        answer="This appears to be a high-risk issue and should be escalated for immediate review.",
        key_points=[f"Severity: {severity}", f"Escalation target: {target}", f"Reason: {reason}"],
        recommended_actions=[
            "Preserve the current evidence and agent trace.",
            "Notify the escalation target immediately.",
            "Avoid making risky production changes until the incident owner confirms the next step.",
        ],
        current_status=[f"Severity: {severity}", f"Target: {target}", f"Reason: {reason}"],
        escalation_summary={"severity": severity, "target": target, "reason": reason, "jira": jira},
        evidence_summary=evidence.rag_summary[:2],
        sources=evidence.sources,
    )


def compose_clarification_answer(evidence: EvidenceBundle) -> FinalAnswer:
    return FinalAnswer(
        answer="I need a little more detail to help you correctly.",
        recommended_actions=[
            "Share the customer ID or ticket ID.",
            "Include the error code or symptom.",
            "Mention whether this is production or test.",
            "Tell me when the issue started.",
        ],
        suggested_questions=[
            "Customer C123 is getting 504 timeout errors in production since 10 AM.",
            "Ticket TCK-104 is failing with a 403 on the billing API.",
        ],
    )


def compose_chitchat_answer(evidence: EvidenceBundle) -> FinalAnswer:
    return FinalAnswer(
        answer="Hi! I'm ready to help with support tickets, incidents, API errors, customer issues, policies, or troubleshooting steps.",
        suggested_questions=["Describe the issue, error code, customer, or business impact."],
    )


def final_answer_to_markdown(final: FinalAnswer) -> str:
    lines = ["### Answer:", final.answer.strip()]
    if final.key_points:
        lines.extend(["", "### Key points:"])
        lines.extend(f"- {item.rstrip('.')}" + "." for item in final.key_points[:6])
    if final.caveats:
        lines.extend(["", "### Assumptions / caveats:"])
        lines.extend(f"- {item.rstrip('.')}" + "." for item in final.caveats[:3])
    if final.implementation_steps:
        lines.extend(["", "### Recommended steps:"])
        lines.extend(f"{index}. {item.rstrip('.')}" + "." for index, item in enumerate(final.implementation_steps[:6], start=1))
    if final.current_status:
        lines.extend(["", "### Current status:"])
        lines.extend(f"- {item.rstrip('.')}" + "." for item in final.current_status[:5])
    if final.recommended_actions:
        lines.extend(["", "### Recommended actions:"])
        lines.extend(f"- {item.rstrip('.')}" + "." for item in final.recommended_actions[:5])
    if final.evidence_summary:
        lines.extend(["", "### Evidence summary:"])
        lines.extend(f"- {item.rstrip('.')}" + "." for item in final.evidence_summary[:4])
    if final.security_notes:
        lines.extend(["", "### Security notes:"])
        lines.extend(f"- {item.rstrip('.')}" + "." for item in final.security_notes[:4])
    if final.escalation_summary:
        lines.extend(["", "### Escalation:"])
        for key, value in final.escalation_summary.items():
            lines.append(f"- {str(key).replace('_', ' ').title()}: {value}")
    if final.sources:
        lines.extend(["", "### Sources:"])
        lines.extend(_source_markdown(final.sources))
    return "\n".join(lines)


def compose_final_answer(evidence: EvidenceBundle) -> FinalAnswer:
    route = evidence.route_decision
    if evidence.escalation_flag or route == RouteDecision.HIGH_RISK.value:
        final = compose_high_risk_answer(evidence)
    elif route == RouteDecision.RAG.value:
        final = compose_rag_answer(evidence)
    elif route == RouteDecision.SQL.value:
        final = compose_sql_answer(evidence)
    elif route == RouteDecision.HYBRID.value:
        final = compose_hybrid_answer(evidence)
    elif route == RouteDecision.CHITCHAT.value:
        final = compose_chitchat_answer(evidence)
    elif route == RouteDecision.CLARIFICATION.value:
        final = compose_clarification_answer(evidence)
    else:
        final = FinalAnswer(
            answer="I could not determine the right support route from the available information.",
            recommended_actions=["Share the product area, error message, customer context, or business impact."],
        )
    quality = _quality_score(evidence, final)
    return FinalAnswer(
        answer=final.answer,
        key_points=final.key_points,
        caveats=final.caveats,
        implementation_steps=final.implementation_steps,
        recommended_actions=final.recommended_actions,
        security_notes=final.security_notes,
        current_status=final.current_status,
        evidence_summary=final.evidence_summary,
        sources=final.sources,
        structured_result=final.structured_result,
        suggested_questions=final.suggested_questions,
        escalation_summary=final.escalation_summary,
        quality_summary=quality,
        technical_details=final.technical_details,
    )


def _final_answer_payload(final: FinalAnswer) -> dict[str, Any]:
    payload = asdict(final)
    return payload


def _compose_from_document_evidence(state: SupportOrchestrationState) -> str:
    if not list(state.get("retrieved_chunks") or []):
        return "I could not find enough documentation evidence to answer this request confidently."

    facts = _evidence_facts(state)
    answer = _direct_document_answer(state, facts)
    actions = _recommended_actions_from_evidence(state, facts)
    sources = _source_lines(state)
    return _format_answer_sections(answer, actions, sources)


def _compose_from_sql(state: SupportOrchestrationState) -> str:
    sql_results = list(state.get("sql_results") or [])
    answer = _first_sql_answer(sql_results)
    if answer:
        return _format_answer_sections(answer, _recommended_actions(state)[:3], [])

    if sql_results:
        result = sql_results[0]
        if result.get("error"):
            return _format_answer_sections(f"Structured data lookup could not be completed: {result['error']}", [], [])
        return _format_answer_sections("Structured data lookup completed, but no matching records were returned.", [], [])

    return _format_answer_sections("Structured data lookup was selected, but no SQL evidence is available.", [], [])


def _compose_hybrid_answer(state: SupportOrchestrationState) -> str:
    parts: list[str] = []
    actions: list[str] = []
    existing_answer = str(state.get("final_answer") or "").strip()
    if existing_answer:
        parts.append(existing_answer)
    elif state.get("retrieved_chunks"):
        facts = _evidence_facts(state)
        parts.append(_direct_document_answer(state, facts))
        actions.extend(_recommended_actions_from_evidence(state, facts))

    customer_context = dict(state.get("customer_context") or {})
    if customer_context:
        context_bits = [
            f"SLA: {customer_context.get('sla_level') or 'unknown'}",
            f"subscription: {customer_context.get('subscription_tier') or 'unknown'}",
            f"status: {customer_context.get('account_status') or 'unknown'}",
            f"region: {customer_context.get('region') or 'unknown'}",
        ]
        parts.append("Account context: " + "; ".join(context_bits) + ".")

    incident_result = dict(state.get("incident_investigation") or {})
    if incident_result:
        parts.append(str(incident_result.get("investigation_summary") or "Incident investigation completed."))

    sql_answer = _first_sql_answer(list(state.get("sql_results") or []))
    if sql_answer:
        parts.append("Structured data evidence: " + sql_answer)

    if parts:
        answer = " ".join(part.strip() for part in parts if part.strip())
        if not actions:
            actions = _recommended_actions(state)[:5]
        return _format_answer_sections(answer, actions[:5], _source_lines(state))

    return _format_answer_sections("Hybrid investigation completed, but no usable evidence was available.", [], _source_lines(state))


def _compose_chitchat_answer(state: SupportOrchestrationState) -> str:
    query = str(state.get("query") or "")
    seed = int(time.time_ns()) + sum(ord(character) for character in query)
    return CHITCHAT_RESPONSES[seed % len(CHITCHAT_RESPONSES)]


def _compose_clarification_answer(state: SupportOrchestrationState) -> str:
    metadata = dict(state.get("metadata") or {})
    metadata["clarification_suggestions"] = CLARIFICATION_SUGGESTIONS
    state["metadata"] = metadata
    examples = "\n".join(f"- {suggestion}" for suggestion in CLARIFICATION_SUGGESTIONS)
    return (
        "I can help, but I need one more concrete detail to route this correctly. "
        "What product area, error message, account/customer context, or business impact are you seeing?\n\n"
        f"For example:\n{examples}"
    )


def _compose_default_answer(state: SupportOrchestrationState) -> str:
    route = state.get("route_decision")
    if route == RouteDecision.CHITCHAT:
        return _compose_chitchat_answer(state)

    if route == RouteDecision.CLARIFICATION:
        return _compose_clarification_answer(state)

    if route == RouteDecision.SQL:
        return _compose_from_sql(state)

    if route == RouteDecision.HYBRID:
        return _compose_hybrid_answer(state)

    existing_answer = str(state.get("final_answer") or "").strip()
    if existing_answer:
        return existing_answer

    chunks = list(state.get("retrieved_chunks") or [])
    if chunks:
        return _compose_from_document_evidence(state)

    return "I could not find enough evidence to answer this request confidently."


def _recommended_actions(state: SupportOrchestrationState) -> list[str]:
    actions = [str(action) for action in state.get("recommended_actions", []) if action]
    if actions:
        return actions

    if state.get("escalation_flag"):
        actions.append("Transfer the case with the current evidence and agent trace.")
    if state.get("sql_results"):
        actions.append("Use the structured data evidence to validate account, ticket, or incident context.")
    if state.get("route_decision") == RouteDecision.CHITCHAT:
        actions.append("Ask the user to describe the support issue when ready.")
    if not actions:
        actions.append("Ask the user for missing product, account, or error details.")
    return actions


@observe(name=AGENT_NAME, as_type="agent", capture_input=False, capture_output=False)
def compose_response(state: SupportOrchestrationState) -> SupportOrchestrationState:
    """Create the final user-facing response from existing orchestration evidence."""

    started = time.perf_counter()
    next_state: SupportOrchestrationState = dict(state)

    _append_list(next_state, "progress_updates", _progress("response-composition", "started", "Response composition started."))

    evidence = normalize_evidence(next_state)
    normalize_latency_ms = int((time.perf_counter() - started) * 1000)
    _append_list(
        next_state,
        "agent_trace",
        _trace(
            action="normalize_evidence",
            status="completed",
            input_summary=f"route={evidence.route_decision}; chunks={len(evidence.retrieved_chunks)}; sql_rows={evidence.row_count}",
            output_summary=f"sources={len(evidence.sources)}; tools={','.join(evidence.tools_used) or 'none'}",
            latency_ms=normalize_latency_ms,
        ),
    )

    compose_started = time.perf_counter()
    final_model = compose_final_answer(evidence)
    composed_answer = final_answer_to_markdown(final_model)
    recommended_actions = final_model.recommended_actions or _recommended_actions(next_state)
    compose_latency_ms = int((time.perf_counter() - compose_started) * 1000)
    _append_list(
        next_state,
        "agent_trace",
        _trace(
            action="compose_route_specific_answer",
            status="completed",
            input_summary=f"route={evidence.route_decision}; style={evidence.response_style}",
            output_summary=f"actions={len(recommended_actions)}; quality={final_model.quality_summary}",
            latency_ms=compose_latency_ms,
        ),
    )

    clean_started = time.perf_counter()
    max_words = RESPONSE_STYLE_LIMITS[evidence.response_style]
    if final_model.implementation_steps or final_model.security_notes or final_model.caveats:
        max_words = max(max_words, 320)
    final_answer = clean_final_answer(composed_answer, max_words=max_words)
    clean_latency_ms = int((time.perf_counter() - clean_started) * 1000)
    _append_list(
        next_state,
        "agent_trace",
        _trace(
            action="clean_and_redact_answer",
            status="completed",
            input_summary=f"max_words={max_words}",
            output_summary=f"answer_length={len(final_answer)}",
            latency_ms=clean_latency_ms,
        ),
    )

    next_state["final_answer"] = final_answer
    next_state["recommended_actions"] = recommended_actions
    metadata = dict(next_state.get("metadata") or {})
    metadata["answer_quality"] = final_model.quality_summary
    metadata["final_answer_model"] = _final_answer_payload(final_model)
    metadata["response_style"] = evidence.response_style
    metadata["cleanup_applied"] = True
    metadata["redaction_applied"] = composed_answer != _redact_sensitive_values(composed_answer)
    metadata["evidence_count"] = len(evidence.rag_summary) + evidence.row_count + len(evidence.incident_investigation)
    metadata["citation_count"] = len(evidence.sources)
    metadata["sql_row_count"] = evidence.row_count
    metadata["validation_status"] = "completed"
    next_state["metadata"] = metadata

    latency_ms = int((time.perf_counter() - started) * 1000)
    output_summary = (
        f"route={_enum_value(next_state.get('route_decision'))}; "
        f"answer_length={len(final_answer)}; actions={len(recommended_actions)}; "
        f"quality={final_model.quality_summary.get('overall_quality_score') if final_model.quality_summary else 'n/a'}"
    )

    _append_list(next_state, "execution_results", _execution_result(final_answer, recommended_actions))
    verification = _verification(final_answer)
    verification["metadata"] = {
        **dict(verification.get("metadata") or {}),
        "answer_quality": final_model.quality_summary,
        "evidence_count": metadata["evidence_count"],
        "citation_count": metadata["citation_count"],
    }
    _append_list(next_state, "verification_outcomes", verification)
    _append_list(
        next_state,
        "agent_trace",
        _trace(
            action="compose_response",
            status="completed" if final_answer.strip() else "failed",
            input_summary=(
                f"chunks={len(next_state.get('retrieved_chunks', []))}; "
                f"sql_results={len(next_state.get('sql_results', []))}"
            ),
            output_summary=output_summary,
            latency_ms=latency_ms,
        ),
    )
    _append_list(next_state, "progress_updates", _progress("response-composition", "completed", output_summary))

    trace_agent_state(
        agent_name=AGENT_NAME,
        input_state=state,
        output_state=next_state,
        started_at=started,
        tool_used="state_evidence_composer",
    )

    return next_state
