import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from llama_index.llms.azure_openai import AzureOpenAI
from pydantic import BaseModel, Field, ValidationError

from app.core.config import settings
from app.core.langfuse import observe, trace_agent_state
from app.orchestration.state import (
    AgentTraceEvent,
    ExecutionResult,
    ProgressUpdate,
    RouteDecision,
    SupportIntent,
    SupportOrchestrationState,
    VerificationOutcome,
)

logger = logging.getLogger(__name__)

CLASSIFIER_CONFIDENCE_THRESHOLD = 0.70
AGENT_NAME = "intent_classification_agent"

_LLM: AzureOpenAI | None = None


@dataclass(frozen=True)
class ClassificationResult:
    intent: SupportIntent
    route_decision: RouteDecision
    confidence_score: float
    reason: str
    classifier: str
    matched_hits: dict[str, list[str]] = field(default_factory=dict)


class LLMClassificationPayload(BaseModel):
    """Validated structured output expected from the LLM classifier fallback."""

    intent: SupportIntent
    route_decision: RouteDecision
    confidence_score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=500)


GENERIC_RAG_KEYWORDS = {
    "what",
    "how",
    "steps",
    "recommended",
    "recommendation",
}

VAGUE_SUPPORT_PATTERNS = [
    re.compile(pattern)
    for pattern in [
        r"^\s*(issue|problem|error|failure)\s+(is\s+)?(happening|occurring|there|reported)?[\s!.?]*$",
        r"^\s*(it|this|that|something|everything)\s+(is\s+)?(broken|failing|failed|not\s+working|wrong)[\s!.?]*$",
        r"^\s*(not\s+working|broken|failing|failed|issue|problem|error)[\s!.?]*$",
        r"^\s*(we|i|users?)\s+(have|has|are\s+having|am\s+having)\s+(an?\s+)?(issue|problem|error)[\s!.?]*$",
        r"^\s*(help|please\s+help|need\s+help)[\s!.?]*$",
    ]
]

UNSUPPORTED_DOC_LOOKUP_PATTERNS = [
    re.compile(pattern)
    for pattern in [
        r"\bkeyboard\s+shortcut\b.{0,80}\b(legacy|desktop|payroll)\b",
        r"\b(legacy|desktop|payroll)\b.{0,80}\bkeyboard\s+shortcut\b",
        r"\b(private|deprecated|internal-only)\b.{0,80}\b(runbook|module|doc|document)\b",
        r"\b(runbook|module|doc|document)\b.{0,80}\b(private|deprecated|internal-only)\b",
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

VAGUE_PRODUCTION_PATTERNS = [
    re.compile(pattern)
    for pattern in [
        r"\bproduction\b.{0,80}\b(wrong|broken|issue|problem|cannot\s+describe|not\s+enough)\b",
        r"\b(something|everything|it)\b.{0,40}\bwrong\b.{0,80}\bproduction\b",
    ]
]

UNSAFE_DATA_ACCESS_PATTERNS = [
    re.compile(pattern)
    for pattern in [
        r"\b(show|list|display|find|get|reveal|dump)\b.{0,80}\b(password|passwords|api[_\s-]?key|secret|secrets|token|credential|credentials)\b",
        r"\b(password|passwords|api[_\s-]?key|secret|secrets|token|credential|credentials)\b.{0,80}\b(internal_users|internal|table|database)\b",
    ]
]

MEANINGFUL_SUPPORT_KEYWORDS = {
    "api",
    "oauth",
    "token",
    "401",
    "403",
    "500",
    "503",
    "customer",
    "customers",
    "account",
    "accounts",
    "ticket",
    "tickets",
    "incident",
    "incidents",
    "sla",
    "billing",
    "payment",
    "subscription",
    "latency",
    "performance",
    "region",
    "production",
    "outage",
    "security",
    "breach",
    "data loss",
    "vulnerability",
    "authentication",
    "authorization",
    "integration",
    "webhook",
    "endpoint",
}


RAG_KEYWORDS = {
    "api",
    "auth",
    "missing auth",
    "400",
    "401",
    "403",
    "429",
    "504",
    "authentication",
    "oauth",
    "token",
    "what",
    "configure",
    "configuration",
    "setup",
    "install",
    "installation",
    "deployment",
    "docker-compose",
    "health check",
    "guide",
    "documentation",
    "docs",
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
    "recommendation",
    "best practice",
    "troubleshoot",
    "troubleshooting",
    "error",
    "errors",
    "policy",
    "sla",
    "sla commitment",
    "sla commitments",
    "rca",
    "root cause analysis",
    "itil",
    "cache",
    "caching",
    "read-heavy",
    "rate limiting",
    "invalid json",
    "gateway timeout",
    "regional endpoint",
    "regional endpoints",
    "response timeline",
    "patch timeline",
    "performance",
    "latency",
    "how",
    "steps",
}

SQL_KEYWORDS = {
    "account",
    "accounts",
    "customer",
    "customers",
    "ticket",
    "tickets",
    "count",
    "how many",
    "number of",
    "list",
    "show",
    "active",
    "current",
    "all open",
    "open ticket",
    "open tickets",
    "critical severity tickets",
    "incident",
    "incidents",
    "incident log",
    "incident logs",
    "subscription",
    "plan",
    "status",
    "region",
    "sla tier",
    "payment",
    "suspended",
    "downgrade",
    "premium",
    "knowledge article",
    "article usage",
}

HIGH_RISK_KEYWORDS = {
    "outage",
    "production outage",
    "service down",
    "system down",
    "unavailable",
    "security vulnerability",
    "vulnerability",
    "data breach",
    "security breach",
    "credential breach",
    "credentials breached",
    "privacy breach",
    "unauthorized access",
    "customer data exposed",
    "api key exposed",
    "data loss",
    "critical incident",
    "unresolved critical",
    "unresolved alert",
    "production impact",
    "security exposure",
    "exposed api",
    "subscription downgrade impacting",
    "active integrations",
    "account suspension",
    "payment failure during incident",
    "multiple tickets",
    "systemic failure",
    "systemic failure pattern",
}

SECURITY_KEYWORDS = {
    "security",
    "security vulnerability",
    "vulnerability",
    "data breach",
    "security breach",
    "credential breach",
    "credentials breached",
    "unauthorized access",
    "exposure",
    "exposed",
    "token leak",
    "secret",
    "secrets",
    "password",
    "passwords",
    "api key",
    "api_key",
    "credential",
    "credentials",
}
BILLING_KEYWORDS = {
    "billing",
    "payment",
    "invoice",
    "subscription",
    "downgrade",
    "suspended",
    "account status",
    "account",
}
PERFORMANCE_KEYWORDS = {"latency", "performance", "slow", "timeout", "throughput", "scalability"}
INTEGRATION_KEYWORDS = {"api", "oauth", "token", "webhook", "integration", "401", "403", "endpoint"}
INCIDENT_KEYWORDS = {"incident", "outage", "alert", "production", "critical", "down", "unavailable"}
USAGE_KEYWORDS = {"how", "configure", "setup", "install", "use", "troubleshoot", "guide", "documentation"}

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
        r"\b(when|what|how|explain|describe)\b.{0,100}\b(rca|root cause analysis|post[-\s]?mortem)\b",
        r"\b(rca|root cause analysis|post[-\s]?mortem)\b.{0,100}\b(mandatory|required|recommended|timeline|requirement|requirements)\b",
        r"\b(support\s+)?(ownership|owner|responsibility|responsibilities|roles?|handoff|operating model|support model|ownership model)\b.{0,100}\b(incident|resolution)\b",
        r"\b(incident|resolution)\b.{0,100}\b(ownership|owner|responsibility|responsibilities|roles?|handoff|operating model|support model|ownership model)\b",
    ]
]

DOCUMENTATION_REFERENCE_PATTERNS = [
    re.compile(pattern)
    for pattern in [
        r"\b(what|when|how|explain|describe|which)\b.{0,120}\b(sla commitment|sla commitments|response time commitments?|support tier|priority customers?|priority support)\b",
        r"\b(sla commitment|sla commitments|response time commitments?|support tier|priority customers?|priority support)\b.{0,120}\b(what|when|how|explain|describe|which)\b",
        r"\b(what|when|how|explain|describe)\b.{0,120}\b(rca|root cause analysis|post[-\s]?mortem)\b",
        r"\b(rca|root cause analysis|post[-\s]?mortem)\b.{0,120}\b(mandatory|required|recommended|timeline|requirements?)\b",
        r"\b(what|how|which|recommended|recommendation)\b.{0,120}\b(cache|caching|read-heavy|cache-control|etag)\b",
        r"\b(cache|caching|read-heavy|cache-control|etag)\b.{0,120}\b(strategy|recommended|recommendation|headers?)\b",
        r"\b(verify|validate|check)\b.{0,120}\b(installation|deployment|deploy|docker-compose|health check|logs?)\b",
        r"\b(installation|deployment|deploy)\b.{0,120}\b(verify|validate|check|health check|logs?)\b",
        r"\b(400|invalid json|504|gateway timeout|429|rate limit|rate limiting)\b.{0,120}\b(what|how|check|recommended|action|handle|fix|resolve)\b",
        r"\b(what|how|check|recommended|action|handle|fix|resolve)\b.{0,120}\b(400|invalid json|504|gateway timeout|429|rate limit|rate limiting)\b",
        r"\b(regional endpoints?|region endpoint|nearest region|reduce latency|latency)\b.{0,120}\b(select|selected|choose|chosen|reduce|optimize)\b",
        r"\b(what|when|how)\b.{0,120}\b(high security vulnerability|security vulnerability|vulnerability response|response timeline|patch timeline)\b",
        r"\b(high security vulnerability|security vulnerability|vulnerability response|response timeline|patch timeline)\b.{0,120}\b(timeline|response|sla|patch|how long|when)\b",
    ]
]

OPERATIONAL_INCIDENT_TERMS = {
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

CHITCHAT_PATTERNS = [
    re.compile(pattern)
    for pattern in [
        r"^\s*(hi|hello|hey|hiya|yo|good\s+(morning|afternoon|evening))[\s!.?]*$",
        r"^\s*(hi|hello|hey)[\s,!.]*(eris|there)?[\s,!.]*(how\s+are\s+you|how's\s+it\s+going)?[\s!.?]*$",
        r"^\s*(how\s+are\s+you|how's\s+it\s+going|how\s+do\s+you\s+do)[\s!.?]*$",
        r"^\s*(thanks|thank\s+you|thank\s+you\s+so\s+much|thx|appreciate\s+it)[\s!.?]*$",
        r"^\s*(bye|goodbye|see\s+you|talk\s+later|take\s+care)[\s!.?]*$",
        r"^\s*(ok|okay|k|cool|great|nice|got\s+it|understood|yes|no|test|testing)[\s!.?]*$",
    ]
]


def _build_llm() -> AzureOpenAI:
    return AzureOpenAI(
        model=settings.azure_openai_chat_deployment,
        deployment_name=settings.azure_openai_chat_deployment,
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        temperature=0.0,
    )


def _get_llm() -> AzureOpenAI:
    global _LLM
    if _LLM is None:
        _LLM = _build_llm()
    return _LLM


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_list(state: SupportOrchestrationState, key: str, item: Any) -> None:
    values = list(state.get(key, []))
    values.append(item)
    state[key] = values  # type: ignore[literal-required]


@lru_cache(maxsize=512)
def _keyword_pattern(keyword: str) -> re.Pattern[str]:
    """Compile a whole-token matcher for one keyword or phrase."""

    normalized_keyword = " ".join(keyword.lower().strip().split())
    parts = [re.escape(part) for part in normalized_keyword.split()]
    phrase_pattern = r"\s+".join(parts)
    return re.compile(rf"(?<![a-zA-Z0-9_]){phrase_pattern}(?![a-zA-Z0-9_])")


def _keyword_matches(query: str, keyword: str) -> bool:
    return bool(_keyword_pattern(keyword).search(query))


def _keyword_hits(query: str, keywords: set[str]) -> list[str]:
    return sorted(keyword for keyword in keywords if _keyword_matches(query, keyword))


def _pattern_hits(query: str, patterns: list[re.Pattern[str]], label: str) -> list[str]:
    return [label for pattern in patterns if pattern.search(query)]


def _is_sla_breach_query(query: str) -> bool:
    return any(pattern.search(query) for pattern in SLA_BREACH_PATTERNS)


def _is_incident_process_documentation_query(query: str) -> bool:
    if not any(pattern.search(query) for pattern in INCIDENT_PROCESS_PATTERNS):
        return False
    return not _keyword_hits(query, OPERATIONAL_INCIDENT_TERMS)


def _is_documentation_reference_query(query: str) -> bool:
    if not any(pattern.search(query) for pattern in DOCUMENTATION_REFERENCE_PATTERNS):
        return False
    structured_lookup = bool(
        re.search(
            r"\b(list|show|display|find|get|count|how many|number of)\b.{0,80}\b(customers?|accounts?|tickets?|incidents?|article usage)\b",
            query,
        )
    )
    active_incident = bool(
        re.search(
            r"\b(active|ongoing|current|production down|service down|data loss|unauthorized access|breach detected|is exposed|exploited)\b",
            query,
        )
        and not re.search(r"\b(what|when|how|policy|timeline|commitment|recommended|guide|docs?)\b", query)
    )
    return not structured_lookup and not active_incident


def _is_documentation_process_query(query: str) -> bool:
    documentation_hits = _keyword_hits(
        query,
        {
            "what",
            "how",
            "explain",
            "describe",
            "recommended",
            "recommendation",
            "process",
            "procedure",
            "policy",
            "guide",
            "documentation",
            "docs",
            "lifecycle",
            "workflow",
            "management",
            "ownership",
            "support model",
            "ownership model",
            "best practice",
            "steps",
        },
    )
    if not documentation_hits:
        return False
    return not _keyword_hits(query, OPERATIONAL_INCIDENT_TERMS)


def _is_false_positive_risk_explanation(query: str) -> bool:
    return any(pattern.search(query) for pattern in FALSE_POSITIVE_RISK_EXPLANATION_PATTERNS)


def _is_unsupported_doc_lookup(query: str) -> bool:
    return any(pattern.search(query) for pattern in UNSUPPORTED_DOC_LOOKUP_PATTERNS)


def _is_vague_production_issue(query: str) -> bool:
    return any(pattern.search(query) for pattern in VAGUE_PRODUCTION_PATTERNS)


def _is_unsafe_data_access_request(query: str) -> bool:
    return any(pattern.search(query) for pattern in UNSAFE_DATA_ACCESS_PATTERNS)


def _classify_chitchat(query: str) -> ClassificationResult | None:
    normalized_query = " ".join(query.lower().strip().split())
    if not normalized_query:
        return None
    matched_patterns = [
        f"chitchat_pattern_{index}"
        for index, pattern in enumerate(CHITCHAT_PATTERNS, start=1)
        if pattern.search(normalized_query)
    ]
    if not matched_patterns:
        return None
    return ClassificationResult(
        intent=SupportIntent.CHITCHAT,
        route_decision=RouteDecision.CHITCHAT,
        confidence_score=1.0,
        reason="Small-talk input detected; bypassing support retrieval workflow.",
        classifier="chitchat_precheck",
        matched_hits={"chitchat": matched_patterns},
    )


def _classify_vague_support_query(query: str) -> ClassificationResult | None:
    normalized_query = " ".join(query.lower().strip().split())
    if not normalized_query:
        return None
    matched_patterns = [
        f"vague_support_pattern_{index}"
        for index, pattern in enumerate(VAGUE_SUPPORT_PATTERNS, start=1)
        if pattern.search(normalized_query)
    ]
    if not matched_patterns:
        return None
    return ClassificationResult(
        intent=SupportIntent.UNKNOWN,
        route_decision=RouteDecision.CLARIFICATION,
        confidence_score=0.66,
        reason="The query is too vague to route safely; request product, error, customer, and impact details.",
        classifier="vague_precheck",
        matched_hits={"vague_support": matched_patterns},
    )


def _has_meaningful_support_signal(query: str, result: ClassificationResult) -> bool:
    normalized_query = " ".join(query.lower().strip().split())
    if result.classifier in {"chitchat_precheck", "vague_precheck"}:
        return False
    if _keyword_hits(normalized_query, MEANINGFUL_SUPPORT_KEYWORDS):
        return True
    if any(result.matched_hits.get(key) for key in ("high_risk", "sql", "rag", "intent")):
        return True
    return len(normalized_query.split()) >= 6


def _blocks_llm_fallback(result: ClassificationResult) -> bool:
    if result.classifier in {"chitchat_precheck", "vague_precheck"}:
        return True
    if result.classifier == "guardrail_escalation":
        return True
    if result.matched_hits.get("unsupported_lookup"):
        return True
    if "vague production impact" in result.matched_hits.get("high_risk", []):
        return True
    return False


def _classify_guardrail_escalation(metadata: dict[str, Any]) -> ClassificationResult | None:
    if not metadata.get("manual_action_required"):
        return None

    reason = str(
        metadata.get("guardrail_escalation_reason")
        or "Operational state-change request requires human approval before action."
    )
    return ClassificationResult(
        intent=SupportIntent.INCIDENT,
        route_decision=RouteDecision.HIGH_RISK,
        confidence_score=0.96,
        reason=reason,
        classifier="guardrail_escalation",
        matched_hits={
            "high_risk": ["manual_action_required"],
            "guardrail": ["operational_state_change_requires_human_approval"],
        },
    )


def _score_from_hits(hits: list[str], base: float = 0.55, per_hit: float = 0.08, cap: float = 0.95) -> float:
    if not hits:
        return 0.0
    return min(cap, base + (len(hits) * per_hit))


def _score_rag_hits(hits: list[str]) -> float:
    if not hits:
        return 0.0
    specific_hits = [hit for hit in hits if hit not in GENERIC_RAG_KEYWORDS]
    generic_hits = [hit for hit in hits if hit in GENERIC_RAG_KEYWORDS]
    if not specific_hits:
        return min(0.62, 0.42 + (len(generic_hits) * 0.04))
    return min(0.92, 0.52 + (len(specific_hits) * 0.08) + (len(generic_hits) * 0.02))


def _detect_business_intent(query: str) -> tuple[SupportIntent, float, list[str]]:
    if re.search(r"\b(list|show|display|find|get|count|how many|number of)\b.*\b(incident|incidents|incident logs?)\b", query):
        return SupportIntent.INCIDENT, 0.86, ["structured incident lookup"]
    if re.search(r"\b(incident|incidents|incident logs?)\b.*\b(open|active|current|more than|status|region)\b", query):
        return SupportIntent.INCIDENT, 0.84, ["structured incident lookup"]
    if re.search(r"\b(401|403|oauth|api|authentication|authorization|token|endpoint|webhook|integration)\b", query):
        return SupportIntent.INTEGRATION, 0.86, _keyword_hits(query, INTEGRATION_KEYWORDS) or ["integration signal"]
    if re.search(r"\b(account status|suspended accounts?|customer|customers|subscription|billing|payment|invoice)\b", query):
        return SupportIntent.BILLING, 0.82, _keyword_hits(query, BILLING_KEYWORDS) or ["account/billing signal"]

    domain_candidates = [
        (SupportIntent.SECURITY, _keyword_hits(query, SECURITY_KEYWORDS), 1.50),
        (SupportIntent.BILLING, _keyword_hits(query, BILLING_KEYWORDS), 1.35),
        (SupportIntent.PERFORMANCE, _keyword_hits(query, PERFORMANCE_KEYWORDS), 1.30),
        (SupportIntent.INTEGRATION, _keyword_hits(query, INTEGRATION_KEYWORDS), 1.25),
        (SupportIntent.INCIDENT, _keyword_hits(query, INCIDENT_KEYWORDS), 1.20),
    ]
    candidates = [candidate for candidate in domain_candidates if candidate[1]]
    if not candidates:
        candidates = [(SupportIntent.USAGE, _keyword_hits(query, USAGE_KEYWORDS), 0.80)]
    candidates.sort(key=lambda item: len(item[1]) * item[2], reverse=True)
    intent, hits, _weight = candidates[0]
    if not hits:
        return SupportIntent.UNKNOWN, 0.0, []
    return intent, _score_from_hits(hits, base=0.50, per_hit=0.10, cap=0.92), hits


def _classify_local(query: str, user_role: str | None = None) -> ClassificationResult:
    normalized_query = " ".join(query.lower().strip().split())
    if not normalized_query:
        return ClassificationResult(
            intent=SupportIntent.UNKNOWN,
            route_decision=RouteDecision.CLARIFICATION,
            confidence_score=1.0,
            reason="Query is empty or whitespace-only.",
            classifier="local",
            matched_hits={"validation": ["empty_query"]},
        )

    if _is_incident_process_documentation_query(normalized_query):
        return ClassificationResult(
            intent=SupportIntent.INCIDENT,
            route_decision=RouteDecision.RAG,
            confidence_score=0.91,
            reason="Incident lifecycle/process documentation question detected.",
            classifier="local",
            matched_hits={"documentation_process": ["incident_process_documentation"]},
        )

    if _is_documentation_reference_query(normalized_query):
        intent = SupportIntent.USAGE
        if re.search(r"\b(oauth|api|400|401|403|429|504|gateway timeout|invalid json|rate limit)\b", normalized_query):
            intent = SupportIntent.INTEGRATION
        elif re.search(r"\b(sla|incident|itil|rca|root cause|post[-\s]?mortem)\b", normalized_query):
            intent = SupportIntent.INCIDENT
        elif re.search(r"\b(cache|caching|latency|regional|performance)\b", normalized_query):
            intent = SupportIntent.PERFORMANCE
        elif re.search(r"\b(security vulnerability|vulnerability response|patch timeline)\b", normalized_query):
            intent = SupportIntent.SECURITY
        return ClassificationResult(
            intent=intent,
            route_decision=RouteDecision.RAG,
            confidence_score=0.92,
            reason="Specific documentation, policy, troubleshooting, or timeline question detected.",
            classifier="local",
            matched_hits={"rag": ["documentation_reference_query"]},
        )

    if _is_false_positive_risk_explanation(normalized_query):
        return ClassificationResult(
            intent=SupportIntent.USAGE,
            route_decision=RouteDecision.RAG,
            confidence_score=0.90,
            reason="Documentation/policy explanation about risk terminology detected; not a live security incident.",
            classifier="local",
            matched_hits={"documentation_process": ["risk_term_explanation"]},
        )

    if _is_unsafe_data_access_request(normalized_query):
        return ClassificationResult(
            intent=SupportIntent.SECURITY,
            route_decision=RouteDecision.HIGH_RISK,
            confidence_score=0.94,
            reason="Unsafe sensitive-data access request detected; block automated data retrieval and escalate.",
            classifier="local",
            matched_hits={"high_risk": ["unsafe sensitive-data access request"]},
        )

    if _is_unsupported_doc_lookup(normalized_query):
        return ClassificationResult(
            intent=SupportIntent.UNKNOWN,
            route_decision=RouteDecision.RAG,
            confidence_score=0.55,
            reason="Unsupported or private-document lookup should run evidence validation and escalate if no grounded source exists.",
            classifier="local",
            matched_hits={"unsupported_lookup": ["unsupported_or_private_doc_lookup"]},
        )

    if _is_vague_production_issue(normalized_query):
        return ClassificationResult(
            intent=SupportIntent.INCIDENT,
            route_decision=RouteDecision.HIGH_RISK,
            confidence_score=0.65,
            reason="Vague production-impact signal requires human review with low confidence.",
            classifier="local",
            matched_hits={"high_risk": ["vague production impact"]},
        )

    high_risk_hits = _keyword_hits(normalized_query, HIGH_RISK_KEYWORDS)
    if not _is_sla_breach_query(normalized_query):
        high_risk_hits.extend(_pattern_hits(normalized_query, SECURITY_BREACH_PATTERNS, "security breach context"))
    rag_hits = _keyword_hits(normalized_query, RAG_KEYWORDS)
    sql_hits = _keyword_hits(normalized_query, SQL_KEYWORDS)
    intent, intent_score, intent_hits = _detect_business_intent(normalized_query)
    documentation_process_query = _is_documentation_process_query(normalized_query)

    if re.search(r"\b(tck|ticket|inc|incident)[-_]?\d+\b", normalized_query):
        sql_hits.append("record identifier")

    if re.search(r"\b(list|show|display|find|get)\b.*\b(ticket|tickets|customer|customers|incident|incidents)\b", normalized_query):
        sql_hits.append("record list query")

    if re.search(r"\b(count|how many|number of)\b.*\b(ticket|tickets|customer|customers|incident|incidents)\b", normalized_query):
        sql_hits.append("record count query")

    if re.search(r"\b(open|active|current|closed|resolved|critical|high|medium|low)\b.*\b(ticket|tickets|incident|incidents)\b", normalized_query):
        sql_hits.append("structured status filter")

    if _is_sla_breach_query(normalized_query):
        sql_hits.append("sla breach validation")
        rag_hits.append("sla policy")
        intent = SupportIntent.INCIDENT
        intent_score = max(intent_score, 0.82)

    matched_hits = {
        "high_risk": sorted(set(high_risk_hits)),
        "rag": sorted(set(rag_hits)),
        "sql": sorted(set(sql_hits)),
        "intent": sorted(set(intent_hits)),
    }
    if documentation_process_query:
        matched_hits["documentation_process"] = ["documentation_process_query"]

    if high_risk_hits and not documentation_process_query:
        high_risk_intent = SupportIntent.SECURITY if _keyword_hits(normalized_query, SECURITY_KEYWORDS) else SupportIntent.INCIDENT
        confidence = max(0.88, _score_from_hits(high_risk_hits, base=0.72, per_hit=0.08, cap=0.98))
        return ClassificationResult(
            intent=high_risk_intent,
            route_decision=RouteDecision.HIGH_RISK,
            confidence_score=confidence,
            reason=f"High-risk indicators detected: {', '.join(sorted(set(high_risk_hits)))}.",
            classifier="local",
            matched_hits=matched_hits,
        )

    if high_risk_hits and documentation_process_query:
        rag_hits.extend(["risk policy", "risk process"])
        matched_hits["rag"] = sorted(set(rag_hits))

    meaningful_rag_hits = [hit for hit in rag_hits if hit not in GENERIC_RAG_KEYWORDS]
    rag_score = _score_rag_hits(rag_hits)
    sql_score = _score_from_hits(sql_hits)

    policy_or_howto_query = bool(meaningful_rag_hits) and bool(
        re.search(r"\b(how|what|when|why|explain|describe|should)\b", normalized_query)
        and re.search(r"\b(policy|guide|documentation|docs|procedure|steps|troubleshoot|sla)\b", normalized_query)
    )
    explicit_structured_request = bool(
        re.search(
            r"\b(list|show|display|find|get|check|lookup|customer|account|ticket|tickets|incident logs?|subscription|database|table)\b",
            normalized_query,
        )
    )
    operational_context_requested = bool(
        re.search(
            r"\b(account|accounts|account context|account/sla|sla context|ticket|tickets|support ticket|customer account|customer status|customer tier|customer record)\b",
            normalized_query,
        )
        or "structured status filter" in sql_hits
    )
    if meaningful_rag_hits and sql_hits and operational_context_requested:
        confidence = min(0.95, 0.70 + (0.04 * (len(set(rag_hits)) + len(set(sql_hits)))))
        return ClassificationResult(
            intent=intent if intent != SupportIntent.UNKNOWN else SupportIntent.INTEGRATION,
            route_decision=RouteDecision.HYBRID,
            confidence_score=confidence,
            reason="Documentation evidence and structured account/ticket context are both required.",
            classifier="local",
            matched_hits=matched_hits,
        )

    api_error_documentation_query = bool(
        re.search(r"\b(401|403|missing auth|auth|authentication|authorization|oauth|api)\b", normalized_query)
        and re.search(r"\b(what does|mean|explain|check first|troubleshoot|fix|resolve)\b", normalized_query)
        and not re.search(r"\b(list|show|display|lookup|database|table|count|how many|number of)\b", normalized_query)
        and not operational_context_requested
    )
    if api_error_documentation_query:
        return ClassificationResult(
            intent=SupportIntent.INTEGRATION,
            route_decision=RouteDecision.RAG,
            confidence_score=max(rag_score, intent_score, 0.86),
            reason="API error-code documentation question detected.",
            classifier="local",
            matched_hits=matched_hits,
        )

    if meaningful_rag_hits and sql_hits and policy_or_howto_query and not explicit_structured_request:
        return ClassificationResult(
            intent=intent if intent != SupportIntent.UNKNOWN else SupportIntent.USAGE,
            route_decision=RouteDecision.RAG,
            confidence_score=max(rag_score, intent_score, 0.78),
            reason="Documentation/policy question detected without an explicit structured-data lookup request.",
            classifier="local",
            matched_hits=matched_hits,
        )

    if meaningful_rag_hits and sql_hits:
        confidence = min(0.95, 0.68 + (0.04 * (len(set(rag_hits)) + len(set(sql_hits)))))
        return ClassificationResult(
            intent=intent if intent != SupportIntent.UNKNOWN else SupportIntent.INTEGRATION,
            route_decision=RouteDecision.HYBRID,
            confidence_score=confidence,
            reason="Both documentation and structured-data indicators were detected.",
            classifier="local",
            matched_hits=matched_hits,
        )

    if sql_hits and sql_score >= rag_score:
        boosted_sql_score = max(sql_score, 0.79) if any(
            hit in sql_hits
            for hit in {
                "record identifier",
                "record list query",
                "record count query",
                "structured status filter",
                "active",
                "suspended",
                "trial",
                "inactive",
                "closed",
            }
        ) else sql_score
        return ClassificationResult(
            intent=intent if intent != SupportIntent.UNKNOWN else SupportIntent.INCIDENT,
            route_decision=RouteDecision.SQL,
            confidence_score=max(boosted_sql_score, intent_score),
            reason=f"Structured-data indicators detected: {', '.join(sorted(set(sql_hits)))}.",
            classifier="local",
            matched_hits=matched_hits,
        )

    if rag_hits:
        return ClassificationResult(
            intent=intent if intent != SupportIntent.UNKNOWN else SupportIntent.USAGE,
            route_decision=RouteDecision.RAG,
            confidence_score=max(rag_score, intent_score),
            reason=f"Documentation indicators detected: {', '.join(sorted(set(rag_hits)))}.",
            classifier="local",
            matched_hits=matched_hits,
        )

    return ClassificationResult(
        intent=SupportIntent.UNKNOWN,
        route_decision=RouteDecision.CLARIFICATION,
        confidence_score=0.35,
        reason="No strong local classification indicators were found.",
        classifier="local",
        matched_hits=matched_hits,
    )


def _json_from_llm_response(raw_response: str) -> dict[str, Any]:
    stripped = raw_response.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    else:
        object_match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if object_match:
            stripped = object_match.group(0)
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("LLM classifier did not return a JSON object")
    return parsed


def _classify_with_llm(query: str, user_role: str | None, local_result: ClassificationResult) -> ClassificationResult:
    prompt = f"""
Classify this enterprise software support query.

Return only one valid JSON object. Do not include markdown fences or extra text.

JSON schema:
- intent: one of usage, integration, incident, billing, security, performance, unknown
- route_decision: one of rag, sql, hybrid, high_risk, clarification
- confidence_score: number from 0.0 to 1.0
- reason: short explanation

Routing meaning:
- rag: documentation, policies, troubleshooting, guides, API docs
- sql: customer, account, ticket, incident log, subscription, SLA tier, structured operational records
- hybrid: needs both documentation and structured operational validation
- high_risk: outage, security vulnerability, data loss, critical incident, production impact, escalation-required case
- clarification: query is too ambiguous

Safety rules:
- Do not choose high_risk unless the query contains concrete critical evidence such as outage, production down,
  data loss, security breach, unauthorized access, active critical alert, or severe customer impact.
- Vague text such as "issue is happening", "it is broken", or "not working" must be clarification.
- SLA breach questions are operational/support-process validation, not security breach, unless security/data-loss
  indicators are also present.

User role: {user_role or "unknown"}
Query: {query}
""".strip()
    response = _get_llm().complete(prompt)
    payload = _json_from_llm_response(str(response))
    try:
        validated = LLMClassificationPayload.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"LLM classifier JSON failed validation: {exc}") from exc

    intent = validated.intent
    route_decision = validated.route_decision
    confidence_score = validated.confidence_score
    reason = validated.reason

    if intent == SupportIntent.CHITCHAT or route_decision == RouteDecision.CHITCHAT:
        raise ValueError("LLM fallback cannot emit chitchat; chitchat must be handled by precheck")

    if confidence_score <= 0:
        raise ValueError("LLM classifier returned non-positive confidence")

    return ClassificationResult(
        intent=intent,
        route_decision=route_decision,
        confidence_score=confidence_score,
        reason=reason,
        classifier="llm",
        matched_hits={
            "llm": [reason],
            "local_high_risk": local_result.matched_hits.get("high_risk", []),
            "local_rag": local_result.matched_hits.get("rag", []),
            "local_sql": local_result.matched_hits.get("sql", []),
            "local_intent": local_result.matched_hits.get("intent", []),
        },
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


def _non_empty_hits(matched_hits: dict[str, list[str]]) -> dict[str, list[str]]:
    return {key: values for key, values in matched_hits.items() if values}


def _debug_payload(
    *,
    result: ClassificationResult,
    local_result: ClassificationResult,
    query: str,
    user_role: str | None,
    llm_error: str | None,
) -> dict[str, Any]:
    return {
        "agent_name": AGENT_NAME,
        "classifier": result.classifier,
        "local_classifier": local_result.classifier,
        "llm_fallback_attempted": not _blocks_llm_fallback(local_result)
        and local_result.confidence_score < CLASSIFIER_CONFIDENCE_THRESHOLD
        and _has_meaningful_support_signal(query, local_result),
        "meaningful_support_signal": _has_meaningful_support_signal(query, local_result),
        "llm_error": llm_error,
        "user_role": user_role or "unknown",
        "intent": result.intent.value,
        "route_decision": result.route_decision.value,
        "confidence_score": result.confidence_score,
        "reason": result.reason,
        "matched_hits": _non_empty_hits(result.matched_hits),
        "local_matched_hits": _non_empty_hits(local_result.matched_hits),
    }


def _execution_result(result: ClassificationResult) -> ExecutionResult:
    return {
        "step_id": "intent-classification",
        "agent_name": AGENT_NAME,
        "result_type": "intent_classification",
        "summary": f"{result.intent.value} -> {result.route_decision.value} ({result.confidence_score:.2f})",
        "data": {
            "intent": result.intent.value,
            "route_decision": result.route_decision.value,
            "confidence_score": result.confidence_score,
            "reason": result.reason,
            "classifier": result.classifier,
            "matched_hits": _non_empty_hits(result.matched_hits),
        },
        "error": None,
        "timestamp": _utc_now(),
    }


def _verification(result: ClassificationResult) -> VerificationOutcome:
    passed = bool(result.intent and result.route_decision and result.confidence_score is not None)
    return {
        "check_name": "intent_classification_complete",
        "passed": passed,
        "score": result.confidence_score,
        "reason": "Intent, route decision, and confidence score were produced."
        if passed
        else "Intent classification output was incomplete.",
        "corrective_action": None if passed else "Retry classification or request clarification.",
        "metadata": {"classifier": result.classifier, "matched_hits": _non_empty_hits(result.matched_hits)},
    }


@observe(name=AGENT_NAME, as_type="agent", capture_input=False, capture_output=False)
def classify_intent(state: SupportOrchestrationState) -> SupportOrchestrationState:
    """Classify support query intent and route, updating orchestration state."""

    started = time.perf_counter()
    next_state: SupportOrchestrationState = dict(state)
    query = str(next_state.get("query") or "")
    metadata = dict(next_state.get("metadata") or {})
    user_role = metadata.get("user_role")
    user_role_text = str(user_role) if user_role is not None else None

    _append_list(next_state, "progress_updates", _progress("intent-classification", "started", "Intent classification started."))

    guardrail_escalation_result = _classify_guardrail_escalation(metadata)
    chitchat_result = _classify_chitchat(query) if guardrail_escalation_result is None else None
    if guardrail_escalation_result is not None:
        local_result = guardrail_escalation_result
    elif chitchat_result is not None:
        local_result = chitchat_result
    else:
        vague_result = _classify_vague_support_query(query)
        local_result = vague_result if vague_result is not None else _classify_local(query, user_role_text)
    result = local_result
    llm_error: str | None = None

    if (
        not _blocks_llm_fallback(local_result)
        and
        local_result.confidence_score < CLASSIFIER_CONFIDENCE_THRESHOLD
        and _has_meaningful_support_signal(query, local_result)
    ):
        try:
            result = _classify_with_llm(query, user_role_text, local_result)
        except Exception as exc:
            llm_error = f"LLM intent fallback failed: {exc}"
            logger.warning(llm_error)
            _append_list(next_state, "errors", llm_error)
            result = local_result

    next_state["intent"] = result.intent
    next_state["route_decision"] = result.route_decision
    next_state["confidence_score"] = result.confidence_score
    metadata["intent_classifier_debug"] = _debug_payload(
        result=result,
        local_result=local_result,
        query=query,
        user_role=user_role_text,
        llm_error=llm_error,
    )
    next_state["metadata"] = metadata

    latency_ms = int((time.perf_counter() - started) * 1000)
    hit_groups = ", ".join(sorted(_non_empty_hits(result.matched_hits).keys())) or "none"
    output_summary = (
        f"intent={result.intent.value}; route={result.route_decision.value}; "
        f"confidence={result.confidence_score:.2f}; classifier={result.classifier}; hit_groups={hit_groups}"
    )

    _append_list(next_state, "execution_results", _execution_result(result))
    _append_list(next_state, "verification_outcomes", _verification(result))
    _append_list(
        next_state,
        "agent_trace",
        _trace(
            action="classify_intent",
            status="completed" if llm_error is None else "completed_with_fallback_error",
            input_summary=f"query_length={len(query)}; user_role={user_role_text or 'unknown'}",
            output_summary=output_summary,
            latency_ms=latency_ms,
        ),
    )
    _append_list(next_state, "progress_updates", _progress("intent-classification", "completed", output_summary))

    trace_agent_state(
        agent_name=AGENT_NAME,
        input_state=state,
        output_state=next_state,
        started_at=started,
        tool_used="azure_openai_llm" if result.classifier == "llm" else "local_classifier",
    )

    return next_state
