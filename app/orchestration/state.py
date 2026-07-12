from enum import Enum
from typing import Any, TypedDict


MAX_ORCHESTRATION_ITERATIONS = 2


class SupportIntent(str, Enum):
    """Business intent detected from a support query."""

    USAGE = "usage"
    INTEGRATION = "integration"
    INCIDENT = "incident"
    BILLING = "billing"
    SECURITY = "security"
    PERFORMANCE = "performance"
    CHITCHAT = "chitchat"
    UNKNOWN = "unknown"


class RouteDecision(str, Enum):
    """Primary workflow route selected for the query."""

    RAG = "rag"
    SQL = "sql"
    HYBRID = "hybrid"
    HIGH_RISK = "high_risk"
    CLARIFICATION = "clarification"
    CHITCHAT = "chitchat"


class SeverityLevel(str, Enum):
    """Operational severity classification."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SeverityPriority(str, Enum):
    """Operational priority classification."""

    P0 = "p0"
    P1 = "p1"
    P2 = "p2"
    P3 = "p3"


class EscalationTarget(str, Enum):
    """Human or operational team selected for escalation."""

    NONE = "none"
    L1_SUPPORT = "l1_support"
    L2_SUPPORT = "l2_support"
    L3_SUPPORT = "l3_support"
    SECURITY_TEAM = "security_team"
    INCIDENT_RESPONSE = "incident_response"
    ENGINEERING = "engineering"


class ExecutionPlanStep(TypedDict, total=False):
    """Planned unit of work for the orchestration run."""

    step_id: str
    agent_name: str
    action: str
    expected_output: str
    status: str
    depends_on: list[str]
    metadata: dict[str, Any]


class ProgressUpdate(TypedDict, total=False):
    """Progress update emitted while executing the plan."""

    step_id: str
    agent_name: str
    status: str
    message: str
    timestamp: str
    metadata: dict[str, Any]


class ExecutionResult(TypedDict, total=False):
    """Normalized result produced by an orchestration step."""

    step_id: str
    agent_name: str
    result_type: str
    summary: str
    data: dict[str, Any]
    error: str | None
    timestamp: str


class AgentResult(TypedDict, total=False):
    """Common normalized result envelope for every orchestration agent."""

    agent_name: str
    status: str
    route_decision: str | None
    summary: str
    data: dict[str, Any]
    confidence_score: float | None
    latency_ms: int | None
    token_estimate: int | None
    error: str | None
    timestamp: str


class VerificationOutcome(TypedDict, total=False):
    """Verification result for a plan step or final response."""

    check_name: str
    passed: bool
    score: float | None
    reason: str
    corrective_action: str | None
    metadata: dict[str, Any]


class IterationRecord(TypedDict, total=False):
    """One orchestration attempt, including corrections and verification."""

    iteration_number: int
    plan_summary: str | None
    completed_steps: list[str]
    failed_steps: list[str]
    verification_outcomes: list[VerificationOutcome]
    correction_summary: str | None
    should_retry: bool
    timestamp: str


class RetrievedChunk(TypedDict, total=False):
    """Evidence retrieved from unstructured documentation."""

    chunk_text: str
    source_file: str | None
    page_number: int | None
    content_type: str | None
    category: str | None
    score: float | None
    metadata: dict[str, Any]


class SQLResult(TypedDict, total=False):
    """Evidence retrieved from structured operational data."""

    answer: str
    sql_query: str | None
    tables_used: list[str]
    row_count: int
    raw_results: list[Any]
    error: str | None


class CustomerContext(TypedDict, total=False):
    """Validated customer account context from the operational database."""

    customer_id: int | None
    company_name: str | None
    sla_level: str | None
    subscription_tier: str | None
    account_status: str | None
    region: str | None
    account_suspended: bool
    lookup_status: str
    lookup_reason: str | None


class IncidentRecord(TypedDict, total=False):
    """Operational incident record correlated with the support query."""

    incident_id: int | None
    incident_type: str | None
    severity: str | None
    affected_region: str | None
    start_time: str | None
    end_time: str | None
    resolution_status: str | None
    root_cause: str | None
    escalation_flag: bool
    correlation_score: float
    correlation_reasons: list[str]


class IncidentInvestigationResult(TypedDict, total=False):
    """Incident investigation result from structured incident logs."""

    filters_used: dict[str, Any]
    matched_incidents: list[IncidentRecord]
    active_critical_incident: bool
    active_critical_incident_correlated: bool
    max_correlation_score: float
    correlation_threshold: float
    investigation_summary: str


class JiraTrackingResult(TypedDict, total=False):
    """Jira engineering issue tracking result produced during escalation."""

    enabled: bool
    attempted: bool
    should_create: bool
    action: str
    reason_code: str | None
    project_key: str | None
    issue_type: str | None
    priority: str | None
    issue_key: str | None
    issue_url: str | None
    duplicate_found: bool
    dedupe_jql: str | None
    comment_added: bool
    triage_transition_attempted: bool
    triage_transition_status: str | None
    status: str
    error: str | None
    metadata: dict[str, Any]


class HybridResult(TypedDict, total=False):
    """Synthesized result from documentation and structured data evidence."""

    combined_summary: str
    rag_evidence_count: int
    sql_evidence_count: int
    conflicts: list[str]
    recommended_action: str | None


class GuardrailFlag(TypedDict, total=False):
    """Result of a safety, policy, access, or grounding check."""

    name: str
    passed: bool
    severity: SeverityLevel | str
    reason: str
    metadata: dict[str, Any]


class AgentTraceEvent(TypedDict, total=False):
    """Auditable event emitted by an orchestration node or agent."""

    agent_name: str
    action: str
    input_summary: str | None
    output_summary: str | None
    status: str
    timestamp: str
    latency_ms: int | None


class SupportOrchestrationState(TypedDict, total=False):
    """Shared LangGraph state contract for support query orchestration."""

    query: str
    conversation_id: str | None
    user_id: str | None
    metadata: dict[str, Any]

    execution_plan: list[ExecutionPlanStep]
    progress_updates: list[ProgressUpdate]
    execution_results: list[ExecutionResult]
    agent_results: list[AgentResult]
    verification_outcomes: list[VerificationOutcome]
    iteration_count: int
    max_iterations: int
    iteration_history: list[IterationRecord]

    intent: SupportIntent
    route_decision: RouteDecision
    severity_priority: SeverityPriority
    severity: SeverityLevel
    severity_reason: str | None

    retrieved_chunks: list[RetrievedChunk]
    sql_results: list[SQLResult]
    customer_context: CustomerContext
    incident_investigation: IncidentInvestigationResult
    hybrid_result: HybridResult | None

    confidence_score: float | None
    guardrail_flags: list[GuardrailFlag]

    escalation_flag: bool
    escalation_target: EscalationTarget
    escalation_reason: str | None
    jira_tracking_result: JiraTrackingResult
    jira_issue_key: str | None
    jira_issue_url: str | None

    final_answer: str | None
    citations: list[str]
    recommended_actions: list[str]

    agent_trace: list[AgentTraceEvent]
    errors: list[str]
    latency_ms: int | None
