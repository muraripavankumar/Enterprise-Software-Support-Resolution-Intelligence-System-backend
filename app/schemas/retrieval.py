from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class RetrievalMode(str, Enum):
    AGENT = "agent"
    VECTOR = "vector"
    SQL = "sql"


class RetrievalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=1, max_length=2000)
    mode: RetrievalMode = Field(default=RetrievalMode.AGENT)
    include_sources: bool = True
    include_raw_results: bool = False
    conversation_id: Optional[str] = Field(default=None, max_length=128)
    user_id: Optional[str] = Field(default=None, max_length=128)
    metadata: Dict[str, Any] = Field(default_factory=dict, max_length=50)


class RetrievalSourceNode(BaseModel):
    text_preview: str
    source_file: Optional[str] = None
    page_number: Optional[int] = None
    content_type: Optional[str] = None
    category: Optional[str] = None
    similarity_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CitationReference(BaseModel):
    citation_id: int
    document_name: str
    pages: List[int] = Field(default_factory=list)
    source_file: Optional[str] = None


class StructuredQueryResult(BaseModel):
    answer: str
    sql_query: Optional[str] = None
    table_used: Optional[str] = None
    row_count: int = 0
    raw_results: List[Any] = Field(default_factory=list)


class IncidentEvidenceRecord(BaseModel):
    incident_id: Optional[int] = None
    incident_type: Optional[str] = None
    severity: Optional[str] = None
    affected_region: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    resolution_status: Optional[str] = None
    root_cause: Optional[str] = None
    escalation_flag: bool = False
    correlation_score: Optional[float] = None
    correlation_reasons: List[str] = Field(default_factory=list)


class IncidentInvestigationEvidence(BaseModel):
    filters_used: Dict[str, Any] = Field(default_factory=dict)
    matched_incidents: List[IncidentEvidenceRecord] = Field(default_factory=list)
    active_critical_incident: bool = False
    active_critical_incident_correlated: bool = False
    max_correlation_score: float = 0.0
    correlation_threshold: float = 0.65
    investigation_summary: str = ""


class JiraTrackingEvidence(BaseModel):
    enabled: bool = False
    attempted: bool = False
    should_create: bool = False
    action: Optional[str] = None
    reason_code: Optional[str] = None
    project_key: Optional[str] = None
    issue_type: Optional[str] = None
    priority: Optional[str] = None
    issue_key: Optional[str] = None
    issue_url: Optional[str] = None
    duplicate_found: bool = False
    dedupe_jql: Optional[str] = None
    comment_added: bool = False
    triage_transition_attempted: bool = False
    triage_transition_status: Optional[str] = None
    status: Optional[str] = None
    error: Optional[str] = None


class AgentTraceEventResponse(BaseModel):
    agent_name: str
    action: Optional[str] = None
    input_summary: Optional[str] = None
    output_summary: Optional[str] = None
    status: Optional[str] = None
    timestamp: Optional[str] = None
    latency_ms: Optional[int] = None


class AnswerQuality(BaseModel):
    faithfulness_score: Optional[float] = None
    answer_relevance_score: Optional[float] = None
    overall_quality_score: Optional[float] = None
    evaluation_reasoning: Optional[str] = None
    evaluation_status: Optional[str] = None


class QualitySLOWarning(BaseModel):
    metric: str
    display_name: str
    value: float
    target: float
    message: str


class RetrievalResponse(BaseModel):
    success: bool
    mode: RetrievalMode
    question: str
    answer: str
    citations: List[str] = Field(default_factory=list)
    citation_references: List[CitationReference] = Field(default_factory=list)
    source_nodes: List[RetrievalSourceNode] = Field(default_factory=list)
    chunk_count: int = 0
    structured_result: Optional[StructuredQueryResult] = None
    incident_investigation: Optional[IncidentInvestigationEvidence] = None
    jira_tracking: Optional[JiraTrackingEvidence] = None
    intent: Optional[str] = None
    route_decision: Optional[str] = None
    severity: Optional[str] = None
    confidence_score: Optional[float] = None
    escalation_flag: bool = False
    escalation_target: Optional[str] = None
    tools_used: List[str] = Field(default_factory=list)
    agent_trace: List[AgentTraceEventResponse] = Field(default_factory=list)
    suggested_questions: List[str] = Field(default_factory=list)
    answer_quality: Optional[AnswerQuality] = None
    quality_warnings: List[QualitySLOWarning] = Field(default_factory=list)
    final_answer_model: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    latency_ms: int
    agent_trace_latency_ms: Optional[int] = None
    runtime_overhead_ms: Optional[int] = None
    cache_status: Optional[str] = None
    cache_route: Optional[str] = None
    cache_strategy: Optional[str] = None


class RetrievalErrorResponse(BaseModel):
    error: str
    detail: str
