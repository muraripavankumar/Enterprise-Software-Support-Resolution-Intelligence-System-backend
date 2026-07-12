from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


ScoreType = Literal["numeric", "ratio", "latency_seconds", "violation_count"]


class SLOConfig(BaseModel):
    """Langfuse/evaluation score configuration for one production SLO."""

    model_config = ConfigDict(frozen=True)

    name: str
    display_name: str
    category: str
    description: str
    min_value: float
    max_value: float
    target: float
    unit: str
    higher_is_better: bool
    score_type: ScoreType
    route_targets: dict[str, float] = Field(default_factory=dict)


SLO_CONFIGS: tuple[SLOConfig, ...] = (
    SLOConfig(
        name="task_success_rate",
        display_name="Task Success Rate",
        category="quality",
        description="Share of support queries that reach a useful answer or correct escalation outcome.",
        min_value=0.0,
        max_value=1.0,
        target=0.90,
        unit="ratio",
        higher_is_better=True,
        score_type="ratio",
    ),
    SLOConfig(
        name="faithfulness",
        display_name="Faithfulness",
        category="rag_quality",
        description="How well the answer is grounded in retrieved context without unsupported claims.",
        min_value=0.0,
        max_value=1.0,
        target=0.90,
        unit="score",
        higher_is_better=True,
        score_type="numeric",
    ),
    SLOConfig(
        name="answer_relevance",
        display_name="Answer Relevance",
        category="answer_quality",
        description="How directly the answer addresses the user's question.",
        min_value=0.0,
        max_value=1.0,
        target=0.85,
        unit="score",
        higher_is_better=True,
        score_type="numeric",
    ),
    SLOConfig(
        name="llm_judge_quality",
        display_name="LLM Judge Quality",
        category="answer_quality",
        description="Overall semantic quality judged across correctness, clarity, grounding, safety, and route-appropriate handling.",
        min_value=0.0,
        max_value=1.0,
        target=0.85,
        unit="score",
        higher_is_better=True,
        score_type="numeric",
    ),
    SLOConfig(
        name="context_precision",
        display_name="Context Precision",
        category="retrieval_quality",
        description="Share of retrieved context that is useful for answering the question.",
        min_value=0.0,
        max_value=1.0,
        target=0.80,
        unit="score",
        higher_is_better=True,
        score_type="numeric",
    ),
    SLOConfig(
        name="retrieval_recall_at_5",
        display_name="Retrieval Recall@5",
        category="retrieval_quality",
        description="Whether the relevant supporting evidence appears in the top five retrieved items.",
        min_value=0.0,
        max_value=1.0,
        target=0.85,
        unit="score",
        higher_is_better=True,
        score_type="numeric",
    ),
    SLOConfig(
        name="source_attribution_rate",
        display_name="Source Attribution Rate",
        category="trust",
        description="Share of RAG answers that include at least one source citation.",
        min_value=0.0,
        max_value=1.0,
        target=1.00,
        unit="ratio",
        higher_is_better=True,
        score_type="ratio",
    ),
    SLOConfig(
        name="sql_correctness",
        display_name="SQL Correctness",
        category="structured_data",
        description="Correctness of generated/read-only SQL and structured-data evidence.",
        min_value=0.0,
        max_value=1.0,
        target=0.95,
        unit="score",
        higher_is_better=True,
        score_type="numeric",
    ),
    SLOConfig(
        name="query_routing_accuracy",
        display_name="Query Routing Accuracy",
        category="orchestration",
        description="Accuracy of routing queries to RAG, SQL, hybrid, high-risk, or clarification paths.",
        min_value=0.0,
        max_value=1.0,
        target=0.95,
        unit="score",
        higher_is_better=True,
        score_type="numeric",
    ),
    SLOConfig(
        name="intent_classification_accuracy",
        display_name="Intent Classification Accuracy",
        category="orchestration",
        description="Accuracy of business intent classification such as usage, integration, security, or incident.",
        min_value=0.0,
        max_value=1.0,
        target=0.90,
        unit="score",
        higher_is_better=True,
        score_type="numeric",
    ),
    SLOConfig(
        name="risk_classification_accuracy",
        display_name="Risk Classification Accuracy",
        category="safety",
        description="Accuracy of severity and risk classification for production, security, and customer-impact cases.",
        min_value=0.0,
        max_value=1.0,
        target=0.95,
        unit="score",
        higher_is_better=True,
        score_type="numeric",
    ),
    SLOConfig(
        name="escalation_accuracy",
        display_name="Escalation Accuracy",
        category="human_handoff",
        description="Accuracy of escalation decisions and target-team selection.",
        min_value=0.0,
        max_value=1.0,
        target=0.90,
        unit="score",
        higher_is_better=True,
        score_type="numeric",
    ),
    SLOConfig(
        name="critical_escalation_recall",
        display_name="Critical Escalation Recall",
        category="human_handoff",
        description="Recall for escalating P0/critical incidents, outages, data loss, and security vulnerabilities.",
        min_value=0.0,
        max_value=1.0,
        target=1.00,
        unit="score",
        higher_is_better=True,
        score_type="numeric",
    ),
    SLOConfig(
        name="guardrail_effectiveness",
        display_name="Guardrail Effectiveness",
        category="safety",
        description="Share of unsafe, low-confidence, or policy-violating cases correctly blocked, redacted, or escalated.",
        min_value=0.0,
        max_value=1.0,
        target=0.95,
        unit="score",
        higher_is_better=True,
        score_type="numeric",
    ),
    SLOConfig(
        name="unauthorized_data_access",
        display_name="Unauthorized Data Access",
        category="security",
        description="Count of unauthorized data access violations; target is zero.",
        min_value=0.0,
        max_value=100.0,
        target=0.0,
        unit="violations",
        higher_is_better=False,
        score_type="violation_count",
    ),
    SLOConfig(
        name="p95_response_latency_seconds",
        display_name="P95 Response Latency",
        category="performance",
        description="P95 end-to-end response latency; standard routes target <= 5s and agent routes target <= 10s.",
        min_value=0.0,
        max_value=120.0,
        target=5.0,
        unit="seconds",
        higher_is_better=False,
        score_type="latency_seconds",
        route_targets={"standard": 5.0, "agent": 10.0},
    ),
)

SLO_CONFIG_BY_NAME: dict[str, SLOConfig] = {config.name: config for config in SLO_CONFIGS}


def get_slo_config(name: str) -> SLOConfig:
    return SLO_CONFIG_BY_NAME[name]


def list_slo_configs() -> list[SLOConfig]:
    return list(SLO_CONFIGS)
