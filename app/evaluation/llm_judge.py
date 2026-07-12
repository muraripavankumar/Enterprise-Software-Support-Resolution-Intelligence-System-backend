import json
import re
from functools import lru_cache
from typing import Any, Literal

from openai import AzureOpenAI, OpenAIError
from pydantic import BaseModel, Field, ValidationError, field_validator

from app.core.config import settings
from app.core.logging import get_logger
from app.evaluation.slo_config import SLO_CONFIG_BY_NAME

logger = get_logger(__name__)

JudgeMetric = Literal["faithfulness", "answer_relevance", "llm_judge_quality"]


class JudgeEvaluation(BaseModel):
    metric_name: JudgeMetric
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    failure_reasons: list[str] = Field(default_factory=list)
    reasoning: str
    error: str | None = None

    @field_validator("reasoning")
    @classmethod
    def _reasoning_not_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("reasoning must not be empty")
        return cleaned

    @field_validator("failure_reasons")
    @classmethod
    def _failure_reasons_are_codes(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for value in values:
            reason = re.sub(r"[^a-z0-9_]+", "_", str(value).strip().lower()).strip("_")
            if reason and reason not in cleaned:
                cleaned.append(reason)
        return cleaned


class _JudgePayload(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    passed: bool | None = None
    failure_reasons: list[str] = Field(default_factory=list)
    reasoning: str


@lru_cache(maxsize=1)
def _client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
    )


def _passes_target(metric_name: JudgeMetric, score: float) -> bool:
    slo_config = SLO_CONFIG_BY_NAME.get(metric_name)
    if slo_config is None:
        return score >= 0.8
    return score >= slo_config.target if slo_config.higher_is_better else score <= slo_config.target


def _failed(metric_name: JudgeMetric, reason: str) -> JudgeEvaluation:
    return JudgeEvaluation(
        metric_name=metric_name,
        score=0.0,
        passed=False,
        failure_reasons=["judge_unavailable"],
        reasoning=reason,
        error=reason,
    )


def _truncate(value: str, max_chars: int = 12000) -> str:
    text = str(value or "").strip()
    return text[:max_chars]


def _json_from_response(raw_response: str) -> dict[str, Any]:
    stripped = raw_response.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    else:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if match:
            stripped = match.group(0)
    payload = json.loads(stripped)
    if not isinstance(payload, dict):
        raise ValueError("judge response was not a JSON object")
    return payload


def _judge_prompt(metric_name: JudgeMetric, question: str, context: str, answer: str) -> list[dict[str, str]]:
    criteria = {
        "faithfulness": (
            "Score whether the answer is supported by the provided context. "
            "Penalize unsupported claims, invented facts, or contradictions. "
            "Strongly penalize missing exact row/table values when the question asks for a specific tier, severity, error code, phase, version, or timeline. "
            "Penalize raw chunk dumps, repeated headings, or copied irrelevant context. "
            "A score of 1.0 means every material claim is grounded in context. "
            "For RAG answers, citations should support the cited facts. For SQL answers, exact structured values should match the SQL/structured evidence."
        ),
        "answer_relevance": (
            "Score whether the answer directly and completely addresses the user's question. "
            "Penalize generic answers, missing requested entities, irrelevant recommended actions, and wrong route behavior "
            "(for example SQL-style output for documentation questions or missing citations for RAG answers). "
            "Ignore minor citation formatting and focus on usefulness and focus. "
            "Reward concise answers that include the specific steps, values, or follow-up question the user needs."
        ),
        "llm_judge_quality": (
            "Score the overall enterprise support answer quality across correctness, grounding, relevance, "
            "clarity, escalation judgment, and operational usefulness. Penalize wrong route choice, missing RAG citations, "
            "generic boilerplate recommendations, missing structured values in SQL/Hybrid answers, raw chunks/JSON, and unsafe or overbroad advice. "
            "For high-risk answers, verify severity, escalation target, reason, immediate actions, and human handoff are clear."
        ),
    }[metric_name]

    scoring_scale = (
        "Scoring scale:\n"
        "1.00 = Excellent: fully correct, grounded, complete, actionable, and route-appropriate.\n"
        "0.80 = Good: mostly correct with only minor omissions or wording issues.\n"
        "0.60 = Needs review: partially correct but missing important details, exact values, citations, or actions.\n"
        "0.40 = Poor: generic, weakly grounded, materially incomplete, or confusing.\n"
        "0.00 = Failing: wrong, unsafe, hallucinated, contradictory, irrelevant, or unusable.\n"
    )
    route_contract = (
        "Route-aware checks:\n"
        "- RAG/documentation answers should include specific facts from context and citations when sources are available.\n"
        "- SQL/structured-data answers should include exact table-derived values and should not require document citations.\n"
        "- Hybrid answers should combine structured status with document/policy guidance.\n"
        "- High-risk answers should preserve escalation, state severity/target/reason, and avoid claiming final automated resolution.\n"
        "- Clarification answers should ask a focused follow-up question instead of guessing.\n"
        "- Chitchat answers should be brief and should not pretend evidence was retrieved.\n"
    )
    failure_reason_catalog = (
        "Use zero or more failure_reasons from this catalog when applicable: "
        "unsupported_claim, contradiction, missing_exact_value, missing_citation, wrong_route, "
        "generic_answer, irrelevant_action, incomplete_answer, raw_chunk_dump, raw_json_dump, "
        "unsafe_advice, incorrect_escalation, missing_escalation_detail, poor_clarity."
    )

    user_content = (
        f"Metric: {metric_name}\n"
        f"Criteria: {criteria}\n\n"
        f"{scoring_scale}\n"
        f"{route_contract}\n"
        f"{failure_reason_catalog}\n\n"
        f"Question:\n{_truncate(question, 4000)}\n\n"
        f"Context:\n{_truncate(context)}\n\n"
        f"Answer:\n{_truncate(answer, 8000)}\n\n"
        "Return strict JSON only with this schema:\n"
        '{"score": number_between_0_and_1, "passed": boolean, "failure_reasons": ["reason_code"], "reasoning": "short reason"}'
    )

    return [
        {
            "role": "system",
            "content": (
                "You are a strict evaluator for an enterprise software support AI system. "
                "Return only valid JSON. Do not include markdown."
            ),
        },
        {"role": "user", "content": user_content},
    ]


def _run_judge(metric_name: JudgeMetric, question: str, context: str, answer: str) -> JudgeEvaluation:
    if not settings.enable_llm_judge:
        return _failed(metric_name, "LLM judge is disabled by ENABLE_LLM_JUDGE=false.")

    try:
        response = _client().chat.completions.create(
            model=settings.llm_judge_model or settings.azure_openai_chat_deployment,
            messages=_judge_prompt(metric_name, question, context, answer),
            temperature=0,
            response_format={"type": "json_object"},
            max_tokens=300,
        )
        raw = response.choices[0].message.content or "{}"
        payload = _JudgePayload.model_validate(_json_from_response(raw))
        passed = _passes_target(metric_name, payload.score) if payload.passed is None else payload.passed
        failure_reasons = payload.failure_reasons
        if not passed and not failure_reasons:
            failure_reasons = ["below_slo_target"]
        return JudgeEvaluation(
            metric_name=metric_name,
            score=payload.score,
            passed=passed,
            failure_reasons=failure_reasons,
            reasoning=payload.reasoning,
        )
    except (OpenAIError, ValidationError, ValueError, json.JSONDecodeError, IndexError, AttributeError) as exc:
        logger.warning("LLM judge failed for metric=%s: %s", metric_name, exc)
        return _failed(metric_name, f"LLM judge failed for {metric_name}: {exc}")


def evaluate_faithfulness(question: str, context: str, answer: str) -> JudgeEvaluation:
    return _run_judge("faithfulness", question, context, answer)


def evaluate_answer_relevance(question: str, answer: str) -> JudgeEvaluation:
    return _run_judge("answer_relevance", question, "", answer)


def evaluate_overall_quality(question: str, context: str, answer: str) -> JudgeEvaluation:
    return _run_judge("llm_judge_quality", question, context, answer)
