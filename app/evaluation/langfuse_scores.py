import json
from typing import Any

from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.langfuse import get_langfuse_client, trace_final_response
from app.core.logging import get_logger
from app.evaluation.llm_judge import JudgeEvaluation, evaluate_answer_relevance, evaluate_faithfulness, evaluate_overall_quality
from app.evaluation.slo_config import SLO_CONFIG_BY_NAME

logger = get_logger(__name__)


class ScoreAttachmentResult(BaseModel):
    name: str
    value: float = Field(ge=0.0)
    attached: bool
    comment: str | None = None
    error: str | None = None


class EvaluationScoreResult(BaseModel):
    trace_id: str
    scores: dict[str, JudgeEvaluation]
    attachments: dict[str, ScoreAttachmentResult]


class RuntimeSLOScoreResult(BaseModel):
    trace_id: str
    attachments: dict[str, ScoreAttachmentResult]
    judge_result: EvaluationScoreResult | None = None


def _bounded_score(name: str, value: float) -> float:
    score_value = float(value)
    slo_config = SLO_CONFIG_BY_NAME.get(name)
    if not slo_config:
        return score_value
    return max(slo_config.min_value, min(slo_config.max_value, score_value))


def _score_metadata(name: str) -> dict[str, Any]:
    slo_config = SLO_CONFIG_BY_NAME.get(name)
    if not slo_config:
        return {}
    return {
        "display_name": slo_config.display_name,
        "category": slo_config.category,
        "min_value": slo_config.min_value,
        "max_value": slo_config.max_value,
        "target": slo_config.target,
        "unit": slo_config.unit,
        "higher_is_better": slo_config.higher_is_better,
        "score_type": slo_config.score_type,
        "route_targets": slo_config.route_targets,
    }


def _target_for_score(name: str, response_data: dict[str, Any]) -> float | None:
    slo_config = SLO_CONFIG_BY_NAME.get(name)
    if not slo_config:
        return None
    if name == "p95_response_latency_seconds":
        route_targets = slo_config.route_targets or {}
        mode = str(response_data.get("mode") or "").lower()
        if mode == "agent" and "agent" in route_targets:
            return route_targets["agent"]
        if "standard" in route_targets:
            return route_targets["standard"]
    return slo_config.target


def _score_passed(name: str, value: float, response_data: dict[str, Any]) -> bool:
    slo_config = SLO_CONFIG_BY_NAME.get(name)
    target = _target_for_score(name, response_data)
    if slo_config is None or target is None:
        return True
    return value >= target if slo_config.higher_is_better else value <= target


def attach_score_to_trace(
    trace_id: str,
    name: str,
    value: float,
    comment: str | None = None,
) -> ScoreAttachmentResult:
    """Attach a numeric score to a Langfuse trace by trace_id."""

    if not trace_id:
        return ScoreAttachmentResult(
            name=name,
            value=_bounded_score(name, value),
            attached=False,
            comment=comment,
            error="trace_id is required",
        )

    client = get_langfuse_client()
    if client is None:
        return ScoreAttachmentResult(
            name=name,
            value=_bounded_score(name, value),
            attached=False,
            comment=comment,
            error="Langfuse client is not configured",
        )

    score_value = _bounded_score(name, value)
    try:
        client.create_score(
            trace_id=trace_id,
            name=name,
            value=score_value,
            data_type="NUMERIC",
            comment=comment,
            metadata=_score_metadata(name),
        )
        return ScoreAttachmentResult(name=name, value=score_value, attached=True, comment=comment)
    except Exception as exc:
        logger.warning("Failed to attach Langfuse score name=%s trace_id=%s: %s", name, trace_id, exc)
        return ScoreAttachmentResult(
            name=name,
            value=score_value,
            attached=False,
            comment=comment,
            error=str(exc),
        )


def _attach_slo_summary_scores(
    trace_id: str,
    response_data: dict[str, Any],
    numeric_scores: dict[str, tuple[float, str]],
    judge_scores: dict[str, JudgeEvaluation],
) -> None:
    client = get_langfuse_client()
    if client is None or not trace_id:
        return

    metrics: list[dict[str, Any]] = []
    for name, (raw_value, comment) in numeric_scores.items():
        value = _bounded_score(name, raw_value)
        slo_config = SLO_CONFIG_BY_NAME.get(name)
        metrics.append(
            {
                "name": name,
                "display_name": slo_config.display_name if slo_config else name,
                "value": value,
                "target": _target_for_score(name, response_data),
                "unit": slo_config.unit if slo_config else "score",
                "higher_is_better": slo_config.higher_is_better if slo_config else True,
                "passed": _score_passed(name, value, response_data),
                "comment": comment,
            }
        )

    for name, evaluation in judge_scores.items():
        value = _bounded_score(name, evaluation.score)
        slo_config = SLO_CONFIG_BY_NAME.get(name)
        metrics.append(
            {
                "name": name,
                "display_name": slo_config.display_name if slo_config else name,
                "value": value,
                "target": _target_for_score(name, response_data),
                "unit": slo_config.unit if slo_config else "score",
                "higher_is_better": slo_config.higher_is_better if slo_config else True,
                "passed": evaluation.passed,
                "comment": evaluation.reasoning,
                "failure_reasons": evaluation.failure_reasons,
                "error": evaluation.error,
            }
        )

    if not metrics:
        return

    passed_count = sum(1 for metric in metrics if metric["passed"])
    summary_value = passed_count / len(metrics)
    metadata = {
        "passed": passed_count,
        "total": len(metrics),
        "metrics": metrics,
        "hint": "Use this score metadata or the slo_metrics text score to inspect all SLO values without the collapsed trace-tree chips.",
    }

    try:
        client.create_score(
            trace_id=trace_id,
            name="slo_summary",
            value=summary_value,
            data_type="NUMERIC",
            comment=f"{passed_count}/{len(metrics)} runtime SLO checks passed. Open metadata for all metric values.",
            metadata=metadata,
        )
    except Exception as exc:
        logger.warning("Failed to attach Langfuse SLO summary scores trace_id=%s: %s", trace_id, exc)


def _response_context(response_data: dict[str, Any]) -> str:
    context_parts: list[str] = []
    for node in response_data.get("source_nodes") or []:
        if isinstance(node, dict):
            preview = str(node.get("text_preview") or "").strip()
            source = str(node.get("source_file") or "source").strip()
            if preview:
                context_parts.append(f"Source: {source}\n{preview}")

    structured_result = response_data.get("structured_result")
    if isinstance(structured_result, dict):
        answer = str(structured_result.get("answer") or "").strip()
        sql_query = str(structured_result.get("sql_query") or "").strip()
        if answer or sql_query:
            context_parts.append(f"Structured evidence:\nSQL: {sql_query}\nResult: {answer}")

    incident_investigation = response_data.get("incident_investigation")
    if isinstance(incident_investigation, dict):
        summary = str(incident_investigation.get("investigation_summary") or "").strip()
        incidents = incident_investigation.get("matched_incidents") or []
        if summary or incidents:
            context_parts.append(f"Incident evidence:\n{summary}\nMatched incidents: {incidents}")

    return "\n\n".join(context_parts)


def _bool_score(value: bool) -> float:
    return 1.0 if value else 0.0


def _online_runtime_scores(response_data: dict[str, Any]) -> dict[str, tuple[float, str]]:
    success = bool(response_data.get("success")) and not response_data.get("error")
    answer_present = bool(str(response_data.get("answer") or "").strip())
    source_nodes = list(response_data.get("source_nodes") or [])
    citations = list(response_data.get("citations") or [])
    structured_result = response_data.get("structured_result")
    latency_seconds = float(response_data.get("latency_ms") or 0) / 1000.0
    escalation_flag = bool(response_data.get("escalation_flag"))
    severity = str(response_data.get("severity") or "").lower()
    has_critical_context = severity == "critical" or escalation_flag
    error_text = str(response_data.get("error") or "").lower()

    scores: dict[str, tuple[float, str]] = {
        "task_success_rate": (
            _bool_score(success and (answer_present or escalation_flag)),
            "Online proxy: response succeeded and produced either an answer or a valid escalation.",
        ),
        "p95_response_latency_seconds": (
            latency_seconds,
            "Per-request latency in seconds. Use Langfuse aggregation/P95 for the SLO.",
        ),
        "guardrail_effectiveness": (
            _bool_score(success or escalation_flag),
            "Online proxy: request completed safely or escalated instead of returning an unsafe result.",
        ),
        "unauthorized_data_access": (
            1.0 if "unauthorized data" in error_text or "forbidden table" in error_text else 0.0,
            "Violation count proxy for unauthorized data access in this trace.",
        ),
        "escalation_accuracy": (
            _bool_score(escalation_flag if has_critical_context else True),
            "Online proxy: critical/high-risk traces are escalated; non-critical traces are not penalized here.",
        ),
        "critical_escalation_recall": (
            _bool_score(escalation_flag) if has_critical_context else 1.0,
            "Online proxy: critical traces should always escalate.",
        ),
    }

    if source_nodes:
        scores["source_attribution_rate"] = (
            _bool_score(bool(citations)),
            "Document-backed response has at least one citation.",
        )
        scores["context_precision"] = (
            _bool_score(bool(source_nodes)),
            "Online proxy: retrieved context was returned. True precision requires judged/golden evaluation.",
        )
        scores["retrieval_recall_at_5"] = (
            _bool_score(bool(source_nodes[:5])),
            "Online proxy: evidence exists in top retrieved items. True recall@5 requires golden labels.",
        )

    if isinstance(structured_result, dict):
        sql_query = str(structured_result.get("sql_query") or "").strip().lower()
        sql_ok = bool(sql_query.startswith("select") or sql_query.startswith("with") or not sql_query)
        scores["sql_correctness"] = (
            _bool_score(sql_ok and not response_data.get("error")),
            "Online proxy: SQL result returned without error and exposed SQL is read-only.",
        )

    if response_data.get("route_decision"):
        scores["query_routing_accuracy"] = (
            1.0,
            "Route was selected. True routing accuracy requires golden-query labels.",
        )
    if response_data.get("intent"):
        scores["intent_classification_accuracy"] = (
            1.0,
            "Intent was selected. True intent accuracy requires golden-query labels.",
        )
    if response_data.get("severity"):
        scores["risk_classification_accuracy"] = (
            1.0,
            "Risk/severity was selected. True risk accuracy requires golden-query labels.",
        )

    return scores


def attach_runtime_slo_scores(
    trace_id: str,
    question: str,
    response_data: dict[str, Any],
) -> RuntimeSLOScoreResult:
    """Attach online runtime SLO scores and optional LLM judge scores to a trace."""

    trace_final_response(trace_id=trace_id, question=question, response_data=response_data)

    runtime_scores = _online_runtime_scores(response_data)
    attachments = {
        name: attach_score_to_trace(trace_id, name, value, comment)
        for name, (value, comment) in runtime_scores.items()
    }

    judge_result: EvaluationScoreResult | None = None
    answer = str(response_data.get("answer") or "")
    context = _response_context(response_data)
    if answer.strip() and settings.enable_llm_judge:
        judge_result = attach_evaluation_scores(trace_id, question, context, answer)

    _attach_slo_summary_scores(
        trace_id=trace_id,
        response_data=response_data,
        numeric_scores=runtime_scores,
        judge_scores=judge_result.scores if judge_result is not None else {},
    )

    return RuntimeSLOScoreResult(trace_id=trace_id, attachments=attachments, judge_result=judge_result)


def attach_evaluation_scores(
    trace_id: str,
    question: str,
    context: str,
    answer: str,
) -> EvaluationScoreResult:
    """Run LLM-as-a-Judge evaluators and attach their scores to the given Langfuse trace."""

    evaluations = {
        "faithfulness": evaluate_faithfulness(question, context, answer),
        "answer_relevance": evaluate_answer_relevance(question, answer),
        "llm_judge_quality": evaluate_overall_quality(question, context, answer),
    }
    attachments = {
        name: attach_score_to_trace(trace_id, name, evaluation.score, evaluation.reasoning)
        for name, evaluation in evaluations.items()
    }
    return EvaluationScoreResult(trace_id=trace_id, scores=evaluations, attachments=attachments)
