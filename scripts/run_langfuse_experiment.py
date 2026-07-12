"""Run the ERIS LangGraph dataset as a Langfuse experiment.

Prerequisites:
    1. Import dataset items first:
       python scripts/import_langfuse_dataset.py

    2. Run the experiment:
       python scripts/run_langfuse_experiment.py

Smoke test one path:
       python scripts/run_langfuse_experiment.py --path-filter chitchat --limit 1
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from dotenv import load_dotenv
from langfuse import Evaluation
from langfuse import Langfuse


BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent
DEFAULT_DATASET_NAME = "eris-software-support-path-coverage"
PASS_FAIL_EVALUATIONS = {
    "intent_match",
    "route_match",
    "severity_match",
    "escalation_match",
    "path_expected_behavior",
}
EXPECTED_ROUTE_BY_PATH = {
    "intent_classification.clean_single_intent_rag": "rag",
    "intent_classification.clean_single_intent_sql": "sql",
    "intent_classification.clean_single_intent_high_risk": "high_risk",
    "intent_classification.clean_single_intent_chitchat": "chitchat",
    "intent_classification.ambiguous_intent_clarification": "clarification",
    "intent_classification.out_of_scope_unsupported_fallback": "clarification",
    "documentation_retrieval.single_document_bm25_dense_agree": "rag",
    "documentation_retrieval.multi_hop_rrf_across_docs": "rag",
    "documentation_retrieval.no_relevant_document_low_confidence": "rag",
    "documentation_retrieval.reranker_demotes_high_bm25_irrelevant": "rag",
    "account_validation.valid_allowlisted_schema_returns_rows": "sql",
    "account_validation.rejected_non_allowlisted_table_or_column": "high_risk",
    "account_validation.ambiguous_natural_language_needs_clarification": "clarification",
    "account_validation.valid_sql_zero_rows_account_not_found": "sql",
    "severity_assessment.true_p0_security_breach": "high_risk",
    "severity_assessment.true_p0_production_outage": "high_risk",
    "severity_assessment.false_positive_trigger_words_not_p0": "hybrid",
    "severity_assessment.p1_high_impact": "high_risk",
    "severity_assessment.p2_functional_support": "hybrid",
    "severity_assessment.p3_general_documentation": "rag",
    "severity_assessment.borderline_compound_matching_guardrail": "rag",
    "escalation_manager.high_confidence_auto_escalates_without_interrupt": "high_risk",
    "escalation_manager.low_confidence_interrupt_human_in_loop": "high_risk",
    "escalation_manager.interrupt_resumes_after_human_input_async_postgres_saver": "high_risk",
    "escalation_manager.bounded_retries_exhausted_forced_fallback": "rag",
    "cross_cutting.redis_cache_hit_short_ttl_sql_repeat": "sql",
    "cross_cutting.redis_cache_hit_long_ttl_rag_repeat": "rag",
    "cross_cutting.plan_act_check_reflect_retry_success": "rag",
    "cross_cutting.agent_timeout_failure_supervisor_graceful_fallback": "sql",
    "cross_cutting.hybrid_parallel_evidence_merge_success": "hybrid",
}

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def _load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(BACKEND_ROOT / ".env")


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _client_from_env() -> Langfuse:
    public_key = _require_env("LANGFUSE_PUBLIC_KEY")
    secret_key = _require_env("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST") or "https://cloud.langfuse.com"
    return Langfuse(public_key=public_key, secret_key=secret_key, host=host)


def _model_dump(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump())
    if hasattr(value, "dict"):
        return dict(value.dict())
    return {}


def _enum_value(value: Any) -> str | None:
    if value is None:
        return None
    return getattr(value, "value", str(value))


def _normalize_priority(value: Any) -> str | None:
    text = str(_enum_value(value) or "").strip()
    return text.upper() if text else None


def _dataset_item_input(item: Any) -> dict[str, Any]:
    return _model_dump(getattr(item, "input", None))


def _dataset_item_expected(item: Any) -> dict[str, Any]:
    return _model_dump(getattr(item, "expected_output", None))


def _dataset_item_metadata(item: Any) -> dict[str, Any]:
    return _model_dump(getattr(item, "metadata", None))


async def eris_langgraph_task(*, item: Any, **_: Any) -> dict[str, Any]:
    """Run one Langfuse DatasetItem through the actual ERIS LangGraph."""

    from app.agents.graph import run_support_graph

    item_input = _dataset_item_input(item)
    item_metadata = _dataset_item_metadata(item)
    ticket_text = str(item_input.get("ticket_text") or "").strip()
    path_tested = str(item_metadata.get("path_tested") or "")
    metadata = {
        "dataset_item_id": getattr(item, "id", None),
        "dataset_path_tested": path_tested,
        "customer_tier": item_input.get("customer_tier"),
        "prior_context": item_input.get("prior_context"),
        "thread_id": f"langfuse-exp-{path_tested or getattr(item, 'id', 'item')}-{uuid4()}",
    }
    if path_tested == "cross_cutting.agent_timeout_failure_supervisor_graceful_fallback":
        metadata["inject_sql_timeout"] = True
    if path_tested == "escalation_manager.interrupt_resumes_after_human_input_async_postgres_saver":
        metadata["human_handoff"] = {
            "needs_engineering": True,
            "confirmed_severity": "High",
            "jira_issue_type": "Bug",
            "source": "langfuse_dataset_fault_injection",
        }
    metadata = {key: value for key, value in metadata.items() if value is not None}

    try:
        state = await run_support_graph(ticket_text, metadata=metadata)
    except Exception as exc:
        return {
            "path_tested": path_tested,
            "answer": None,
            "intent": None,
            "route_decision": None,
            "severity_priority": None,
            "severity": None,
            "escalate": False,
            "escalation_target": None,
            "confidence_score": None,
            "latency_ms": None,
            "errors": [f"{type(exc).__name__}: {exc}"],
            "agent_trace": [],
            "task_error": f"{type(exc).__name__}: {exc}",
        }

    route_decision = _enum_value(state.get("route_decision"))
    severity_priority = _normalize_priority(state.get("severity_priority"))
    if not severity_priority and route_decision in {"chitchat", "clarification"}:
        severity_priority = "P3"

    return {
        "path_tested": path_tested,
        "answer": state.get("final_answer"),
        "intent": _enum_value(state.get("intent")),
        "route_decision": route_decision,
        "severity_priority": severity_priority,
        "severity": _enum_value(state.get("severity")),
        "escalate": bool(state.get("escalation_flag")),
        "escalation_target": _enum_value(state.get("escalation_target")),
        "confidence_score": state.get("confidence_score"),
        "latency_ms": state.get("latency_ms"),
        "errors": list(state.get("errors") or []),
        "agent_trace": list(state.get("agent_trace") or [])[-20:],
    }


def intent_match_evaluator(*, output: Any, expected_output: Any = None, **_: Any) -> Evaluation:
    output_payload = _model_dump(output)
    expected = _model_dump(expected_output)
    actual_intent = str(output_payload.get("intent") or "")
    expected_intent = str(expected.get("intent") or "")
    passed = actual_intent == expected_intent
    return Evaluation(
        name="intent_match",
        value=1.0 if passed else 0.0,
        comment=f"actual={actual_intent}; expected={expected_intent}",
    )


def _expected_route_from_payload(output_payload: dict[str, Any], expected: dict[str, Any]) -> str:
    expected_route = str(expected.get("route_decision") or "").strip()
    if expected_route:
        return expected_route
    path_tested = str(output_payload.get("path_tested") or "").strip()
    return EXPECTED_ROUTE_BY_PATH.get(path_tested, "")


def route_match_evaluator(*, output: Any, expected_output: Any = None, **_: Any) -> Evaluation:
    output_payload = _model_dump(output)
    expected = _model_dump(expected_output)
    actual_route = str(output_payload.get("route_decision") or "")
    expected_route = _expected_route_from_payload(output_payload, expected)
    passed = actual_route == expected_route
    comment = f"actual={actual_route}; expected={expected_route}"
    if not expected_route:
        comment = "expected route missing from dataset item and fallback path map"
    return Evaluation(
        name="route_match",
        value=1.0 if passed else 0.0,
        comment=comment,
    )


def severity_match_evaluator(*, output: Any, expected_output: Any = None, **_: Any) -> Evaluation:
    output_payload = _model_dump(output)
    expected = _model_dump(expected_output)
    actual_severity = str(output_payload.get("severity_priority") or "").upper()
    if not actual_severity:
        severity_level = str(output_payload.get("severity") or "").lower()
        actual_severity = {
            "critical": "P0",
            "high": "P1",
            "medium": "P2",
            "low": "P3",
        }.get(severity_level, "")
    if not actual_severity and output_payload.get("route_decision") in {"chitchat", "clarification"}:
        actual_severity = "P3"
    expected_severity = str(expected.get("severity") or "").upper()
    passed = actual_severity == expected_severity
    return Evaluation(
        name="severity_match",
        value=1.0 if passed else 0.0,
        comment=f"actual={actual_severity}; expected={expected_severity}",
    )


def escalation_match_evaluator(*, output: Any, expected_output: Any = None, **_: Any) -> Evaluation:
    output_payload = _model_dump(output)
    expected = _model_dump(expected_output)
    actual_escalate = bool(output_payload.get("escalate"))
    expected_escalate = bool(expected.get("escalate"))
    passed = actual_escalate == expected_escalate
    return Evaluation(
        name="escalation_match",
        value=1.0 if passed else 0.0,
        comment=f"actual={actual_escalate}; expected={expected_escalate}",
    )


def path_pass_evaluator(*, output: Any, expected_output: Any = None, **kwargs: Any) -> Evaluation:
    intent_score = float(intent_match_evaluator(output=output, expected_output=expected_output, **kwargs).value)
    route_score = float(route_match_evaluator(output=output, expected_output=expected_output, **kwargs).value)
    severity_score = float(severity_match_evaluator(output=output, expected_output=expected_output, **kwargs).value)
    escalation_score = float(escalation_match_evaluator(output=output, expected_output=expected_output, **kwargs).value)
    passed = route_score == 1.0 and intent_score == 1.0 and severity_score == 1.0 and escalation_score == 1.0
    return Evaluation(
        name="path_expected_behavior",
        value=1.0 if passed else 0.0,
        comment="Intent, route, severity, and escalation all matched." if passed else "One or more path expectations failed.",
        metadata={
            "intent_match": intent_score,
            "route_match": route_score,
            "severity_match": severity_score,
            "escalation_match": escalation_score,
        },
    )


def latency_seconds_evaluator(*, output: Any, **_: Any) -> Evaluation:
    output_payload = _model_dump(output)
    latency_ms = output_payload.get("latency_ms")
    value = float(latency_ms or 0) / 1000.0
    return Evaluation(
        name="latency_seconds",
        value=value,
        comment=f"{value:.3f}s",
    )


def aggregate_path_pass_rate(*, item_results: list[Any], **_: Any) -> Evaluation:
    values: list[float] = []
    for result in item_results:
        for evaluation in getattr(result, "evaluations", []) or []:
            if getattr(evaluation, "name", None) == "path_expected_behavior":
                values.append(float(getattr(evaluation, "value", 0.0) or 0.0))
    pass_rate = sum(values) / len(values) if values else 0.0
    return Evaluation(
        name="path_pass_rate",
        value=pass_rate,
        comment=f"{sum(1 for value in values if value == 1.0)}/{len(values)} items matched intent/route/severity/escalation expectations.",
    )


def _filter_items(items: list[Any], path_filter: str | None, limit: int | None) -> list[Any]:
    selected = items
    if path_filter:
        needle = path_filter.lower()
        selected = [
            item
            for item in selected
            if needle in str(_dataset_item_metadata(item).get("path_tested") or "").lower()
        ]
    if limit is not None:
        selected = selected[:limit]
    return selected


def _evaluation_value(evaluation: Any) -> float | None:
    try:
        return float(getattr(evaluation, "value", None))
    except (TypeError, ValueError):
        return None


def _failed_evaluations(evaluations: list[Any]) -> list[Any]:
    failed: list[Any] = []
    for evaluation in evaluations:
        name = str(getattr(evaluation, "name", "") or "")
        if name not in PASS_FAIL_EVALUATIONS:
            continue
        value = _evaluation_value(evaluation)
        if value is None or value < 1.0:
            failed.append(evaluation)
    return failed


def _expected_snapshot(item: Any) -> str:
    expected = _dataset_item_expected(item)
    return (
        f"intent={expected.get('intent') or 'n/a'}; "
        f"route={expected.get('route_decision') or 'n/a'}; "
        f"severity={expected.get('severity') or 'n/a'}; "
        f"escalate={expected.get('escalate')}"
    )


def _actual_snapshot(output: Any) -> str:
    payload = _model_dump(output)
    severity = payload.get("severity_priority") or payload.get("severity") or "n/a"
    latency_ms = payload.get("latency_ms")
    latency_text = f"{latency_ms}ms" if latency_ms is not None else "n/a"
    return (
        f"intent={payload.get('intent') or 'n/a'}; "
        f"route={payload.get('route_decision') or 'n/a'}; "
        f"severity={severity}; "
        f"escalate={payload.get('escalate')}; "
        f"target={payload.get('escalation_target') or 'n/a'}; "
        f"latency={latency_text}"
    )


def _item_errors(output: Any) -> list[str]:
    payload = _model_dump(output)
    errors = [str(error) for error in payload.get("errors") or [] if error]
    task_error = payload.get("task_error")
    if task_error and str(task_error) not in errors:
        errors.insert(0, str(task_error))
    return errors


def _print_item_level_report(result: Any, *, show_all: bool) -> None:
    item_results = list(getattr(result, "item_results", []) or [])
    if not item_results:
        print("\nItem-level results: none returned by Langfuse.")
        return

    visible_count = 0
    expectation_failed_count = 0
    runtime_warning_count = 0
    print("\nItem-level path report:")
    for index, item_result in enumerate(item_results, start=1):
        item = getattr(item_result, "item", None)
        output = getattr(item_result, "output", None)
        evaluations = list(getattr(item_result, "evaluations", []) or [])
        metadata = _dataset_item_metadata(item)
        path_tested = str(metadata.get("path_tested") or f"item_{index}")
        failed_evals = _failed_evaluations(evaluations)
        errors = _item_errors(output)
        expectation_failed = bool(failed_evals)
        runtime_warning = bool(errors) and not expectation_failed

        if expectation_failed:
            expectation_failed_count += 1
        if runtime_warning:
            runtime_warning_count += 1
        if not show_all and not expectation_failed and not runtime_warning:
            continue

        visible_count += 1
        status = "FAIL" if expectation_failed else "WARN" if runtime_warning else "PASS"
        print(f"\n[{status}] {index:02d}. {path_tested}")
        print(f"  Expected: {_expected_snapshot(item)}")
        print(f"  Actual:   {_actual_snapshot(output)}")

        if failed_evals:
            print("  Failed checks:")
            for evaluation in failed_evals:
                name = str(getattr(evaluation, "name", "") or "unknown")
                value = getattr(evaluation, "value", None)
                comment = str(getattr(evaluation, "comment", "") or "").strip()
                suffix = f" - {comment}" if comment else ""
                print(f"    - {name}: {value}{suffix}")

        if errors:
            print("  Runtime errors:")
            for error in errors[:5]:
                print(f"    - {error}")
            if len(errors) > 5:
                print(f"    - ... {len(errors) - 5} more error(s)")

    if visible_count == 0:
        print("  No failing items. Use --show-items to print passing items too.")
    print(
        "\nItem-level summary: "
        f"{expectation_failed_count}/{len(item_results)} item(s) failed pass/fail checks; "
        f"{runtime_warning_count} item(s) had runtime warnings with matching expectations."
    )


def _print_experiment_summary(result: Any, *, run_name: str, experiment_name: str) -> None:
    item_results = list(getattr(result, "item_results", []) or [])
    score_values: dict[str, list[float]] = {}
    for item_result in item_results:
        for evaluation in getattr(item_result, "evaluations", []) or []:
            value = _evaluation_value(evaluation)
            if value is None:
                continue
            score_values.setdefault(str(getattr(evaluation, "name", "") or "unknown"), []).append(value)

    print("\nExperiment summary:")
    print(f"  Experiment: {experiment_name}")
    print(f"  Run name: {run_name}")
    print(f"  Items: {len(item_results)}")
    if score_values:
        print("  Average scores:")
        for name in sorted(score_values):
            values = score_values[name]
            average = sum(values) / len(values) if values else 0.0
            print(f"    - {name}: {average:.3f}")

    path_values = score_values.get("path_expected_behavior", [])
    if path_values:
        passed = sum(1 for value in path_values if value == 1.0)
        print(
            "  Path pass rate: "
            f"{passed}/{len(path_values)} matched intent/route/severity/escalation expectations."
        )

    dataset_run_url = getattr(result, "dataset_run_url", None)
    if dataset_run_url:
        print(f"  Dataset run: {dataset_run_url}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ERIS Langfuse dataset experiment.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--experiment-name", default="ERIS LangGraph path coverage")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--path-filter", default=None, help="Only run dataset items whose path_tested contains this text.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of dataset items for a smoke run.")
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--show-items", action="store_true", help="Print every item result, not only failures.")
    parser.add_argument(
        "--allow-jira-side-effects",
        action="store_true",
        help="Allow real Jira MCP create/link actions during the experiment. Default is disabled.",
    )
    return parser.parse_args()


def main() -> int:
    _load_env()
    args = parse_args()
    if not args.allow_jira_side_effects:
        os.environ["ENABLE_JIRA_MCP"] = "false"
    langfuse = _client_from_env()
    dataset = langfuse.get_dataset(args.dataset_name)
    items = _filter_items(list(dataset.items), args.path_filter, args.limit)
    if not items:
        raise RuntimeError("No dataset items matched the requested filter.")

    run_name = args.run_name or f"eris-path-coverage-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    print(f"Running Langfuse experiment: dataset={args.dataset_name}, run={run_name}, items={len(items)}")

    try:
        result = langfuse.run_experiment(
            name=args.experiment_name,
            run_name=run_name,
            description="Runs ERIS LangGraph orchestration over the path-coverage support dataset.",
            data=items,
            task=eris_langgraph_task,
            evaluators=[
                intent_match_evaluator,
                route_match_evaluator,
                severity_match_evaluator,
                escalation_match_evaluator,
                path_pass_evaluator,
                latency_seconds_evaluator,
            ],
            run_evaluators=[aggregate_path_pass_rate],
            max_concurrency=args.max_concurrency,
            metadata={
                "project": "Enterprise Software Support & Resolution Intelligence System",
                "dataset_name": args.dataset_name,
            },
        )
        _print_experiment_summary(result, run_name=run_name, experiment_name=args.experiment_name)
        _print_item_level_report(result, show_all=args.show_items)
        if result.dataset_run_url:
            print(f"Open results in Langfuse: {result.dataset_run_url}")
    finally:
        flush = getattr(langfuse, "flush", None)
        if callable(flush):
            flush()
        try:
            from app.agents.graph import close_support_graph_checkpointer

            asyncio.run(close_support_graph_checkpointer())
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Experiment failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
