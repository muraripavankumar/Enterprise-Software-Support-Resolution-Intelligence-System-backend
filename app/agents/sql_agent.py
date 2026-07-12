import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.core.langfuse import observe, trace_agent_state, trace_guardrail_event
from app.orchestration.state import (
    AgentTraceEvent,
    EscalationTarget,
    ExecutionResult,
    GuardrailFlag,
    ProgressUpdate,
    SQLResult,
    SeverityLevel,
    SupportOrchestrationState,
    VerificationOutcome,
)
from app.services.tools.sql_tool import execute_nl2sql

logger = logging.getLogger(__name__)

AGENT_NAME = "sql_agent"

MUTATION_KEYWORDS = {
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "truncate",
    "create",
    "grant",
    "revoke",
    "merge",
    "call",
    "exec",
}


@dataclass(frozen=True)
class SQLValidationResult:
    status: str
    passed: bool
    reason: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_list(state: SupportOrchestrationState, key: str, item: Any) -> None:
    values = list(state.get(key, []))
    values.append(item)
    state[key] = values  # type: ignore[literal-required]


def _split_tables(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip() and str(item).strip() != "unknown"]
    return [part.strip() for part in str(value).split(",") if part.strip() and part.strip() != "unknown"]


def _row_count(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _raw_results(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _normalize_sql_result(result: dict[str, Any]) -> SQLResult:
    return {
        "answer": str(result.get("answer") or ""),
        "sql_query": str(result.get("sql_query") or ""),
        "tables_used": _split_tables(result.get("table_used") or result.get("tables_used")),
        "row_count": _row_count(result.get("row_count")),
        "raw_results": _raw_results(result.get("raw_results")),
        "error": str(result["error"]) if result.get("error") else None,
    }


def _strip_sql_comments(sql_query: str) -> str:
    without_line_comments = re.sub(r"--.*?$", "", sql_query, flags=re.MULTILINE)
    return re.sub(r"/\*.*?\*/", "", without_line_comments, flags=re.DOTALL)


def _validate_select_only(sql_query: str) -> SQLValidationResult:
    stripped = _strip_sql_comments(sql_query).strip()
    if not stripped:
        return SQLValidationResult(
            status="unknown",
            passed=True,
            reason="SQL query was not exposed by the SQL tool; validation status is unknown.",
        )

    statements = [segment.strip() for segment in stripped.split(";") if segment.strip()]
    if len(statements) > 1:
        return SQLValidationResult(
            status="failed",
            passed=False,
            reason="SQL validation failed because multiple statements were detected.",
        )

    lowered = stripped.lower()
    first_token_match = re.match(r"^\s*([a-zA-Z_]+)", lowered)
    first_token = first_token_match.group(1) if first_token_match else ""
    if first_token not in {"select", "with"}:
        return SQLValidationResult(
            status="failed",
            passed=False,
            reason=f"SQL validation failed because query starts with '{first_token or 'unknown'}' instead of SELECT/WITH.",
        )

    mutation_pattern = r"\b(" + "|".join(re.escape(keyword) for keyword in sorted(MUTATION_KEYWORDS)) + r")\b"
    mutation_match = re.search(mutation_pattern, lowered)
    if mutation_match:
        return SQLValidationResult(
            status="failed",
            passed=False,
            reason=f"SQL validation failed because mutation keyword '{mutation_match.group(1)}' was detected.",
        )

    return SQLValidationResult(
        status="passed",
        passed=True,
        reason="SQL validation passed: query is read-only SELECT/WITH.",
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


def _guardrail(validation: SQLValidationResult, sql_result: SQLResult) -> GuardrailFlag:
    return {
        "name": "sql_select_only",
        "passed": validation.passed,
        "severity": SeverityLevel.HIGH if not validation.passed else SeverityLevel.LOW,
        "reason": validation.reason,
        "metadata": {
            "validation_status": validation.status,
            "sql_query": sql_result.get("sql_query"),
            "tables_used": sql_result.get("tables_used", []),
        },
    }


def _execution_result(sql_result: SQLResult, validation: SQLValidationResult) -> ExecutionResult:
    return {
        "step_id": "sql-query",
        "agent_name": AGENT_NAME,
        "result_type": "structured_sql_query",
        "summary": (
            f"rows={sql_result.get('row_count', 0)}; validation={validation.status}; "
            f"tables={','.join(sql_result.get('tables_used', [])) or 'unknown'}"
        ),
        "data": {
            "answer": sql_result.get("answer"),
            "sql_query": sql_result.get("sql_query"),
            "tables_used": sql_result.get("tables_used", []),
            "row_count": sql_result.get("row_count", 0),
            "validation_status": validation.status,
        },
        "error": sql_result.get("error"),
        "timestamp": _utc_now(),
    }


def _verification(sql_result: SQLResult, validation: SQLValidationResult) -> VerificationOutcome:
    has_tool_error = bool(sql_result.get("error"))
    passed = validation.passed and not has_tool_error
    return {
        "check_name": "sql_query_validated",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": sql_result.get("error") or validation.reason,
        "corrective_action": None if passed else "Block SQL result for final answer or reroute to human review.",
        "metadata": {
            "validation_status": validation.status,
            "row_count": sql_result.get("row_count", 0),
        },
    }


@observe(name=AGENT_NAME, as_type="agent", capture_input=False, capture_output=False)
async def query_structured_data(state: SupportOrchestrationState) -> SupportOrchestrationState:
    """Run structured SQL retrieval and validate the exposed SQL query."""

    started = time.perf_counter()
    next_state: SupportOrchestrationState = dict(state)
    query = str(next_state.get("query") or "")
    metadata = dict(next_state.get("metadata") or {})

    _append_list(next_state, "progress_updates", _progress("sql-query", "started", "Structured SQL query started."))

    status = "completed"
    try:
        if metadata.get("inject_sql_timeout"):
            raise TimeoutError("Injected SQL timeout for fault-injection evaluation.")
        tool_result = await execute_nl2sql(query)
        sql_result = _normalize_sql_result(dict(tool_result))
    except Exception as exc:
        logger.exception("SQL agent failed")
        sql_result = {
            "answer": "Structured data retrieval failed.",
            "sql_query": "",
            "tables_used": [],
            "row_count": 0,
            "raw_results": [],
            "error": str(exc),
        }
        status = "failed"

    validation = _validate_select_only(str(sql_result.get("sql_query") or ""))
    next_state["sql_results"] = [sql_result]

    if sql_result.get("error"):
        status = "failed"
        _append_list(next_state, "errors", f"SQL retrieval failed: {sql_result['error']}")
        next_state["escalation_flag"] = True
        next_state["escalation_target"] = EscalationTarget.L2_SUPPORT
        next_state["escalation_reason"] = f"Structured data retrieval failed: {sql_result['error']}"

    if not validation.passed:
        status = "failed"
        _append_list(next_state, "errors", validation.reason)
        next_state["escalation_flag"] = True
        next_state["escalation_target"] = EscalationTarget.L2_SUPPORT
        next_state["escalation_reason"] = validation.reason

    latency_ms = int((time.perf_counter() - started) * 1000)
    output_summary = (
        f"rows={sql_result.get('row_count', 0)}; validation={validation.status}; status={status}"
    )

    _append_list(next_state, "guardrail_flags", _guardrail(validation, sql_result))
    trace_guardrail_event(
        name="sql_select_only",
        passed=validation.passed,
        reason=validation.reason,
        metadata={"validation_status": validation.status, "tables_used": sql_result.get("tables_used", [])},
    )
    _append_list(next_state, "execution_results", _execution_result(sql_result, validation))
    _append_list(next_state, "verification_outcomes", _verification(sql_result, validation))
    _append_list(
        next_state,
        "agent_trace",
        _trace(
            action="query_structured_data",
            status=status,
            input_summary=f"query_length={len(query)}",
            output_summary=output_summary,
            latency_ms=latency_ms,
        ),
    )
    _append_list(next_state, "progress_updates", _progress("sql-query", "completed", output_summary))

    trace_agent_state(
        agent_name=AGENT_NAME,
        input_state=state,
        output_state=next_state,
        started_at=started,
        tool_used="nl2sql",
    )

    return next_state
