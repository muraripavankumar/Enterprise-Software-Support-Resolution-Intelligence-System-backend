import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import psycopg
from psycopg.rows import dict_row

from app.core.config import settings
from app.core.langfuse import observe, trace_agent_state, trace_guardrail_event
from app.orchestration.state import (
    AgentTraceEvent,
    CustomerContext,
    ExecutionResult,
    GuardrailFlag,
    ProgressUpdate,
    SeverityLevel,
    SupportOrchestrationState,
    VerificationOutcome,
)

logger = logging.getLogger(__name__)

AGENT_NAME = "account_validator_agent"


@dataclass(frozen=True)
class AccountLookup:
    customer_id: int | None = None
    company_name: str | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_list(state: SupportOrchestrationState, key: str, item: Any) -> None:
    values = list(state.get(key, []))
    values.append(item)
    state[key] = values  # type: ignore[literal-required]


def _metadata(state: SupportOrchestrationState) -> dict[str, Any]:
    return dict(state.get("metadata") or {})


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _extract_lookup(state: SupportOrchestrationState) -> AccountLookup:
    metadata = _metadata(state)
    existing_context = dict(metadata.get("customer_context") or {})
    query = str(state.get("query") or "")

    customer_id = (
        _coerce_int(metadata.get("customer_id"))
        or _coerce_int(existing_context.get("customer_id"))
        or _coerce_int(metadata.get("account_id"))
    )
    if customer_id is None:
        id_match = re.search(r"\b(?:customer|customer_id|account|account_id)\s*[:#-]?\s*(\d+)\b", query, re.IGNORECASE)
        if id_match:
            customer_id = _coerce_int(id_match.group(1))

    company_name = (
        _clean_text(metadata.get("company_name"))
        or _clean_text(metadata.get("customer_name"))
        or _clean_text(metadata.get("company"))
        or _clean_text(metadata.get("account_name"))
        or _clean_text(existing_context.get("company_name"))
    )
    if company_name is None:
        quoted_match = re.search(r"\b(?:customer|company|account)\s+['\"]([^'\"]+)['\"]", query, re.IGNORECASE)
        if quoted_match:
            company_name = _clean_text(quoted_match.group(1))

    return AccountLookup(customer_id=customer_id, company_name=company_name)


def _empty_context(status: str, reason: str) -> CustomerContext:
    return {
        "customer_id": None,
        "company_name": None,
        "sla_level": None,
        "subscription_tier": None,
        "account_status": None,
        "region": None,
        "account_suspended": False,
        "lookup_status": status,
        "lookup_reason": reason,
    }


def _context_from_row(row: dict[str, Any]) -> CustomerContext:
    account_status = _clean_text(row.get("account_status"))
    return {
        "customer_id": _coerce_int(row.get("customer_id")),
        "company_name": _clean_text(row.get("company_name")),
        "sla_level": _clean_text(row.get("sla_level")),
        "subscription_tier": _clean_text(row.get("subscription_tier")),
        "account_status": account_status,
        "region": _clean_text(row.get("region")),
        "account_suspended": bool(account_status and account_status.lower() == "suspended"),
        "lookup_status": "found",
        "lookup_reason": "Customer account was found in the customers table.",
    }


def _fetch_customer_context(lookup: AccountLookup) -> CustomerContext:
    if lookup.customer_id is None and not lookup.company_name:
        return _empty_context(
            status="missing_lookup",
            reason="No customer_id or company_name was available in state metadata or query text.",
        )

    with psycopg.connect(settings.database_dsn, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            if lookup.customer_id is not None:
                cur.execute(
                    """
                    SELECT customer_id, company_name, sla_level, subscription_tier, account_status, region
                    FROM customers
                    WHERE customer_id = %s
                    LIMIT 1
                    """,
                    (lookup.customer_id,),
                )
            else:
                cur.execute(
                    """
                    SELECT customer_id, company_name, sla_level, subscription_tier, account_status, region
                    FROM customers
                    WHERE company_name ILIKE %s
                    ORDER BY customer_id
                    LIMIT 1
                    """,
                    (lookup.company_name,),
                )
            row = cur.fetchone()

    if not row:
        identifier = f"customer_id={lookup.customer_id}" if lookup.customer_id is not None else f"company_name={lookup.company_name}"
        return _empty_context(status="not_found", reason=f"No customer account found for {identifier}.")
    return _context_from_row(dict(row))


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


def _guardrail(context: CustomerContext) -> GuardrailFlag:
    suspended = bool(context.get("account_suspended"))
    return {
        "name": "account_status_check",
        "passed": not suspended,
        "severity": SeverityLevel.HIGH if suspended else SeverityLevel.LOW,
        "reason": "Customer account is suspended." if suspended else "Customer account is not suspended or was not found.",
        "metadata": {
            "customer_id": context.get("customer_id"),
            "company_name": context.get("company_name"),
            "account_status": context.get("account_status"),
            "lookup_status": context.get("lookup_status"),
        },
    }


def _execution_result(context: CustomerContext, error: str | None = None) -> ExecutionResult:
    return {
        "step_id": "account-validation",
        "agent_name": AGENT_NAME,
        "result_type": "account_validation",
        "summary": (
            f"lookup_status={context.get('lookup_status')}; "
            f"customer_id={context.get('customer_id')}; "
            f"account_status={context.get('account_status') or 'unknown'}"
        ),
        "data": dict(context),
        "error": error,
        "timestamp": _utc_now(),
    }


def _verification(context: CustomerContext, error: str | None = None) -> VerificationOutcome:
    found = context.get("lookup_status") == "found"
    passed = error is None and found
    return {
        "check_name": "account_validation_complete",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": error or context.get("lookup_reason") or "Account validation completed.",
        "corrective_action": None if passed else "Provide a customer_id or company_name, or continue without account validation.",
        "metadata": {
            "lookup_status": context.get("lookup_status"),
            "account_suspended": context.get("account_suspended", False),
        },
    }


def _merge_customer_context_into_metadata(state: SupportOrchestrationState, context: CustomerContext) -> None:
    metadata = _metadata(state)
    metadata["customer_context"] = dict(context)
    state["metadata"] = metadata


@observe(name=AGENT_NAME, as_type="agent", capture_input=False, capture_output=False)
async def validate_account(state: SupportOrchestrationState) -> SupportOrchestrationState:
    """Validate customer account context directly from the customers table."""

    started = time.perf_counter()
    next_state: SupportOrchestrationState = dict(state)

    _append_list(next_state, "progress_updates", _progress("account-validation", "started", "Account validation started."))

    lookup = _extract_lookup(next_state)
    error: str | None = None
    status = "completed"
    try:
        context = _fetch_customer_context(lookup)
    except Exception as exc:
        logger.exception("Account validation failed")
        error = str(exc)
        status = "failed"
        context = _empty_context(status="error", reason=f"Account validation failed: {error}")
        _append_list(next_state, "errors", context["lookup_reason"])

    next_state["customer_context"] = context
    _merge_customer_context_into_metadata(next_state, context)

    if context.get("account_suspended"):
        _append_list(next_state, "errors", "Customer account is suspended.")

    latency_ms = int((time.perf_counter() - started) * 1000)
    output_summary = (
        f"lookup_status={context.get('lookup_status')}; "
        f"account_status={context.get('account_status') or 'unknown'}; "
        f"suspended={context.get('account_suspended', False)}"
    )

    guardrail = _guardrail(context)
    _append_list(next_state, "guardrail_flags", guardrail)
    trace_guardrail_event(
        name=str(guardrail["name"]),
        passed=bool(guardrail["passed"]),
        reason=str(guardrail["reason"]),
        metadata=dict(guardrail.get("metadata") or {}),
    )
    _append_list(next_state, "execution_results", _execution_result(context, error))
    _append_list(next_state, "verification_outcomes", _verification(context, error))
    _append_list(
        next_state,
        "agent_trace",
        _trace(
            action="validate_account",
            status=status,
            input_summary=f"customer_id={lookup.customer_id}; company_name={lookup.company_name or 'unknown'}",
            output_summary=output_summary,
            latency_ms=latency_ms,
        ),
    )
    _append_list(next_state, "progress_updates", _progress("account-validation", "completed", output_summary))

    trace_agent_state(
        agent_name=AGENT_NAME,
        input_state=state,
        output_state=next_state,
        started_at=started,
        tool_used="postgres_customers_table",
    )

    return next_state
