from __future__ import annotations

import asyncio
from unittest.mock import patch

from app.agents import account_validator_agent
from app.agents.account_validator_agent import AccountLookup, _fetch_customer_context, validate_account
from app.agents.graph import (
    _needs_incident_investigation,
    _reflection_check,
    _route_after_account,
    _route_after_high_risk_parallel,
    _route_after_hybrid_parallel,
    _route_after_reflection,
)
from app.orchestration.state import RouteDecision
from app.services.tools import jira_mcp_tool


def _validation_outcome(passed: bool, *, evidence: bool = True, confidence: bool = True, citations: bool = True) -> dict:
    return {
        "check_name": "response_validation_complete",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": "ok" if passed else "failed",
        "metadata": {
            "retrieved_chunks_present": evidence,
            "confidence_passed": confidence,
            "citations_present": citations,
        },
    }


def test_reflection_success_routes_to_composer() -> None:
    state = {"metadata": {}, "verification_outcomes": [_validation_outcome(True)], "escalation_flag": False}

    reflected = _reflection_check(state)

    assert _route_after_reflection(reflected) == "response_composer"


def test_reflection_failure_triggers_retry_target_before_max_iterations() -> None:
    state = {
        "metadata": {},
        "route_decision": RouteDecision.RAG,
        "verification_outcomes": [_validation_outcome(False, evidence=False)],
        "iteration_count": 1,
        "max_iterations": 2,
        "iteration_history": [],
        "escalation_flag": False,
    }

    reflected = _reflection_check(state)

    assert _route_after_reflection(reflected) == "document_retrieval"
    assert reflected["iteration_count"] == 2
    assert reflected["iteration_history"][-1]["should_retry"] is True


def test_reflection_high_risk_retry_uses_parallel_evidence_path() -> None:
    state = {
        "metadata": {},
        "route_decision": RouteDecision.HIGH_RISK,
        "verification_outcomes": [_validation_outcome(False, evidence=False)],
        "iteration_count": 1,
        "max_iterations": 2,
        "iteration_history": [],
        "escalation_flag": False,
    }

    reflected = _reflection_check(state)

    assert _route_after_reflection(reflected) == "high_risk_parallel_tools"


def test_reflection_failure_after_max_iterations_forces_escalation() -> None:
    state = {
        "metadata": {},
        "route_decision": RouteDecision.RAG,
        "verification_outcomes": [_validation_outcome(False, evidence=False)],
        "iteration_count": 2,
        "max_iterations": 2,
        "iteration_history": [],
        "escalation_flag": False,
    }

    reflected = _reflection_check(state)

    assert _route_after_reflection(reflected) == "escalation_manager"
    assert reflected["escalation_flag"] is True
    assert reflected["iteration_history"][-1]["should_retry"] is False


def test_hybrid_route_skips_incident_investigator_without_incident_signals() -> None:
    state = {
        "query": "Customer Beta Systems is getting 401 errors. Check account and suggest fix.",
        "route_decision": RouteDecision.HYBRID,
        "verification_outcomes": [],
        "metadata": {},
    }

    assert _needs_incident_investigation(state) is False
    assert _route_after_hybrid_parallel(state) == "severity_assessor"


def test_hybrid_route_uses_incident_investigator_for_incident_signals() -> None:
    state = {
        "query": "Customer Beta Systems has an active production outage and 401 errors.",
        "route_decision": RouteDecision.HYBRID,
        "verification_outcomes": [],
        "metadata": {},
    }

    assert _needs_incident_investigation(state) is True
    assert _route_after_hybrid_parallel(state) == "incident_investigator"


def test_high_risk_route_uses_parallel_evidence_and_then_severity() -> None:
    state = {"route_decision": RouteDecision.HIGH_RISK, "verification_outcomes": []}

    assert _route_after_account(state) == "high_risk_parallel_tools"
    assert _route_after_high_risk_parallel(state) == "severity_assessor"


def test_account_validator_valid_mocked_db_response() -> None:
    context = {
        "customer_id": 7,
        "company_name": "Beta Systems",
        "sla_level": "Priority",
        "subscription_tier": "Enterprise",
        "account_status": "Active",
        "region": "US",
        "account_suspended": False,
        "lookup_status": "found",
        "lookup_reason": "Customer account was found in the customers table.",
    }
    state = {"query": "Check account 7", "metadata": {"customer_id": 7}, "progress_updates": [], "execution_results": [], "verification_outcomes": [], "agent_trace": [], "guardrail_flags": [], "errors": []}

    with patch.object(account_validator_agent, "_fetch_customer_context", return_value=context), patch(
        "app.agents.account_validator_agent.trace_agent_state"
    ), patch("app.agents.account_validator_agent.trace_guardrail_event"):
        result = asyncio.run(validate_account(state))

    assert result["customer_context"]["lookup_status"] == "found"
    assert result["customer_context"]["company_name"] == "Beta Systems"
    assert result["verification_outcomes"][-1]["passed"] is True


def test_account_validator_nonexistent_account_is_not_found() -> None:
    context = {
        "customer_id": None,
        "company_name": None,
        "sla_level": None,
        "subscription_tier": None,
        "account_status": None,
        "region": None,
        "account_suspended": False,
        "lookup_status": "not_found",
        "lookup_reason": "No customer account found for customer_id=999.",
    }
    state = {"query": "Check account 999", "metadata": {"customer_id": 999}, "progress_updates": [], "execution_results": [], "verification_outcomes": [], "agent_trace": [], "guardrail_flags": [], "errors": []}

    with patch.object(account_validator_agent, "_fetch_customer_context", return_value=context), patch(
        "app.agents.account_validator_agent.trace_agent_state"
    ), patch("app.agents.account_validator_agent.trace_guardrail_event"):
        result = asyncio.run(validate_account(state))

    assert result["customer_context"]["lookup_status"] == "not_found"
    assert result["verification_outcomes"][-1]["passed"] is False


def test_account_validator_company_lookup_uses_parameterized_sql() -> None:
    captured: dict[str, object] = {}

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def execute(self, sql_query, params):
            captured["sql_query"] = str(sql_query)
            captured["params"] = params

        def fetchone(self):
            return None

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def cursor(self):
            return FakeCursor()

    injection_like = "Acme'; DROP TABLE customers; --"
    with patch("app.agents.account_validator_agent.psycopg.connect", return_value=FakeConnection()):
        result = _fetch_customer_context(AccountLookup(company_name=injection_like))

    assert result["lookup_status"] == "not_found"
    assert "ILIKE %s" in str(captured["sql_query"])
    assert captured["params"] == (injection_like,)


def _jira_state(query: str) -> dict:
    return {
        "query": query,
        "severity_priority": "p0",
        "severity": "critical",
        "escalation_reason": "P0 detected due to critical indicators: data loss",
        "metadata": {},
    }


def test_jira_existing_matching_issue_is_updated_not_created() -> None:
    async def fake_search(_mapping):
        return "KAN-7", 'project = "KAN"'

    async def fake_add_comment(issue_key, _body):
        calls["commented"] = issue_key

    async def fake_create(*_args):
        raise AssertionError("create should not be called when a dedupe match exists")

    calls: dict[str, str] = {}
    mapping = jira_mcp_tool.JiraIssueMapping(
        project_key="KAN",
        issue_type="Bug",
        priority="Highest",
        reason_code="data_loss",
        dedupe_key="abc",
        dedupe_label="eris-dedupe-abc",
        labels=["eris-autocreated", "eris-dedupe-abc"],
    )
    existing = {"enabled": True, "attempted": True, "should_create": True}

    with patch.object(jira_mcp_tool, "_rest_search_existing", fake_search), patch.object(
        jira_mcp_tool, "_rest_add_comment", fake_add_comment
    ), patch.object(jira_mcp_tool, "_rest_create_issue", fake_create):
        result = asyncio.run(jira_mcp_tool._track_escalation_via_rest(_jira_state("data loss"), {}, mapping, existing))

    assert result["duplicate_found"] is True
    assert result["issue_key"] == "KAN-7"
    assert calls["commented"] == "KAN-7"


def test_jira_no_matching_issue_creates_new_ticket_with_payload_mapping() -> None:
    async def fake_search(_mapping):
        return None, 'project = "KAN"'

    async def fake_add_comment(issue_key, _body):
        calls.setdefault("comments", []).append(issue_key)

    async def fake_create(_state, mapping):
        calls["project_key"] = mapping.project_key
        calls["labels"] = mapping.labels
        return "KAN-8"

    async def fake_transition(issue_key):
        calls["transition"] = issue_key
        return True, "Triage"

    calls: dict[str, object] = {}
    mapping = jira_mcp_tool.JiraIssueMapping(
        project_key="KAN",
        issue_type="Bug",
        priority="Highest",
        reason_code="data_loss",
        dedupe_key="xyz",
        dedupe_label="eris-dedupe-xyz",
        labels=["eris-autocreated", "eris-data_loss", "eris-dedupe-xyz"],
    )
    existing = {"enabled": True, "attempted": True, "should_create": True}

    with patch.object(jira_mcp_tool, "_rest_search_existing", fake_search), patch.object(
        jira_mcp_tool, "_rest_add_comment", fake_add_comment
    ), patch.object(jira_mcp_tool, "_rest_create_issue", fake_create), patch.object(
        jira_mcp_tool, "_rest_transition_to_triage", fake_transition
    ):
        result = asyncio.run(jira_mcp_tool._track_escalation_via_rest(_jira_state("new data loss"), {}, mapping, existing))

    assert result["duplicate_found"] is False
    assert result["status"] == "created"
    assert result["issue_key"] == "KAN-8"
    assert calls["project_key"] == "KAN"
    assert "eris-dedupe-xyz" in calls["labels"]
