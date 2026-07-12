from __future__ import annotations

from unittest.mock import patch

import pytest

from app.agents.response_validator_agent import validate_response
from app.orchestration.state import EscalationTarget, RouteDecision


def _state(confidence: float | None) -> dict:
    state = {
        "query": "List open tickets.",
        "route_decision": RouteDecision.SQL,
        "confidence_score": confidence,
        "sql_results": [{"answer": "1 row", "row_count": 1, "raw_results": [{"ticket_id": 1}]}],
        "customer_context": {},
        "incident_investigation": {},
        "metadata": {},
        "retrieved_chunks": [],
        "citations": [],
        "progress_updates": [],
        "execution_results": [],
        "verification_outcomes": [],
        "agent_trace": [],
        "guardrail_flags": [],
        "errors": [],
        "escalation_flag": False,
    }
    if confidence is None:
        state.pop("confidence_score")
    return state


@pytest.mark.parametrize(
    ("confidence", "passes"),
    [
        (0.69, False),
        (0.70, True),
        (0.71, True),
        (0.0, False),
        (1.0, True),
        (None, False),
    ],
)
def test_confidence_threshold_boundary_behavior(confidence: float | None, passes: bool) -> None:
    with patch("app.agents.response_validator_agent.trace_agent_state"), patch(
        "app.agents.response_validator_agent.trace_guardrail_event"
    ):
        result = validate_response(_state(confidence))

    validation = result["verification_outcomes"][-1]
    confidence_guardrail = [
        flag for flag in result["guardrail_flags"] if flag["name"] == "confidence_threshold_check"
    ][-1]

    assert confidence_guardrail["passed"] is passes
    assert validation["metadata"]["confidence_passed"] is passes
    if passes:
        assert result["escalation_flag"] is False
    else:
        assert result["escalation_flag"] is True
        assert result["escalation_target"] == EscalationTarget.L2_SUPPORT
        assert "confidence_score" in result["escalation_reason"]
