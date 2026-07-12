from __future__ import annotations

import pytest

from app.orchestration.severity_assessor import _assess_local
from app.orchestration.state import EscalationTarget, RouteDecision, SeverityPriority, SupportIntent


def _state(query: str, *, intent: SupportIntent = SupportIntent.UNKNOWN, route: RouteDecision = RouteDecision.RAG) -> dict:
    return {
        "query": query,
        "intent": intent,
        "route_decision": route,
        "metadata": {},
        "incident_investigation": {},
    }


def test_regression_breach_keyword_false_positive() -> None:
    result = _assess_local(
        _state("Has the Priority customer breached the SLA response commitment?", intent=SupportIntent.PERFORMANCE)
    )

    assert result.severity_priority != SeverityPriority.P0
    assert result.escalation_target != EscalationTarget.SECURITY_TEAM
    assert "security breach context" not in result.matched_indicators


@pytest.mark.parametrize(
    "query",
    [
        "The contract breach report mentions missed performance commitments, not a security event.",
        "Explain the word breach in the SLA policy examples.",
        "This is a response SLA breach for a priority customer ticket.",
    ],
)
def test_breach_language_without_security_context_is_not_p0(query: str) -> None:
    result = _assess_local(_state(query, intent=SupportIntent.PERFORMANCE))

    assert result.severity_priority != SeverityPriority.P0
    assert result.escalation_target != EscalationTarget.SECURITY_TEAM


@pytest.mark.parametrize(
    "query",
    [
        "Customer data breach detected in production.",
        "Unauthorized access breach exposed an API key.",
    ],
)
def test_genuine_security_breach_is_p0(query: str) -> None:
    result = _assess_local(_state(query, intent=SupportIntent.SECURITY, route=RouteDecision.HIGH_RISK))

    assert result.severity_priority == SeverityPriority.P0
    assert result.escalation_flag is True
    assert result.escalation_target == EscalationTarget.SECURITY_TEAM


def test_ambiguous_mixed_breach_signal_documents_current_behavior() -> None:
    result = _assess_local(
        _state("SLA breach with possible customer data exposure needs review.", intent=SupportIntent.SECURITY)
    )

    assert result.severity_priority != SeverityPriority.P0
    assert result.escalation_target != EscalationTarget.SECURITY_TEAM


@pytest.mark.parametrize("query", ["", "ok", "asdf qwer zxcv", "¿cómo configuro esto?", "障害 では ない"])
def test_empty_short_non_english_and_garbled_queries_do_not_throw(query: str) -> None:
    result = _assess_local(_state(query))

    assert result.severity_priority in {SeverityPriority.P1, SeverityPriority.P2, SeverityPriority.P3}
    assert result.severity_reason
