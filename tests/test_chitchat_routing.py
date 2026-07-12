import unittest
from unittest.mock import patch

from app.orchestration.intent_classifier import ClassificationResult, classify_intent
from app.orchestration.severity_assessor import assess_severity
from app.orchestration.state import RouteDecision, SeverityLevel, SeverityPriority, SupportIntent


def _base_state(query: str) -> dict:
    return {
        "query": query,
        "metadata": {},
        "progress_updates": [],
        "execution_results": [],
        "verification_outcomes": [],
        "agent_trace": [],
        "errors": [],
        "guardrail_flags": [],
        "escalation_flag": False,
    }


class ChitchatRoutingTests(unittest.TestCase):
    def test_chitchat_short_circuits_classifier_and_llm(self) -> None:
        for query in ("hi", "hello", "thanks", "ok"):
            with self.subTest(query=query):
                with patch("app.orchestration.intent_classifier._classify_local") as local_classifier:
                    with patch("app.orchestration.intent_classifier._classify_with_llm") as llm_classifier:
                        state = classify_intent(_base_state(query))

                self.assertEqual(state["intent"], SupportIntent.CHITCHAT)
                self.assertEqual(state["route_decision"], RouteDecision.CHITCHAT)
                self.assertEqual(state["confidence_score"], 1.0)
                local_classifier.assert_not_called()
                llm_classifier.assert_not_called()

    def test_vague_real_query_is_clarification_not_chitchat(self) -> None:
        fallback = ClassificationResult(
            intent=SupportIntent.UNKNOWN,
            route_decision=RouteDecision.CLARIFICATION,
            confidence_score=0.35,
            reason="No strong local classification indicators were found.",
            classifier="local",
        )
        with patch("app.orchestration.intent_classifier._classify_with_llm", return_value=fallback):
            state = classify_intent(_base_state("it's broken"))

        self.assertEqual(state["intent"], SupportIntent.UNKNOWN)
        self.assertEqual(state["route_decision"], RouteDecision.CLARIFICATION)

    def test_sla_breach_routing_is_unaffected(self) -> None:
        state = classify_intent(_base_state("Has any Priority customer ticket breached the 4-hour response SLA?"))

        self.assertNotEqual(state["route_decision"], RouteDecision.CHITCHAT)
        self.assertIn(state["route_decision"], {RouteDecision.SQL, RouteDecision.HYBRID})

    def test_security_breach_severity_is_unaffected(self) -> None:
        state = _base_state("Security breach in deployed API")
        state["intent"] = SupportIntent.SECURITY
        state["route_decision"] = RouteDecision.HIGH_RISK

        assessed = assess_severity(state)

        self.assertEqual(assessed["severity_priority"], SeverityPriority.P0)
        self.assertEqual(assessed["severity"], SeverityLevel.CRITICAL)
        self.assertTrue(assessed["escalation_flag"])


if __name__ == "__main__":
    unittest.main()

