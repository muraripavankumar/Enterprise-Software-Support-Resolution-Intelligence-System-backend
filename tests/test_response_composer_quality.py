import unittest

from app.agents.response_composer_agent import clean_final_answer, compose_response
from app.orchestration.intent_classifier import _classify_local
from app.orchestration.state import RouteDecision


def _base_state(query: str, route: RouteDecision) -> dict:
    return {
        "query": query,
        "route_decision": route,
        "metadata": {"response_style": "concise"},
        "retrieved_chunks": [],
        "sql_results": [],
        "customer_context": {},
        "incident_investigation": {},
        "recommended_actions": [],
        "progress_updates": [],
        "execution_results": [],
        "verification_outcomes": [],
        "agent_trace": [],
        "errors": [],
        "guardrail_flags": [],
        "escalation_flag": False,
    }


def _rag_state(query: str, chunk_text: str, source_file: str = "Support_Document.pdf") -> dict:
    state = _base_state(query, RouteDecision.RAG)
    state["retrieved_chunks"] = [
        {
            "chunk_text": chunk_text,
            "source_file": source_file,
            "page_number": 1,
            "score": 0.93,
        }
    ]
    return state


class ResponseComposerQualityTests(unittest.TestCase):
    def test_rag_answer_is_concise_and_does_not_dump_raw_chunks(self) -> None:
        state = _base_state("How do I use API authentication?", RouteDecision.RAG)
        state["retrieved_chunks"] = [
            {
                "chunk_text": (
                    "# API Integration & Authentication Guide # Generate API Key: POST /v3/auth/api-keys "
                    '{"name":"Production Key","scopes":["read:tickets"],"client_secret":"secret-value"} '
                    "# Use API Key in Requests: curl -H 'X-API-Key: sk_live_1234567890' https://api.example.com/v3/tickets "
                    "- Keys are scoped to specific permissions and can be rotated. "
                    "- Store keys securely in environment variables or a secrets manager. "
                    "# OAuth 2.0 Authorization Flow # Step 1: Redirect User to Authorization URL GET "
                    "https://auth.example.com/oauth/authorize?client_id=abc "
                    "# Step 2: Exchange Authorization Code for Access Token POST https://auth.example.com/oauth/token "
                    '{"access_token":"eyJhbGciOiFakeToken","refresh_token":"secret-refresh"} '
                    "# Rate Limiting Rules # Rate Limit Headers: X-RateLimit-Limit and X-RateLimit-Remaining."
                ),
                "source_file": "API_Integration_Authentication_Guide.pdf",
                "page_number": 1,
                "score": 0.82,
            }
        ]

        result = compose_response(state)
        answer = result["final_answer"]
        quality = result["metadata"]["answer_quality"]

        self.assertIn("Answer:", answer)
        self.assertIn("Recommended actions:", answer)
        self.assertIn("Sources:", answer)
        self.assertIn("[1] API Integration Authentication Guide", answer)
        self.assertIn("credential management", answer)
        self.assertGreaterEqual(quality["overall_quality_score"], 0.75)
        self.assertEqual(quality["evaluation_status"], "completed")
        self.assertLessEqual(len(answer.split()), 180)
        self.assertNotIn("curl -H", answer)
        self.assertNotIn("https://auth.example.com", answer)
        self.assertNotIn("client_secret", answer)
        self.assertNotIn("sk_live_", answer)

    def test_oauth_configuration_answer_has_version_caveat_and_steps(self) -> None:
        state = _base_state("How do I configure OAuth for API v3.2?", RouteDecision.RAG)
        state["retrieved_chunks"] = [
            {
                "chunk_text": (
                    "API v3.5 Base URL. OAuth 2.0 Authorization Flow. "
                    "Step 1: Redirect user to Authorization URL GET https://auth.example.com/oauth/authorize "
                    "with client_id, response_type=code, and redirect_uri. "
                    "Step 2: Exchange Authorization Code for Access Token POST https://auth.example.com/oauth/token "
                    "with grant_type authorization_code, code, client_id, and client_secret. "
                    "Token response returns access_token, token_type Bearer, expires_in, and refresh_token. "
                    "Use HTTPS and keep client secrets out of client-side code."
                ),
                "source_file": "API_Integration_Authentication_Guide.pdf",
                "page_number": 1,
                "score": 0.91,
            }
        ]

        result = compose_response(state)
        answer = result["final_answer"]

        self.assertIn("Assumptions / caveats", answer)
        self.assertIn("API v3.5", answer)
        self.assertIn("API v3.2", answer)
        self.assertIn("Recommended steps", answer)
        self.assertIn("GET /oauth/authorize", answer)
        self.assertIn("POST /oauth/token", answer)
        self.assertIn("Authorization: Bearer [REDACTED]", answer)
        self.assertIn("Security notes", answer)
        self.assertIn("OAuth 2.0 authorization flow", answer)
        self.assertNotIn("https://auth.example.com", answer)
        self.assertNotIn("secret-value", answer)

    def test_clean_final_answer_redacts_tokens(self) -> None:
        answer = clean_final_answer(
            'client_secret: super-secret access_token: eyJhbGciOiFakeToken Authorization: Bearer abcdefghijklmnop',
            max_words=180,
        )

        self.assertIn("client_secret: [REDACTED]", answer)
        self.assertIn("access_token: [REDACTED]", answer)
        self.assertIn("Authorization: Bearer [REDACTED]", answer)
        self.assertNotIn("super-secret", answer)
        self.assertNotIn("abcdefghijklmnop", answer)

    def test_sql_answer_has_no_document_sources_when_no_rag_evidence(self) -> None:
        state = _base_state("Which customers have suspended accounts?", RouteDecision.SQL)
        state["sql_results"] = [
            {
                "answer": "Found 1 matching structured record from customers: Acme Enterprise is Suspended.",
                "row_count": 1,
                "raw_results": [{"company_name": "Acme Enterprise", "account_status": "Suspended"}],
            }
        ]

        result = compose_response(state)
        answer = result["final_answer"]
        quality = result["metadata"]["answer_quality"]

        self.assertIn("Answer:", answer)
        self.assertNotIn("Sources:", answer)
        self.assertGreaterEqual(quality["faithfulness_score"], 0.9)

    def test_hybrid_answer_includes_structured_context_and_sources(self) -> None:
        state = _base_state("Customer C123 has 504 timeout errors. Check account and suggest next action.", RouteDecision.HYBRID)
        state["customer_context"] = {
            "sla_level": "Premium",
            "subscription_tier": "Enterprise",
            "account_status": "Active",
            "region": "APAC",
        }
        state["sql_results"] = [{"answer": "1 customer record returned.", "row_count": 1, "raw_results": []}]
        state["retrieved_chunks"] = [
            {
                "chunk_text": "504 timeout troubleshooting recommends checking upstream latency and regional incident status before escalation.",
                "source_file": "API_Error_Codes_Troubleshooting_Handbook.pdf",
                "page_number": 4,
                "score": 0.78,
            }
        ]

        result = compose_response(state)
        answer = result["final_answer"]

        self.assertIn("Premium", answer)
        self.assertIn("APAC", answer)
        self.assertIn("Sources:", answer)
        self.assertIn("API Error Codes Troubleshooting Handbook - page 4", answer)
        self.assertIn("final_answer_model", result["metadata"])

    def test_priority_sla_commitments_are_specific(self) -> None:
        query = "What are the SLA commitments for Priority customers?"
        route = _classify_local(query).route_decision
        self.assertEqual(route, RouteDecision.RAG)

        result = compose_response(
            _rag_state(
                query,
                "Support Tier Overview: Priority Support response time < 1 hour, 24/5 support, all channels plus phone, direct to L2 escalation, 10% SLA credits. "
                "Response Time Commitments by Severity: Critical P0 30 min, High P1 1 hour, Medium P2 4 hours, Low P3 8 hours.",
                "SLA_Support_Operation_Policy.pdf",
            )
        )
        answer = result["final_answer"]

        self.assertIn("under 1 hour", answer)
        self.assertIn("P0: 30 minutes", answer)
        self.assertIn("P1: 1 hour", answer)
        self.assertIn("P2: 4 hours", answer)
        self.assertIn("P3: 8 hours", answer)
        self.assertNotIn("there are four tiers", answer.lower())

    def test_itil_lifecycle_routes_and_answers_as_rag(self) -> None:
        query = "What are the incident lifecycle stages in ITIL?"
        self.assertEqual(_classify_local(query).route_decision, RouteDecision.RAG)

        result = compose_response(
            _rag_state(
                query,
                "ITIL lifecycle: 1. Identification 2. Classification & Prioritization 3. Diagnosis & Investigation 4. Resolution & Recovery 5. Incident Closure.",
                "ITIL_Incident_Management_Summary.pdf",
            )
        )

        self.assertIn("identification", result["final_answer"].lower())
        self.assertIn("classification and prioritization", result["final_answer"].lower())
        self.assertIn("resolution and recovery", result["final_answer"].lower())

    def test_rca_mandatory_routes_and_answers_as_rag(self) -> None:
        query = "When is RCA mandatory for incidents?"
        self.assertEqual(_classify_local(query).route_decision, RouteDecision.RAG)

        result = compose_response(
            _rag_state(
                query,
                "Root Cause Analysis (RCA) Requirements: RCA is mandatory for all P0 and P1 incidents, and recommended for recurring P2 incidents.",
                "ITIL_Incident_Management_Summary.pdf",
            )
        )

        self.assertIn("mandatory for all P0 and P1 incidents", result["final_answer"])
        self.assertIn("recurring P2", result["final_answer"])

    def test_read_heavy_cache_strategy_routes_and_answers_as_rag(self) -> None:
        query = "What caching strategy is recommended for read-heavy API data?"
        self.assertEqual(_classify_local(query).route_decision, RouteDecision.RAG)

        result = compose_response(
            _rag_state(
                query,
                "Cache-Control public, max-age=300 Cache for 5 minutes (read-heavy data). Cache Hit Ratio Target: >90%. ETag resource version for conditional requests.",
                "Performance_Scalability_Guide.pdf",
            )
        )

        self.assertIn("public, max-age=300", result["final_answer"])
        self.assertIn("above 90%", result["final_answer"])
        self.assertIn("ETags", result["final_answer"])

    def test_installation_verification_mentions_operational_checks(self) -> None:
        query = "How do I verify installation after deployment?"
        self.assertEqual(_classify_local(query).route_decision, RouteDecision.RAG)

        result = compose_response(
            _rag_state(
                query,
                "Step 8: Verify Installation docker-compose ps # Check all services are running curl http://localhost:8080/health # Health check. Review Logs & Troubleshoot.",
                "Product_Installation_Setup_Guide.pdf",
            )
        )

        answer = result["final_answer"]
        self.assertIn("docker-compose ps", answer)
        self.assertIn("curl http://localhost:8080/health", answer)
        self.assertIn("DB connectivity", answer)
        self.assertIn("logs", answer.lower())

    def test_400_invalid_json_answer_is_specific(self) -> None:
        query = "What should I check for a 400 invalid JSON error?"
        self.assertEqual(_classify_local(query).route_decision, RouteDecision.RAG)

        result = compose_response(
            _rag_state(
                query,
                "400 Invalid JSON Unexpected token at position 45 Validate JSON syntax. Request formatting: check Content-Type header and required fields.",
                "API_Error_Codes_Troubleshooting_Handbook.pdf",
            )
        )

        self.assertIn("validate the JSON syntax", result["final_answer"])
        self.assertIn("Content-Type", result["final_answer"])

    def test_504_gateway_timeout_answer_is_specific(self) -> None:
        query = "What is the recommended action for a 504 Gateway Timeout?"
        self.assertEqual(_classify_local(query).route_decision, RouteDecision.RAG)

        result = compose_response(
            _rag_state(
                query,
                "504 Gateway Timeout Request exceeded timeout (30s) Optimize query or increase client timeout.",
                "API_Error_Codes_Troubleshooting_Handbook.pdf",
            )
        )

        self.assertIn("30 seconds", result["final_answer"])
        self.assertIn("Optimize", result["final_answer"])
        self.assertIn("client timeout", result["final_answer"])

    def test_429_rate_limiting_answer_is_specific(self) -> None:
        query = "How should clients handle API 429 rate limiting?"
        self.assertEqual(_classify_local(query).route_decision, RouteDecision.RAG)

        result = compose_response(
            _rag_state(
                query,
                "429 Rate limit exceeded X-RateLimit-Remaining: 0 Retry-After: 60 Wait 60s, implement exponential backoff.",
                "API_Error_Codes_Troubleshooting_Handbook.pdf",
            )
        )

        self.assertIn("Retry-After", result["final_answer"])
        self.assertIn("exponential backoff", result["final_answer"])

    def test_regional_endpoint_latency_answer_is_specific(self) -> None:
        query = "How should regional endpoints be selected to reduce latency?"
        self.assertEqual(_classify_local(query).route_decision, RouteDecision.RAG)

        result = compose_response(
            _rag_state(
                query,
                "Regional Selection Strategy: Auto-routing uses global endpoint. Manual selection pins to specific region for data residency. Latency test: use /v3/ping endpoint.",
                "Performance_Scalability_Guide.pdf",
            )
        )

        self.assertIn("api.example.com", result["final_answer"])
        self.assertIn("/v3/ping", result["final_answer"])
        self.assertIn("data residency", result["final_answer"])

    def test_high_security_vulnerability_timeline_routes_as_rag(self) -> None:
        query = "What is the response timeline for a High security vulnerability?"
        self.assertEqual(_classify_local(query).route_decision, RouteDecision.RAG)

        result = compose_response(
            _rag_state(
                query,
                "High (P1) High likelihood of exploitation, significant risk < 1 hour. Patch Development < 24 hours Testing < 8 hours Deployment < 4 hours Total SLA < 36 hours.",
                "Security_Vulnerability_Response_Policy.pdf",
            )
        )

        answer = result["final_answer"]
        self.assertIn("under 1 hour", answer)
        self.assertIn("under 36 hours", answer)
        self.assertNotIn("requires human escalation", answer.lower())


if __name__ == "__main__":
    unittest.main()
