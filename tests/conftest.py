from __future__ import annotations

import pytest

from app.orchestration.state import RouteDecision, SupportIntent


@pytest.fixture
def base_state() -> dict:
    return {
        "query": "How do I configure OAuth for API v3.2?",
        "intent": SupportIntent.USAGE,
        "route_decision": RouteDecision.RAG,
        "metadata": {},
        "retrieved_chunks": [],
        "sql_results": [],
        "customer_context": {},
        "incident_investigation": {},
        "progress_updates": [],
        "execution_results": [],
        "verification_outcomes": [],
        "agent_trace": [],
        "guardrail_flags": [],
        "errors": [],
        "citations": [],
        "recommended_actions": [],
        "escalation_flag": False,
    }


@pytest.fixture
def fixed_retrieval_rows() -> tuple[list[dict], list[dict]]:
    dense_rows = [
        {"node_id": "doc-a", "text": "OAuth authorization flow", "metadata_": {}, "vector_score": 0.95},
        {"node_id": "doc-b", "text": "SLA policy overview", "metadata_": {}, "vector_score": 0.20},
    ]
    keyword_rows = [
        {"node_id": "doc-b", "text": "SLA policy overview", "metadata_": {}, "keyword_score": 0.99},
        {"node_id": "doc-c", "text": "Incident response policy", "metadata_": {}, "keyword_score": 0.30},
    ]
    return dense_rows, keyword_rows
