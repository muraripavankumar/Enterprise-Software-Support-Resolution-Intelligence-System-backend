import asyncio
from datetime import datetime, timezone

from app.agents import doc_retrieval_agent as dra
from app.agents import incident_investigator_agent as iia
from app.orchestration.state import SeverityLevel, SeverityPriority
from app.services.tools import sql_tool


def test_document_source_node_normalizes_metadata_and_scores() -> None:
    chunk = dra._source_node_to_chunk(
        {
            "text": "OAuth setup requires authorization-code exchange.",
            "metadata": {
                "page": "7",
                "file_name": "API Integration Guide.pdf",
                "hybrid_score": "1.7",
                "content_type": "table_summary",
            },
            "category": "integration",
        }
    )

    assert chunk["chunk_text"] == "OAuth setup requires authorization-code exchange."
    assert chunk["source_file"] == "API Integration Guide.pdf"
    assert chunk["page_number"] == 7
    assert chunk["score"] == 1.0
    assert chunk["content_type"] == "table_summary"


def test_document_confidence_preserves_unsupported_lookup_route() -> None:
    state = {
        "confidence_score": 0.41,
        "metadata": {
            "intent_classifier_debug": {
                "matched_hits": {"unsupported_lookup": ["crm_case"]},
            }
        },
    }
    chunks = [{"score": 0.93, "chunk_text": "x"}]

    top_score = dra._update_confidence(state, chunks)

    assert top_score == 0.93
    assert state["confidence_score"] == 0.41


def test_document_retrieval_success_updates_state(monkeypatch) -> None:
    async def fake_execute_vector_evidence(query: str):
        assert query == "How do I configure OAuth?"
        return {
            "error": None,
            "citations": ["API Integration Guide.pdf"],
            "source_nodes": [
                {
                    "chunk_text": "Use OAuth authorization-code flow.",
                    "source_file": "API Integration Guide.pdf",
                    "page_number": 3,
                    "similarity_score": 0.84,
                }
            ],
        }

    monkeypatch.setattr(dra, "execute_vector_evidence", fake_execute_vector_evidence)
    monkeypatch.setattr(dra, "trace_agent_state", lambda **_kwargs: None)

    result = asyncio.run(
        dra.retrieve_documents(
            {
                "query": "How do I configure OAuth?",
                "progress_updates": [],
                "execution_results": [],
                "verification_outcomes": [],
                "agent_trace": [],
                "errors": [],
            }
        )
    )

    assert result["citations"] == ["API Integration Guide.pdf"]
    assert result["retrieved_chunks"][0]["score"] == 0.84
    assert result["confidence_score"] == 0.84
    assert result["verification_outcomes"][-1]["passed"] is True


def test_document_retrieval_error_records_failure(monkeypatch) -> None:
    async def fake_execute_vector_evidence(_query: str):
        return {"error": "vector store unavailable", "source_nodes": [], "citations": []}

    monkeypatch.setattr(dra, "execute_vector_evidence", fake_execute_vector_evidence)
    monkeypatch.setattr(dra, "trace_agent_state", lambda **_kwargs: None)

    result = asyncio.run(
        dra.retrieve_documents(
            {
                "query": "OAuth docs",
                "progress_updates": [],
                "execution_results": [],
                "verification_outcomes": [],
                "agent_trace": [],
                "errors": [],
            }
        )
    )

    assert result["retrieved_chunks"] == []
    assert "Document retrieval failed: vector store unavailable" in result["errors"]
    assert result["verification_outcomes"][-1]["passed"] is False


def test_incident_filters_and_process_documentation_detection() -> None:
    state = {
        "query": "Show active critical incidents in EU",
        "severity_priority": SeverityPriority.P0,
        "severity": SeverityLevel.CRITICAL,
    }

    assert iia._region_filter(state) == "EU"
    assert iia._severity_filter(state) == "Critical"
    assert iia._status_filter(state["query"]) == iia.ACTIVE_STATUSES
    assert iia._status_filter("show historical incident logs") is None
    assert iia._is_incident_process_documentation_query("What are the ITIL incident lifecycle phases?") is True
    assert iia._is_incident_process_documentation_query("Show active incident lifecycle status for EU") is False


def test_incident_correlation_uses_customer_region_error_code_and_domain() -> None:
    row = {
        "incident_id": 42,
        "incident_type": "Authentication outage",
        "severity": "Critical",
        "affected_region": "EU",
        "start_time": datetime(2026, 7, 10, tzinfo=timezone.utc),
        "end_time": None,
        "resolution_status": "Open",
        "root_cause": "Beta Systems OAuth 401 token failure",
        "escalation_flag": True,
    }

    record = iia._incident_record(row, "Customer Beta Systems is getting 401 OAuth errors in EU", "EU", ["Beta Systems"])

    assert record["incident_id"] == 42
    assert record["correlation_score"] >= iia.ACTIVE_CRITICAL_CORRELATION_THRESHOLD
    assert record["start_time"].startswith("2026-07-10T00:00:00")
    assert any("matched error code" in reason for reason in record["correlation_reasons"])
    assert any("matched customer/account name" in reason for reason in record["correlation_reasons"])
    assert iia._is_active_critical(row) is True


def test_incident_agent_handles_database_failure(monkeypatch) -> None:
    def fail_fetch(_region, _severity, _statuses):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(iia, "_fetch_incidents", fail_fetch)
    monkeypatch.setattr(iia, "trace_agent_state", lambda **_kwargs: None)
    monkeypatch.setattr(iia, "trace_guardrail_event", lambda **_kwargs: None)

    state = {
        "query": "Check active incidents for APAC",
        "progress_updates": [],
        "execution_results": [],
        "verification_outcomes": [],
        "agent_trace": [],
        "guardrail_flags": [],
        "errors": [],
    }
    result = asyncio.run(iia.investigate_incidents(state))

    investigation = result["incident_investigation"]
    assert investigation["matched_incidents"] == []
    assert investigation["active_critical_incident"] is False
    assert "failed" in investigation["investigation_summary"].lower()
    assert result["verification_outcomes"][-1]["passed"] is False


def test_sql_text_comparison_normalization_and_ticket_age_helpers() -> None:
    sql = "SELECT * FROM support_tickets st WHERE st.ticket_status != 'resolved' AND st.severity_level = 'high'"
    normalized = sql_tool._normalize_generated_sql_text_comparisons(sql)

    assert "LOWER(st.ticket_status) != LOWER('resolved')" in normalized
    assert "LOWER(st.severity_level) = LOWER('high')" in normalized
    assert sql_tool._extract_ticket_age_hours_filter("Find tickets unresolved beyond 2 days") == 48
    assert sql_tool._extract_ticket_age_hours_filter("Tickets over 36 hours") == 36
    assert sql_tool._is_ticket_age_query("Find tickets unresolved beyond 48 hours") is True
    assert sql_tool._is_resolution_duration_query("Which tickets took more than 72 hours to resolve?") is True


def test_sql_ticket_age_predicates_and_answer_formatting() -> None:
    where_parts, params, scope_label, duration_based = sql_tool._ticket_age_predicates(
        "Find high severity tickets unresolved beyond 48 hours",
        48,
    )

    assert "unresolved tickets older than 48 hours" in scope_label
    assert "LOWER(st.ticket_status) <> LOWER('Resolved')" in where_parts
    assert "LOWER(st.severity_level) = LOWER(%s)" in where_parts
    assert params == ["High", 48]
    assert duration_based is False

    answer = sql_tool._format_ticket_age_answer(
        [
            {
                "ticket_id": 7,
                "company_name": "Acme Corp",
                "issue_category": "Integration",
                "severity_level": "High",
                "ticket_status": "Open",
                "created_at": "2026-07-08 10:00:00",
                "resolved_at": None,
                "assigned_team": "API Support",
                "escalation_flag": False,
                "age_hours": 51,
            }
        ],
        scope_label=scope_label,
        duration_based=duration_based,
    )

    assert "Found 1 matching support ticket" in answer
    assert "ticket_id=7" in answer
    assert "age_hours=51" in answer
    assert "SLA exposure" in answer


def test_sql_direct_incident_query_uses_case_insensitive_filters(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_execute(sql_query: str, params: list[object]):
        captured["sql"] = sql_query
        captured["params"] = params
        return (
            [
                {
                    "incident_id": 2,
                    "incident_type": "API Failure",
                    "severity": "High",
                    "affected_region": "US",
                    "start_time": "2026-07-09",
                    "end_time": None,
                    "resolution_status": "Open",
                    "root_cause": "rate limit misconfiguration",
                    "escalation_flag": True,
                }
            ],
            {"db_total_ms": 2},
        )

    monkeypatch.setattr(sql_tool, "_execute_direct_select", fake_execute)

    result = sql_tool._try_direct_incident_query("Show active high incidents affecting US")

    assert result is not None
    assert "LOWER(affected_region) = LOWER(%s)" in captured["sql"]
    assert "LOWER(severity) = LOWER(%s)" in captured["sql"]
    assert captured["params"][-2:] == ["US", "High"]
    assert result["row_count"] == 1
    assert "active High severity incident" in result["answer"]
