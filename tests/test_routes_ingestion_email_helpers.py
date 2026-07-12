from __future__ import annotations

import asyncio
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from fastapi import HTTPException, UploadFile

from app.api.routes import jira as jira_route
from app.api.routes import retrieval
from app.core.auth import AuthenticatedUser
from app.ingestion.document_loader import DocumentLoader
from app.ingestion.document_parser import DocumentParser
from app.ingestion.text_normalizer import normalize_extracted_text, remove_markdown_tables
from app.schemas.retrieval import RetrievalMode, RetrievalRequest, RetrievalResponse
from app.services import agent_service
from app.services.tools import email_mcp_tool


def _user(*permissions: str) -> AuthenticatedUser:
    return AuthenticatedUser(sub="u1", permissions=list(permissions), roles=["customer_user"], primary_role="customer_user")


def test_retrieval_route_cleans_paths_and_citations() -> None:
    assert retrieval._display_source(r"C:\docs\API Guide.pdf") == "API Guide.pdf"
    assert retrieval._clean_citations([r"C:\docs\a.pdf", "a.pdf", "", None]) == ["a.pdf"]
    doc, page, source = retrieval._citation_parts("API Guide.pdf pages 12")
    assert (doc, page, source) == ("API Guide.pdf", 12, "API Guide.pdf")


def test_retrieval_source_nodes_and_citation_references() -> None:
    result = {
        "source_nodes": [
            {
                "chunk_text": "OAuth flow",
                "source_file": r"C:\docs\api.pdf",
                "score": "0.82",
                "metadata": {"page_number": 3, "secret": "hidden", "source_file": r"C:\docs\api.pdf"},
            }
        ]
    }

    nodes = retrieval._source_nodes(result, include_sources=True)
    refs = retrieval._citation_references(citations=["api.pdf pages 3"], nodes=nodes)

    assert nodes[0].source_file == "api.pdf"
    assert nodes[0].metadata == {"page_number": 3, "source_file": "api.pdf"}
    assert refs[0].document_name == "api.pdf"
    assert refs[0].pages == [3]


def test_retrieval_permission_scope_hides_ticket_and_jira_evidence() -> None:
    result = {
        "answer": "Ticket details. Jira tracking: KAN-1 https://example.atlassian.net/browse/KAN-1",
        "structured_result": {"table_used": "support_tickets", "raw_results": [{"ticket_id": 1}]},
        "jira_tracking_result": {"issue_key": "KAN-1"},
        "tools_used": ["sql_agent", "jira_mcp_tool"],
    }

    scoped = retrieval._scope_result_to_permissions(result, _user("ask:support_query"))

    assert scoped["structured_result"] is None
    assert scoped["jira_tracking_result"] == {}
    assert "read:tickets" in scoped["answer"]
    assert scoped["tools_used"] == []


def test_retrieval_response_builders_and_cache_payload() -> None:
    request = RetrievalRequest(question="OAuth?", mode=RetrievalMode.VECTOR, include_sources=True)
    response = retrieval._vector_response(
        request,
        {
            "answer": "Use OAuth",
            "source_nodes": [{"chunk_text": "OAuth", "source_file": "api.pdf", "score": 0.9, "metadata": {"page_number": 1}}],
            "citations": ["api.pdf"],
            "chunk_count": 1,
        },
        0.0,
    )

    assert response.success is True
    assert response.route_decision == "rag"
    assert response.confidence_score == 0.9
    assert response.citation_references[0].document_name == "api.pdf"

    cached = retrieval._cacheable_response_payload(
        RetrievalResponse(
            success=True,
            mode=RetrievalMode.SQL,
            question="tickets",
            answer="rows",
            structured_result={"answer": "rows", "row_count": 1, "raw_results": [{"ticket_id": 1}]},
            latency_ms=1,
        )
    )
    assert cached["structured_result"]["raw_results"] == []


def test_quality_slo_warning_does_not_escalate_response() -> None:
    response = RetrievalResponse(
        success=True,
        mode=RetrievalMode.AGENT,
        question="How do I configure OAuth?",
        answer="Use OAuth authorization-code flow.",
        route_decision="rag",
        source_nodes=[{"text_preview": "OAuth flow", "source_file": "api.pdf"}],
        answer_quality={
            "faithfulness_score": 0.70,
            "answer_relevance_score": 0.84,
            "overall_quality_score": 0.80,
        },
        escalation_flag=False,
        latency_ms=1,
    )

    warned = retrieval._with_quality_slo_warnings(response)

    assert warned.escalation_flag is False
    assert warned.quality_warnings
    assert {warning.metric for warning in warned.quality_warnings} == {
        "answer_relevance",
        "faithfulness",
        "llm_judge_quality",
    }
    assert "Quality notice:" in warned.answer
    assert "below the configured target" in warned.answer


def test_document_retrieval_slo_warning_when_evidence_missing() -> None:
    response = RetrievalResponse(
        success=True,
        mode=RetrievalMode.AGENT,
        question="What are the SLA commitments?",
        answer="I found limited evidence.",
        route_decision="rag",
        source_nodes=[],
        citations=[],
        escalation_flag=False,
        latency_ms=1,
    )

    warned = retrieval._with_quality_slo_warnings(response)

    assert warned.escalation_flag is False
    assert {warning.metric for warning in warned.quality_warnings} == {
        "context_precision",
        "retrieval_recall_at_5",
    }
    assert "Quality notice:" in warned.answer


def test_retrieval_sse_and_answer_chunks() -> None:
    message = retrieval._sse_message("answer_delta", {"text": "hello"})
    assert message.startswith("event: answer_delta")
    assert retrieval._answer_chunks("one two three four", words_per_chunk=2) == ["one two ", "three four"]
    assert retrieval._stream_progress_message(99) == "Composing response from verified evidence"


def test_jira_route_helpers() -> None:
    with patch("app.api.routes.jira.settings.jira_url", "https://example.atlassian.net"):
        issue = jira_route._issue_from_jira(
            {
                "key": "KAN-1",
                "fields": {
                    "summary": "Outage",
                    "status": {"name": "Open"},
                    "assignee": {"displayName": "R. Patel"},
                    "priority": {"name": "Highest"},
                    "issuetype": {"name": "Bug"},
                    "created": "2026-07-10T01:02:03.000+0000",
                },
            }
        )

    assert issue["key"] == "KAN-1"
    assert issue["status"] == "Open"
    assert issue["assignee"] == "R. Patel"
    assert issue["url"].endswith("/browse/KAN-1")


def test_jira_list_not_configured_returns_503(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.api.routes.jira.settings.jira_url", "")

    with pytest.raises(HTTPException) as exc:
        asyncio.run(jira_route.list_jira_issues(_current_user=_user("view:evaluation")))

    assert exc.value.status_code == 503


def test_text_normalizer_and_table_removal() -> None:
    assert normalize_extracted_text("A&nbsp;B â€™ C").replace("\xa0", " ") == "A B ' C"
    assert remove_markdown_tables("| A | B |\n| 1 | 2 |\n\nKeep this") == "Keep this"


def _upload(filename: str, content: bytes, content_type: str = "application/pdf") -> UploadFile:
    return UploadFile(file=BytesIO(content), filename=filename, headers={"content-type": content_type})


def test_document_loader_validates_uploads_and_saves_temp(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr("app.ingestion.document_loader.settings.temp_upload_dir", tmp_path)
    monkeypatch.setattr("app.ingestion.document_loader.settings.max_upload_mb", 1)
    loader = DocumentLoader()
    upload = _upload("guide.pdf", b"%PDF-1.4")

    temp_path, metadata = loader.save_temp_pdf(upload, "doc-1")

    assert temp_path.exists()
    assert metadata["original_filename"] == "guide.pdf"
    assert metadata["file_size_bytes"] == len(b"%PDF-1.4")

    with pytest.raises(ValueError):
        loader.validate_pdf(_upload("guide.txt", b"x"))
    with pytest.raises(ValueError):
        loader.validate_pdf(_upload("guide.pdf", b""))


def test_document_parser_pymupdf_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakePage:
        def __init__(self, text: str) -> None:
            self._text = text

        def get_text(self, _kind: str) -> str:
            return self._text

    class FakeDocument:
        page_count = 2

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def __iter__(self):
            return iter([FakePage("Page one"), FakePage("")])

    monkeypatch.setattr("app.ingestion.document_parser.settings.document_parser_provider", "pymupdf")
    monkeypatch.setattr("app.ingestion.document_parser.fitz.open", lambda _path: FakeDocument())

    pages = DocumentParser().parse_pdf("fake.pdf")

    assert pages == [{"text": "Page one", "page_number": 1, "metadata": {"parser": "pymupdf", "page_count": 2}}]


def test_email_tool_subject_body_and_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    state = {"query": "data loss", "escalation_target": "incident_response"}
    package = {
        "target": "incident_response",
        "severity_priority": "p0",
        "reason": "data loss",
        "intent": "incident",
        "route_decision": "high_risk",
        "confidence_score": 0.98,
        "customer_context": {"company_name": "Beta", "customer_id": 2, "account_status": "Active"},
        "incident_investigation": {"active_critical_incident_correlated": True, "matched_incidents": [{"incident_id": 1, "severity": "Critical"}]},
    }
    jira = {"issue_key": "KAN-5", "issue_url": "https://example.atlassian.net/browse/KAN-5"}

    subject = email_mcp_tool._build_subject(state, package, jira)
    body = email_mcp_tool._build_body(state, package, jira)

    assert "[ERIS][p0]" in subject
    assert "KAN-5" in subject
    assert "Customer: Beta" in body
    assert "Incident 1: id=1" in body

    monkeypatch.setattr("app.services.tools.email_mcp_tool.settings.enable_escalation_email_mcp", False)
    with patch("app.services.tools.email_mcp_tool.trace_tool_result"):
        result = asyncio.run(email_mcp_tool.send_escalation_email(state, package, jira))
    assert result["status"] == "disabled"


def test_email_tool_send_success_and_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.services.tools.email_mcp_tool.settings.enable_escalation_email_mcp", True)
    monkeypatch.setattr("app.services.tools.email_mcp_tool.settings.escalation_email_to", "support@company.test")
    monkeypatch.setattr("app.services.tools.email_mcp_tool.settings.escalation_email_from", "eris@company.test")
    monkeypatch.setattr("app.services.tools.email_mcp_tool.settings.smtp_host", "smtp.company.test")
    monkeypatch.setattr("app.services.tools.email_mcp_tool.settings.smtp_username", "eris@company.test")
    monkeypatch.setattr("app.services.tools.email_mcp_tool.settings.smtp_password", "secret")

    with patch("app.services.tools.email_mcp_tool._send_email") as send_email, patch(
        "app.services.tools.email_mcp_tool.trace_tool_result"
    ):
        result = asyncio.run(email_mcp_tool.send_escalation_email({"query": "q"}, {"target": "l2"}, None))
    assert result["status"] == "sent"
    send_email.assert_called_once()

    with patch("app.services.tools.email_mcp_tool._send_email", side_effect=RuntimeError("smtp down")), patch(
        "app.services.tools.email_mcp_tool.trace_tool_result"
    ):
        failed = asyncio.run(email_mcp_tool.send_escalation_email({"query": "q"}, {"target": "l2"}, None))
    assert failed["status"] == "failed"
    assert "smtp down" in failed["error"]


def test_agent_service_hybrid_merge_and_conflicts() -> None:
    original = {"progress_updates": [], "execution_results": [], "verification_outcomes": [], "guardrail_flags": [], "agent_trace": [], "errors": [], "confidence_score": 0.4}
    vector = {**original, "retrieved_chunks": [{"chunk_text": "doc"}], "citations": ["api.pdf"], "confidence_score": 0.8, "progress_updates": [{"step_id": "doc"}]}
    sql = {**original, "sql_results": [{"answer": "row", "row_count": 1}], "confidence_score": 0.7, "progress_updates": [{"step_id": "sql"}]}

    merged = agent_service._merge_parallel_state(original, vector, sql, 12)

    assert merged["confidence_score"] == 0.8
    assert merged["hybrid_result"]["rag_evidence_count"] == 1
    assert merged["hybrid_result"]["sql_evidence_count"] == 1
    assert merged["verification_outcomes"][-1]["passed"] is True

    assert agent_service._hybrid_conflicts([], []) == [
        "No document evidence was retrieved for the hybrid query.",
        "No structured SQL evidence was retrieved for the hybrid query.",
    ]
