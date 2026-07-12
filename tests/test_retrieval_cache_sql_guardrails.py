from __future__ import annotations

import asyncio

import pytest

from app.agents.sql_agent import _validate_select_only
from app.core import semantic_cache
from app.core.semantic_cache import CacheRoute, SemanticCache, _build_exact_hash, build_prompt_cache_key, cache_route_for_query
from app.services.tools.vector_tool import _merge_ranked_rows


def test_hybrid_retrieval_merge_ranks_dense_and_keyword_rows(monkeypatch: pytest.MonkeyPatch, fixed_retrieval_rows: tuple[list[dict], list[dict]]) -> None:
    monkeypatch.setattr("app.services.tools.vector_tool.settings.retrieval_vector_weight", 0.5)
    monkeypatch.setattr("app.services.tools.vector_tool.settings.retrieval_keyword_weight", 0.5)
    monkeypatch.setattr("app.services.tools.vector_tool.settings.retrieval_top_k", 3)

    dense_rows, keyword_rows = fixed_retrieval_rows
    ranked = _merge_ranked_rows(dense_rows, keyword_rows)

    assert [row["node_id"] for row in ranked] == ["doc-b", "doc-a", "doc-c"]
    assert ranked[0]["hybrid_score"] >= ranked[1]["hybrid_score"]
    assert ranked[0]["keyword_score"] == 0.99


def test_hybrid_retrieval_empty_candidate_sets_do_not_throw() -> None:
    assert _merge_ranked_rows([], []) == []


def test_cache_ttl_policy_by_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.core.semantic_cache.settings.cache_rag_ttl_seconds", 600)
    monkeypatch.setattr("app.core.semantic_cache.settings.cache_sql_ttl_seconds", 45)
    monkeypatch.setattr("app.core.semantic_cache.settings.cache_safety_critical_ttl_seconds", 0)
    monkeypatch.setattr("app.core.semantic_cache.settings.cache_ttl_seconds", 300)

    cache = SemanticCache()

    assert 300 <= cache.ttl_for_route(CacheRoute.RAG) <= 600
    assert 30 <= cache.ttl_for_route(CacheRoute.SQL) <= 60
    assert cache.ttl_for_route(CacheRoute.SAFETY_CRITICAL) == 0


def test_safety_critical_cache_bypasses_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.core.semantic_cache.settings.enable_semantic_cache", True)
    monkeypatch.setattr("app.core.semantic_cache.settings.cache_safety_critical_ttl_seconds", 0)

    cache = SemanticCache()
    monkeypatch.setattr(cache, "_get_redis_client", lambda: (_ for _ in ()).throw(AssertionError("redis should not be used")))

    result = asyncio.run(cache.set("Production outage with data loss", {"answer": "x"}, CacheRoute.SAFETY_CRITICAL))

    assert result.stored is False
    assert result.reason == "ttl disabled for route"


def test_cache_key_is_stable_and_collision_resistant_for_route_and_attributes() -> None:
    first = _build_exact_hash(" List  open tickets ", CacheRoute.SQL, {"role": "agent"})
    second = _build_exact_hash("list open tickets", CacheRoute.SQL, {"role": "agent"})
    different_route = _build_exact_hash("list open tickets", CacheRoute.RAG, {"role": "agent"})
    different_attrs = _build_exact_hash("list open tickets", CacheRoute.SQL, {"role": "customer"})

    assert first == second
    assert first != different_route
    assert first != different_attrs


def test_prompt_cache_key_is_stable_and_debuggable() -> None:
    first = build_prompt_cache_key(" List  open tickets ", CacheRoute.SQL, {"role": "agent"})
    second = build_prompt_cache_key("list open tickets", CacheRoute.SQL, {"role": "agent"})

    assert first == second
    assert "list open tickets" in first
    assert '"route":"sql"' in first


def test_process_prompt_cache_fallback_produces_predictable_hit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.core.semantic_cache.settings.enable_semantic_cache", True)
    monkeypatch.setattr("app.core.semantic_cache.settings.cache_rag_ttl_seconds", 600)
    monkeypatch.setattr("app.core.semantic_cache.settings.redis_url", "")
    monkeypatch.setattr("app.core.semantic_cache.settings.redis_api_url", "")
    monkeypatch.setattr("app.core.semantic_cache.settings.redis_api_key", "")
    monkeypatch.setattr("app.core.semantic_cache.settings.redis_store_id", "")

    cache = SemanticCache()
    payload = {"answer": "Use OAuth authorization-code flow."}

    store_result = asyncio.run(cache.set("How do I configure OAuth?", payload, CacheRoute.RAG, {"role": "agent"}))
    hit = asyncio.run(cache.get(" how  do I configure oauth? ", CacheRoute.RAG, {"role": "agent"}))

    assert store_result.stored is True
    assert store_result.reason == "stored in process prompt_cache_key fallback"
    assert hit is not None
    assert hit.strategy == "prompt_cache_key"
    assert hit.response == payload


def test_cache_route_selects_safety_critical_before_sql() -> None:
    route = cache_route_for_query("agent", "List tickets for production outage with data loss", {})

    assert route == CacheRoute.SAFETY_CRITICAL


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT ticket_id FROM support_tickets",
        "WITH open_tickets AS (SELECT * FROM support_tickets) SELECT * FROM open_tickets",
    ],
)
def test_nl2sql_validation_accepts_read_only_sql(sql: str) -> None:
    validation = _validate_select_only(sql)

    assert validation.passed is True
    assert validation.status == "passed"


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM support_tickets",
        "SELECT * FROM support_tickets; DROP TABLE customers;",
        "SELECT * FROM support_tickets WHERE ticket_id = 1; UPDATE customers SET account_status = 'Active'",
    ],
)
def test_nl2sql_validation_rejects_mutation_or_multiple_statements(sql: str) -> None:
    validation = _validate_select_only(sql)

    assert validation.passed is False
    assert validation.status == "failed"
