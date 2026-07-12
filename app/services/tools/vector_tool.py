import asyncio
import logging
import time
from typing import Any, Dict, Iterable

import psycopg
from psycopg import sql
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI

from app.core.config import settings
from app.core.langfuse import observe, trace_tool_result

logger = logging.getLogger(__name__)

VECTOR_TOOL_DESCRIPTION = (
    "Use this tool for questions about product documentation, API guides, troubleshooting steps, "
    "SLA policies, security procedures, ITIL processes, installation guides, performance guidelines, "
    "error code explanations, and any how-to or policy questions. Input must be a natural language "
    "question. The tool searches indexed documentation using PostgreSQL pgvector semantic search "
    "and PostgreSQL full-text search, then returns cited answers."
)

VECTOR_EVIDENCE_TOOL_DESCRIPTION = (
    "Use this tool for questions about product documentation, API guides, troubleshooting steps, "
    "SLA policies, security procedures, ITIL processes, installation guides, performance guidelines, "
    "error code explanations, and any how-to or policy questions. Input must be a natural language "
    "question. The tool searches indexed documentation using PostgreSQL pgvector semantic search "
    "and PostgreSQL full-text search, then returns evidence chunks, citations, source metadata, "
    "and retrieval scores only. It does not generate the final answer."
)

_EMBED_MODEL: AzureOpenAIEmbedding | None = None
_LLM: AzureOpenAI | None = None
_FULL_TEXT_INDEX_READY = False


def _build_llm() -> AzureOpenAI:
    return AzureOpenAI(
        model=settings.azure_openai_chat_deployment,
        deployment_name=settings.azure_openai_chat_deployment,
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        temperature=0.1,
    )


def _get_llm() -> AzureOpenAI:
    global _LLM
    if _LLM is None:
        _LLM = _build_llm()
    return _LLM


def _build_embedding() -> AzureOpenAIEmbedding:
    return AzureOpenAIEmbedding(
        model=settings.azure_openai_embedding_deployment,
        deployment_name=settings.azure_openai_embedding_deployment,
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
    )


def _get_embed_model() -> AzureOpenAIEmbedding:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = _build_embedding()
    return _EMBED_MODEL


def _vector_literal(values: Iterable[float]) -> str:
    return "[" + ",".join(str(float(value)) for value in values) + "]"


def _embedding_for_query(question: str) -> str:
    started = time.perf_counter()
    try:
        embedding = _get_embed_model().get_text_embedding(question)
        logger.info(
            "azure_openai_embedding_succeeded",
            extra={"duration_ms": int((time.perf_counter() - started) * 1000)},
        )
    except Exception:
        logger.exception("azure_openai_embedding_failed")
        raise
    return _vector_literal(embedding)


def _data_table_name() -> str:
    configured = settings.retrieval_vector_table_name or settings.db_table_name
    return configured if configured.startswith("data_") else f"data_{configured}"


def _ensure_full_text_index() -> None:
    global _FULL_TEXT_INDEX_READY
    if _FULL_TEXT_INDEX_READY or not settings.retrieval_enable_full_text:
        return

    table_name = _data_table_name()
    index_name = f"idx_{table_name}_text_fts"
    statement = sql.SQL(
        "CREATE INDEX IF NOT EXISTS {index_name} "
        "ON public.{table_name} "
        "USING gin (to_tsvector('english', coalesce(text, '')))"
    ).format(
        index_name=sql.Identifier(index_name),
        table_name=sql.Identifier(table_name),
    )

    with psycopg.connect(settings.database_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(statement)
        conn.commit()

    _FULL_TEXT_INDEX_READY = True
    logger.info("PostgreSQL full-text index ensured on table=%s", table_name)


def ensure_retrieval_indexes() -> None:
    """Prepare retrieval indexes outside the request path when possible."""

    _ensure_full_text_index()


def _row_to_source(row: dict[str, Any]) -> Dict[str, Any]:
    metadata = dict(row.get("metadata_") or {})
    text = str(row.get("text") or "")
    return {
        "chunk_text": text,
        "source_file": metadata.get("source_file") or metadata.get("file_name") or metadata.get("source"),
        "category": metadata.get("category"),
        "content_type": metadata.get("content_type"),
        "similarity_score": row.get("hybrid_score"),
        "metadata": {
            **metadata,
            "node_id": row.get("node_id"),
            "vector_score": row.get("vector_score"),
            "keyword_score": row.get("keyword_score"),
            "hybrid_score": row.get("hybrid_score"),
            "retrieval_strategy": "pgvector_plus_postgres_full_text",
        },
    }


def _normalize_scores(rows: list[dict[str, Any]], key: str) -> None:
    values = [float(row.get(key) or 0.0) for row in rows]
    if not values:
        return
    low = min(values)
    high = max(values)
    for row in rows:
        value = float(row.get(key) or 0.0)
        row[f"{key}_normalized"] = 1.0 if high == low and value > 0 else (value - low) / (high - low or 1.0)


def _merge_ranked_rows(
    dense_rows: list[dict[str, Any]],
    keyword_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in dense_rows:
        merged[row["node_id"]] = {**row, "keyword_score": 0.0}

    for row in keyword_rows:
        existing = merged.get(row["node_id"])
        if existing:
            existing["keyword_score"] = row.get("keyword_score") or 0.0
        else:
            merged[row["node_id"]] = {**row, "vector_score": 0.0}

    rows = list(merged.values())
    _normalize_scores(rows, "vector_score")
    _normalize_scores(rows, "keyword_score")

    vector_weight = settings.retrieval_vector_weight
    keyword_weight = settings.retrieval_keyword_weight
    for row in rows:
        row["hybrid_score"] = (
            vector_weight * float(row.get("vector_score_normalized") or 0.0)
            + keyword_weight * float(row.get("keyword_score_normalized") or 0.0)
        )

    return sorted(rows, key=lambda item: item.get("hybrid_score") or 0.0, reverse=True)[: settings.retrieval_top_k]


def _fetch_dense_rows(cur: Any, question_embedding: str, limit: int) -> list[dict[str, Any]]:
    table_name = _data_table_name()
    statement = sql.SQL(
        "SELECT id, text, metadata_, node_id, "
        "1.0 / (1.0 + (embedding <=> %s::vector)) AS vector_score, "
        "0.0::float AS keyword_score "
        "FROM public.{table_name} "
        "ORDER BY embedding <=> %s::vector "
        "LIMIT %s"
    ).format(table_name=sql.Identifier(table_name))
    cur.execute(statement, (question_embedding, question_embedding, limit))
    return [dict(row) for row in cur.fetchall()]


def _fetch_keyword_rows(cur: Any, question: str, limit: int) -> list[dict[str, Any]]:
    if not settings.retrieval_enable_full_text:
        return []

    table_name = _data_table_name()
    statement = sql.SQL(
        "SELECT id, text, metadata_, node_id, "
        "0.0::float AS vector_score, "
        "ts_rank_cd(to_tsvector('english', coalesce(text, '')), websearch_to_tsquery('english', %s)) AS keyword_score "
        "FROM public.{table_name} "
        "WHERE to_tsvector('english', coalesce(text, '')) @@ websearch_to_tsquery('english', %s) "
        "ORDER BY keyword_score DESC "
        "LIMIT %s"
    ).format(table_name=sql.Identifier(table_name))
    cur.execute(statement, (question, question, limit))
    return [dict(row) for row in cur.fetchall()]


def _retrieve_sources(question: str) -> list[dict[str, Any]]:
    _ensure_full_text_index()
    question_embedding = _embedding_for_query(question)
    candidate_limit = max(settings.retrieval_top_k * 4, settings.retrieval_top_k)

    with psycopg.connect(settings.database_dsn, row_factory=psycopg.rows.dict_row) as conn:
        with conn.cursor() as cur:
            dense_rows = _fetch_dense_rows(cur, question_embedding, candidate_limit)
            keyword_rows = _fetch_keyword_rows(cur, question, candidate_limit)

    ranked_rows = _merge_ranked_rows(dense_rows, keyword_rows)
    return [_row_to_source(row) for row in ranked_rows]


def _context_from_sources(source_nodes: list[dict[str, Any]]) -> str:
    blocks = []
    for index, node in enumerate(source_nodes, start=1):
        metadata = node.get("metadata", {})
        source = node.get("source_file") or metadata.get("source") or "unknown source"
        page = metadata.get("page_number")
        page_label = f", page {page}" if page else ""
        blocks.append(f"[{index}] Source: {source}{page_label}\n{node.get('chunk_text', '')}")
    return "\n\n".join(blocks)


def _generate_answer(question: str, source_nodes: list[dict[str, Any]]) -> str:
    if not source_nodes:
        return "I could not find relevant documentation for this question."

    prompt = (
        "You are a support resolution assistant. Answer the question using only the provided context. "
        "Give concise troubleshooting steps and mention when escalation is needed. "
        "Do not invent facts outside the context.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{_context_from_sources(source_nodes)}\n\n"
        "Answer:"
    )
    return str(_get_llm().complete(prompt))


def _run_query(question: str) -> Dict[str, Any]:
    source_nodes = _retrieve_sources(question)
    citations = sorted({node.get("source_file") for node in source_nodes if node.get("source_file")})
    return {
        "answer": _generate_answer(question, source_nodes),
        "source_nodes": source_nodes,
        "citations": citations,
        "chunk_count": len(source_nodes),
    }


def _run_retrieve(question: str) -> Dict[str, Any]:
    source_nodes = _retrieve_sources(question)
    citations = sorted({node.get("source_file") for node in source_nodes if node.get("source_file")})
    return {
        "source_nodes": source_nodes,
        "citations": citations,
        "chunk_count": len(source_nodes),
    }


@observe(name="vector_retrieval_tool", as_type="retriever", capture_input=False, capture_output=False)
async def execute_vector_retrieval(question: str) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        result = await asyncio.to_thread(_run_query, question)
    except Exception as exc:
        logger.exception("Vector retrieval failed")
        result = {
            "answer": "Documentation retrieval failed.",
            "source_nodes": [],
            "citations": [],
            "chunk_count": 0,
            "error": str(exc),
        }
    trace_tool_result(tool_name="vector_retrieval_tool", question=question, result=result, started_at=started)
    return result


@observe(name="vector_evidence_tool", as_type="retriever", capture_input=False, capture_output=False)
async def execute_vector_evidence(question: str) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        result = await asyncio.to_thread(_run_retrieve, question)
    except Exception as exc:
        logger.exception("Vector evidence retrieval failed")
        result = {
            "source_nodes": [],
            "citations": [],
            "chunk_count": 0,
            "error": str(exc),
        }
    trace_tool_result(tool_name="vector_evidence_tool", question=question, result=result, started_at=started)
    return result
