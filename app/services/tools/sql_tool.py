import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import re
import threading
import time
from typing import Any, Dict, List

from psycopg import OperationalError
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from llama_index.core import SQLDatabase
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.schema import QueryBundle
from llama_index.llms.azure_openai import AzureOpenAI
from sqlalchemy import create_engine

from app.core.config import settings
from app.core.langfuse import observe, trace_tool_result

logger = logging.getLogger(__name__)
logging.getLogger("llama_index.core.indices.struct_store.sql_retriever").setLevel(logging.WARNING)

SQL_TOOL_DESCRIPTION = (
    "Use this tool for structured data queries about customer accounts, support ticket status, "
    "incident logs, SLA tiers, subscription plans, escalation flags, and knowledge article usage. "
    "Input must be a natural language question. The tool converts the question to SQL, executes "
    "it safely against the database, and returns structured results."
)

_ALLOWED_TABLES = settings.operational_tables
_ENGINE: NLSQLTableQueryEngine | None = None
_EMBED_MODEL: BaseEmbedding | None = None
_DB_POOL: ConnectionPool | None = None
_ENGINE_LOCK = threading.Lock()
_DB_POOL_LOCK = threading.Lock()
_SQL_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="sql-tool")
DIRECT_SQL_MAX_ATTEMPTS = 2
MAX_STRUCTURED_ROWS_IN_ANSWER = 5
SAFE_RESULT_FIELDS_BY_TABLE = {
    "customers": [
        "customer_id",
        "company_name",
        "subscription_tier",
        "account_status",
        "sla_level",
        "renewal_date",
        "region",
    ],
    "support_tickets": [
        "ticket_id",
        "customer_id",
        "issue_category",
        "severity_level",
        "ticket_status",
        "created_at",
        "resolved_at",
        "assigned_team",
        "escalation_flag",
    ],
    "incident_logs": [
        "incident_id",
        "incident_type",
        "severity",
        "affected_region",
        "start_time",
        "end_time",
        "resolution_status",
        "root_cause",
        "escalation_flag",
    ],
    "knowledge_article_usage": [
        "article_id",
        "article_title",
        "product_version",
        "category",
        "last_updated",
        "known_issue_flag",
        "internal_confidence_score",
    ],
}
SENSITIVE_RESULT_FIELD_PATTERNS = re.compile(
    r"(password|secret|token|api_key|apikey|credential|private_key|authorization)",
    re.IGNORECASE,
)
CASE_INSENSITIVE_SQL_COLUMNS = {
    "account_status",
    "affected_region",
    "issue_category",
    "region",
    "resolution_status",
    "severity",
    "severity_level",
    "subscription_tier",
    "ticket_status",
}
TEXT_COMPARISON_PATTERN = re.compile(
    r"(?P<column>\b(?:[a-zA-Z_][a-zA-Z0-9_]*\.)?"
    r"(?:account_status|affected_region|issue_category|region|resolution_status|severity|severity_level|subscription_tier|ticket_status)\b)"
    r"\s*(?P<operator>=|!=|<>)\s*(?P<literal>'(?:''|[^'])*')",
    re.IGNORECASE,
)

TICKET_STATUSES = {
    "open": "Open",
    "closed": "Closed",
    "resolved": "Resolved",
    "in progress": "In Progress",
    "escalated": "Escalated",
}
SEVERITY_LEVELS = {
    "critical": "Critical",
    "high": "High",
    "medium": "Medium",
    "low": "Low",
}
CUSTOMER_ACCOUNT_STATUSES = {
    "active": "Active",
    "suspended": "Suspended",
    "trial": "Trial",
    "inactive": "Inactive",
    "closed": "Closed",
}
ACTIVE_INCIDENT_STATUSES = ("Open", "Investigating", "In Progress", "Escalated")
REGION_ALIASES = {
    "eu": "EU",
    "europe": "EU",
    "us": "US",
    "usa": "US",
    "united states": "US",
    "apac": "APAC",
    "asia pacific": "APAC",
    "mea": "MEA",
    "middle east": "MEA",
}


class CachedSQLDatabase(SQLDatabase):
    """SQLDatabase wrapper that caches table info used in NL2SQL prompts."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._table_info_cache: dict[str, str] = {}
        self._table_info_lock = threading.Lock()

    def get_single_table_info(self, table_name: str) -> str:
        with self._table_info_lock:
            cached = self._table_info_cache.get(table_name)
            if cached is not None:
                return cached

        started = time.perf_counter()
        table_info = super().get_single_table_info(table_name)
        with self._table_info_lock:
            self._table_info_cache[table_name] = table_info
        logger.info("Cached NL2SQL schema info table=%s latency_ms=%s", table_name, _elapsed_ms(started))
        return table_info

    def prefetch_table_info(self, tables: list[str]) -> None:
        started = time.perf_counter()
        for table in tables:
            self.get_single_table_info(table)
        logger.info(
            "Prefetched NL2SQL schema context table_count=%s latency_ms=%s",
            len(tables),
            _elapsed_ms(started),
        )


class NoOpSQLParserEmbedding(BaseEmbedding):
    """Placeholder embedding for the default NL2SQL parser.

    LlamaIndex's NLSQLRetriever resolves an embedding model even when the
    default SQL parser does not use one. Supplying this avoids initializing an
    Azure embedding client on the NL2SQL fallback path.
    """

    def _get_query_embedding(self, query: str) -> list[float]:
        return [0.0]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return [0.0]

    def _get_text_embedding(self, text: str) -> list[float]:
        return [0.0]


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _get_db_pool() -> ConnectionPool:
    global _DB_POOL
    if _DB_POOL is not None:
        return _DB_POOL
    with _DB_POOL_LOCK:
        if _DB_POOL is None:
            _DB_POOL = ConnectionPool(
                settings.database_dsn,
                kwargs={"row_factory": dict_row},
                min_size=1,
                max_size=5,
                open=True,
                timeout=10.0,
                name="support-sql-tool",
            )
    return _DB_POOL


def _execute_direct_select_once(sql_query: str, params: list[Any], attempt: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    timings: dict[str, int] = {}
    total_started = time.perf_counter()
    acquire_started = time.perf_counter()
    with _get_db_pool().connection() as conn:
        timings[f"db_pool_acquire_attempt_{attempt}_ms"] = _elapsed_ms(acquire_started)
        execute_started = time.perf_counter()
        with conn.cursor() as cur:
            cur.execute(sql_query, params)
            rows = [dict(row) for row in cur.fetchall()]
        timings[f"db_execute_attempt_{attempt}_ms"] = _elapsed_ms(execute_started)
    timings[f"db_total_attempt_{attempt}_ms"] = _elapsed_ms(total_started)
    return rows, timings


def _execute_direct_select(sql_query: str, params: list[Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    combined_timings: dict[str, int] = {}
    for attempt in range(1, DIRECT_SQL_MAX_ATTEMPTS + 1):
        try:
            rows, timings = _execute_direct_select_once(sql_query, params, attempt)
            combined_timings.update(timings)
            combined_timings["db_retry_count"] = attempt - 1
            return rows, combined_timings
        except OperationalError as exc:
            combined_timings[f"db_operational_error_attempt_{attempt}_ms"] = 0
            logger.warning(
                "Direct SQL attempt failed attempt=%s max_attempts=%s error=%s",
                attempt,
                DIRECT_SQL_MAX_ATTEMPTS,
                exc,
            )
            if attempt >= DIRECT_SQL_MAX_ATTEMPTS:
                raise
            # psycopg_pool discards the bad connection on context exit; next attempt
            # gets a fresh/healthy connection if the database is reachable.
            time.sleep(0.05 * attempt)
    raise RuntimeError("Direct SQL retry loop exited unexpectedly")


def _build_llm() -> AzureOpenAI:
    return AzureOpenAI(
        model=settings.azure_openai_chat_deployment,
        deployment_name=settings.azure_openai_chat_deployment,
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        temperature=0.1,
    )


def _get_embedding() -> BaseEmbedding:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = NoOpSQLParserEmbedding(model_name="noop-sql-parser-placeholder")
    return _EMBED_MODEL


def _get_engine() -> NLSQLTableQueryEngine:
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is not None:
            return _ENGINE

        started = time.perf_counter()
        sqlalchemy_engine = create_engine(settings.sqlalchemy_database_url, pool_pre_ping=True)
        sql_database = CachedSQLDatabase(sqlalchemy_engine, include_tables=_ALLOWED_TABLES)
        sql_database.prefetch_table_info(_ALLOWED_TABLES)
        _ENGINE = NLSQLTableQueryEngine(
            sql_database=sql_database,
            tables=_ALLOWED_TABLES,
            llm=_build_llm(),
            embed_model=_get_embedding(),
            synthesize_response=False,
            verbose=False,
        )
        logger.info("Initialized cached NL2SQL engine latency_ms=%s", _elapsed_ms(started))
    return _ENGINE


def warm_sql_tool() -> None:
    """Best-effort startup warmup for SQL pool and NL2SQL schema context."""

    started = time.perf_counter()
    try:
        pool = _get_db_pool()
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        logger.info("Warmed SQL connection pool latency_ms=%s", _elapsed_ms(started))
    except Exception:
        logger.exception("SQL connection pool warmup failed")

    engine_started = time.perf_counter()
    try:
        _get_engine()
        logger.info("Warmed cached NL2SQL engine latency_ms=%s", _elapsed_ms(engine_started))
    except Exception:
        logger.exception("NL2SQL engine warmup failed")


def close_sql_tool() -> None:
    global _DB_POOL
    if _DB_POOL is not None:
        _DB_POOL.close()
        _DB_POOL = None
        logger.info("Closed SQL tool connection pool")


def _metadata_value(metadata: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in metadata and metadata[key] is not None:
            return metadata[key]
    return None


def _selected_columns(sql_query: str) -> list[str]:
    match = re.search(r"\bselect\s+(.*?)\s+\bfrom\b", sql_query, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return []

    columns: list[str] = []
    for raw_column in match.group(1).split(","):
        column = " ".join(raw_column.strip().split())
        if not column:
            continue

        alias_match = re.search(r"\bas\s+([a-zA-Z_][a-zA-Z0-9_]*)$", column, flags=re.IGNORECASE)
        if alias_match:
            columns.append(alias_match.group(1))
            continue

        if re.search(r"\bcount\s*\(", column, flags=re.IGNORECASE):
            columns.append("count")
            continue

        column = re.sub(r"::[a-zA-Z_][a-zA-Z0-9_]*", "", column)
        column = column.split(".")[-1]
        column = column.strip('"`[] ')
        column = re.sub(r"[^a-zA-Z0-9_]+", "_", column).strip("_")
        columns.append(column or f"column_{len(columns) + 1}")
    return columns


def _row_to_mapping(row: Any, selected_columns: list[str]) -> dict[str, Any]:
    if isinstance(row, dict):
        return dict(row)
    if hasattr(row, "_mapping"):
        return dict(row._mapping)
    if hasattr(row, "_asdict"):
        return dict(row._asdict())
    if isinstance(row, (list, tuple)):
        columns = selected_columns or [f"column_{index + 1}" for index in range(len(row))]
        return {
            columns[index] if index < len(columns) else f"column_{index + 1}": value
            for index, value in enumerate(row)
        }
    return {"value": row}


def _normalize_result_rows(raw_results: List[Any], sql_query: str) -> list[dict[str, Any]]:
    selected_columns = _selected_columns(sql_query)
    return [_row_to_mapping(row, selected_columns) for row in raw_results]


def _safe_display_fields(tables: str, rows: list[dict[str, Any]]) -> list[str]:
    table_names = [table.strip() for table in tables.split(",") if table.strip() and table.strip() != "unknown"]
    fields: list[str] = []
    for table_name in table_names:
        for field_name in SAFE_RESULT_FIELDS_BY_TABLE.get(table_name, []):
            if field_name not in fields:
                fields.append(field_name)

    available_fields = [field for row in rows for field in row.keys()]
    if fields:
        display_fields = [field for field in fields if field in available_fields]
    else:
        display_fields = []

    if not display_fields:
        display_fields = [
            field
            for field in available_fields
            if not SENSITIVE_RESULT_FIELD_PATTERNS.search(str(field))
        ]
    return list(dict.fromkeys(display_fields))


def _format_value(value: Any) -> str:
    if value is None:
        return "null"
    text = str(value)
    text = " ".join(text.split())
    if len(text) > 160:
        return text[:157].rstrip() + "..."
    return text


def _format_result_row(row: dict[str, Any], display_fields: list[str]) -> str:
    parts = []
    for field_name in display_fields:
        if field_name not in row:
            continue
        parts.append(f"{field_name}={_format_value(row[field_name])}")
    return ", ".join(parts) if parts else "No display-safe fields returned."


def _is_count_result(rows: list[dict[str, Any]]) -> bool:
    return len(rows) == 1 and len(rows[0]) == 1 and next(iter(rows[0].keys())).lower() in {"count", "count_1", "column_1"}


def _format_structured_answer(raw_results: List[Any], sql_query: str) -> str:
    row_count = len(raw_results)
    if row_count == 0:
        return "No matching structured records were found."

    tables = _tables_used(sql_query)
    normalized_rows = _normalize_result_rows(raw_results, sql_query)
    if _is_count_result(normalized_rows):
        count_value = next(iter(normalized_rows[0].values()))
        return f"Count result from {tables}: {count_value}."

    record_label = "record" if row_count == 1 else "records"
    lines = [f"Found {row_count} matching structured {record_label} from {tables}:"]
    display_fields = _safe_display_fields(tables, normalized_rows)
    for index, row in enumerate(normalized_rows[:MAX_STRUCTURED_ROWS_IN_ANSWER], start=1):
        lines.append(f"{index}. {_format_result_row(row, display_fields)}")
    if row_count > MAX_STRUCTURED_ROWS_IN_ANSWER:
        lines.append(f"Showing first {MAX_STRUCTURED_ROWS_IN_ANSWER} of {row_count} records.")
    return "\n".join(lines)


def _tables_used(sql_query: str) -> str:
    lowered = sql_query.lower()
    used = [table for table in _ALLOWED_TABLES if re.search(rf"\b{re.escape(table.lower())}\b", lowered)]
    return ", ".join(used) if used else "unknown"


def _normalize_generated_sql_text_comparisons(sql_query: str) -> str:
    """Make generated SQL comparisons for known text columns case-insensitive.

    NL2SQL can emit predicates like ticket_status != 'resolved' while the
    database stores 'Resolved'. PostgreSQL string comparisons are case-sensitive,
    so normalize known enum/text filters before execution.
    """

    def replace(match: re.Match[str]) -> str:
        column = match.group("column")
        operator = match.group("operator")
        literal = match.group("literal")
        bare_column = column.split(".")[-1].lower()
        if bare_column not in CASE_INSENSITIVE_SQL_COLUMNS:
            return match.group(0)

        prefix = sql_query[max(0, match.start() - 16): match.start()].lower()
        if re.search(r"lower\s*\(\s*$", prefix):
            return match.group(0)

        return f"LOWER({column}) {operator} LOWER({literal})"

    return TEXT_COMPARISON_PATTERN.sub(replace, sql_query)


def _extract_status_filter(question: str) -> str | None:
    lowered = question.lower()
    for keyword, status in TICKET_STATUSES.items():
        if re.search(rf"\b{re.escape(keyword)}\b", lowered):
            return status
    return None


def _extract_severity_filter(question: str) -> str | None:
    lowered = question.lower()
    for keyword, severity in SEVERITY_LEVELS.items():
        if re.search(rf"\b{re.escape(keyword)}\b", lowered):
            return severity
    return None


def _extract_customer_status_filter(question: str) -> str | None:
    lowered = question.lower()
    for keyword, status in CUSTOMER_ACCOUNT_STATUSES.items():
        if re.search(rf"\b{re.escape(keyword)}\b", lowered):
            return status
    return None


def _extract_region_filter(question: str) -> str | None:
    lowered = question.lower()
    for keyword, region in REGION_ALIASES.items():
        if re.search(rf"\b{re.escape(keyword)}\b", lowered):
            return region
    return None


def _incident_status_filter(question: str) -> tuple[str, list[str]] | None:
    lowered = question.lower()
    if re.search(r"\b(active|open|ongoing|current|unresolved|in progress|investigating|escalated)\b", lowered):
        return "active", list(ACTIVE_INCIDENT_STATUSES)
    if re.search(r"\b(resolved|closed|history|historical|past)\b", lowered):
        return "resolved", ["Resolved"]
    return None


def _format_incident_answer(
    rows: List[Dict[str, Any]],
    *,
    status_label: str | None,
    region_filter: str | None,
    severity_filter: str | None,
    count_requested: bool,
) -> str:
    filters = []
    if status_label:
        filters.append(status_label)
    if severity_filter:
        filters.append(f"{severity_filter} severity")
    incident_noun = "incident" if len(rows) == 1 else "incidents"
    filters.append(incident_noun)
    if region_filter:
        filters.append(f"affecting {region_filter}")
    filter_text = " ".join(filters)

    if not rows:
        return f"No matching {filter_text} were found."

    prefix = "There is" if len(rows) == 1 else "There are"
    lines = [f"{prefix} {len(rows)} matching {filter_text}."]

    if count_requested:
        lines.append("Matched incident details:")

    for index, row in enumerate(rows, start=1):
        lines.append(
            (
                f"{index}. incident_id={row.get('incident_id', 'unknown')}, "
                f"type={row.get('incident_type') or 'unknown'}, "
                f"severity={row.get('severity') or 'unknown'}, "
                f"region={row.get('affected_region') or 'unknown'}, "
                f"status={row.get('resolution_status') or 'unknown'}, "
                f"started={row.get('start_time') or 'unknown'}, "
                f"root_cause={row.get('root_cause') or 'unknown'}"
            )
        )

    if status_label == "active":
        lines.append(
            f"Operational note: active {incident_noun} should be validated against current incident ownership and escalation policy."
        )
    return "\n".join(lines)


def _try_direct_incident_query(question: str) -> Dict[str, Any] | None:
    lowered = question.lower()
    if not re.search(r"\b(incident|incidents|incident log|incident logs)\b", lowered):
        return None
    if not re.search(r"\b(count|how many|number of|list|show|display|find|get|active|open|resolved|status|affecting|region)\b", lowered):
        return None

    status_filter = _incident_status_filter(question)
    region_filter = _extract_region_filter(question)
    severity_filter = _extract_severity_filter(question)
    count_requested = bool(re.search(r"\b(count|how many|number of)\b", lowered))

    where_parts: list[str] = []
    params: list[Any] = []

    if status_filter:
        _status_label, statuses = status_filter
        placeholders = ", ".join(["%s"] * len(statuses))
        where_parts.append(f"resolution_status IN ({placeholders})")
        params.extend(statuses)
    if region_filter:
        where_parts.append("LOWER(affected_region) = LOWER(%s)")
        params.append(region_filter)
    if severity_filter:
        where_parts.append("LOWER(severity) = LOWER(%s)")
        params.append(severity_filter)

    if not where_parts:
        return None

    where_sql = " AND ".join(where_parts)
    sql_query = (
        "SELECT incident_id, incident_type, severity, affected_region, start_time, "
        "end_time, resolution_status, root_cause, escalation_flag "
        f"FROM incident_logs WHERE {where_sql} ORDER BY start_time DESC"
    )

    rows, timings = _execute_direct_select(sql_query, params)

    return {
        "answer": _format_incident_answer(
            rows,
            status_label=status_filter[0] if status_filter else None,
            region_filter=region_filter,
            severity_filter=severity_filter,
            count_requested=count_requested,
        ),
        "sql_query": sql_query,
        "raw_results": rows,
        "table_used": "incident_logs",
        "row_count": len(rows),
        "execution_path": "direct_incident",
        "phase_timings_ms": timings,
    }


def _format_customer_account_answer(rows: List[Dict[str, Any]], status_filter: str) -> str:
    if not rows:
        return f"No customers currently have account status '{status_filter}'."

    status_label = status_filter.lower()
    lines = [
        f"Found {len(rows)} customer account(s) with status '{status_filter}':"
    ]
    for index, row in enumerate(rows, start=1):
        lines.append(
            (
                f"{index}. {row.get('company_name', 'Unknown customer')} "
                f"(customer_id={row.get('customer_id', 'unknown')}, "
                f"subscription={row.get('subscription_tier') or 'unknown'}, "
                f"SLA={row.get('sla_level') or 'unknown'}, "
                f"region={row.get('region') or 'unknown'}, "
                f"status={row.get('account_status') or status_filter})"
            )
        )
    if status_label == "suspended":
        lines.append("Recommended action: validate billing/account status before making integration or incident decisions.")
    return "\n".join(lines)


def _extract_ticket_age_hours_filter(question: str) -> int | None:
    lowered = question.lower()
    match = re.search(
        r"\b(?:beyond|over|older\s+than|more\s+than|past)\s+(\d+)\s*(hour|hours|hr|hrs|day|days)\b",
        lowered,
    )
    if not match:
        match = re.search(r"\b(\d+)\s*(hour|hours|hr|hrs|day|days)\b", lowered)
    if not match:
        return None

    value = int(match.group(1))
    unit = match.group(2)
    if unit.startswith("day"):
        return value * 24
    return value


def _is_ticket_age_query(question: str) -> bool:
    lowered = question.lower()
    if not re.search(r"\b(ticket|tickets)\b", lowered):
        return False
    if _extract_ticket_age_hours_filter(question) is None:
        return False
    return bool(
        re.search(
            r"\b(find|list|show|display|get|which|what|count|how many|number of|older than|more than|beyond|over|past|unresolved|open|resolved|escalated|overdue)\b",
            lowered,
        )
    )


def _is_resolution_duration_query(question: str) -> bool:
    lowered = question.lower()
    return bool(
        re.search(r"\b(resolve|resolved|resolution|closed)\b", lowered)
        and re.search(r"\b(took|taken|duration|time to resolve|resolution time|after|within|beyond|over|more than)\b", lowered)
    )


def _ticket_age_predicates(question: str, hours_threshold: int) -> tuple[list[str], list[Any], str, bool]:
    lowered = question.lower()
    where_parts: list[str] = []
    params: list[Any] = []
    duration_based = _is_resolution_duration_query(question)

    if duration_based:
        where_parts.extend(
            [
                "LOWER(st.ticket_status) = LOWER('Resolved')",
                "st.resolved_at IS NOT NULL",
                "st.resolved_at > st.created_at + (%s * INTERVAL '1 hour')",
            ]
        )
        params.append(hours_threshold)
        return where_parts, params, f"resolved tickets with resolution time over {hours_threshold} hours", True

    if re.search(r"\b(unresolved|not\s+resolved|overdue)\b", lowered):
        where_parts.extend(["LOWER(st.ticket_status) <> LOWER('Resolved')", "st.resolved_at IS NULL"])
        scope_label = f"unresolved tickets older than {hours_threshold} hours"
    else:
        status_filter = _extract_status_filter(question)
        if status_filter:
            where_parts.append("LOWER(st.ticket_status) = LOWER(%s)")
            params.append(status_filter)
            scope_label = f"{status_filter.lower()} tickets older than {hours_threshold} hours"
            if status_filter != "Resolved":
                where_parts.append("st.resolved_at IS NULL")
        elif re.search(r"\b(active|pending|current)\b", lowered):
            active_statuses = ["Open", "In Progress", "Escalated"]
            placeholders = ", ".join(["LOWER(%s)"] * len(active_statuses))
            where_parts.append(f"LOWER(st.ticket_status) IN ({placeholders})")
            where_parts.append("st.resolved_at IS NULL")
            params.extend(active_statuses)
            scope_label = f"active tickets older than {hours_threshold} hours"
        else:
            scope_label = f"tickets older than {hours_threshold} hours"

    severity_filter = _extract_severity_filter(question)
    if severity_filter:
        where_parts.append("LOWER(st.severity_level) = LOWER(%s)")
        params.append(severity_filter)
        scope_label = f"{severity_filter.lower()} severity {scope_label}"

    where_parts.append("st.created_at < NOW() - (%s * INTERVAL '1 hour')")
    params.append(hours_threshold)
    return where_parts, params, scope_label, False


def _format_ticket_age_answer(
    rows: List[Dict[str, Any]],
    *,
    scope_label: str,
    duration_based: bool,
) -> str:
    if not rows:
        return f"No matching {scope_label} were found."

    ticket_label = "ticket" if len(rows) == 1 else "tickets"
    lines = [f"Found {len(rows)} matching support {ticket_label}: {scope_label}."]
    for index, row in enumerate(rows, start=1):
        metric_field = "resolution_hours" if duration_based else "age_hours"
        metric_value = row.get(metric_field)
        metric_text = f", {metric_field}={_format_value(metric_value)}" if metric_value is not None else ""
        lines.append(
            (
                f"{index}. ticket_id={row.get('ticket_id', 'unknown')}, "
                f"customer={row.get('company_name') or row.get('customer_id') or 'unknown'}, "
                f"issue_category={row.get('issue_category') or 'unknown'}, "
                f"severity_level={row.get('severity_level') or 'unknown'}, "
                f"ticket_status={row.get('ticket_status') or 'unknown'}, "
                f"created_at={row.get('created_at') or 'unknown'}, "
                f"resolved_at={row.get('resolved_at') or 'null'}, "
                f"assigned_team={row.get('assigned_team') or 'unknown'}, "
                f"escalation_flag={row.get('escalation_flag')}"
                f"{metric_text}"
            )
        )
    lines.append("Operational note: tickets beyond the requested threshold should be reviewed for SLA exposure and ownership.")
    return "\n".join(lines)


def _try_direct_customer_query(question: str) -> Dict[str, Any] | None:
    lowered = question.lower()
    if not re.search(r"\b(customer|customers|account|accounts)\b", lowered):
        return None

    status_filter = _extract_customer_status_filter(question)
    if not status_filter:
        return None

    if not re.search(r"\b(which|list|show|display|find|get|have|has|with|status|account)\b", lowered):
        return None

    sql_query = (
        "SELECT customer_id, company_name, subscription_tier, account_status, sla_level, region "
        "FROM customers WHERE LOWER(account_status) = LOWER(%s) ORDER BY company_name"
    )

    rows, timings = _execute_direct_select(sql_query, [status_filter])

    return {
        "answer": _format_customer_account_answer(rows, status_filter),
        "sql_query": sql_query,
        "raw_results": rows,
        "table_used": "customers",
        "row_count": len(rows),
        "execution_path": "direct_customer",
        "phase_timings_ms": timings,
    }


def _try_direct_ticket_query(question: str) -> Dict[str, Any] | None:
    lowered = question.lower()
    if not re.search(r"\b(ticket|tickets)\b", lowered):
        return None

    ticket_age_query = _is_ticket_age_query(question)
    if not ticket_age_query and not re.search(r"\b(list|show|display|find|get|all|status|which|count|how many|number of)\b", lowered):
        return None

    if ticket_age_query:
        hours_threshold = _extract_ticket_age_hours_filter(question)
        if hours_threshold is None:
            return None
        where_parts, params, scope_label, duration_based = _ticket_age_predicates(question, hours_threshold)
        where_sql = " AND ".join(where_parts)
        metric_select = (
            "ROUND(EXTRACT(EPOCH FROM (st.resolved_at - st.created_at)) / 3600, 1) AS resolution_hours"
            if duration_based
            else "ROUND(EXTRACT(EPOCH FROM (NOW() - st.created_at)) / 3600, 1) AS age_hours"
        )
        order_clause = "st.resolved_at DESC" if duration_based else "st.created_at ASC"
        sql_query = (
            "SELECT st.ticket_id, st.customer_id, c.company_name, c.subscription_tier, c.sla_level, "
            "st.issue_category, st.severity_level, st.ticket_status, st.created_at, st.resolved_at, "
            f"st.assigned_team, st.escalation_flag, {metric_select} "
            "FROM support_tickets st "
            "LEFT JOIN customers c ON c.customer_id = st.customer_id "
            f"WHERE {where_sql} "
            f"ORDER BY {order_clause}"
        )
        rows, timings = _execute_direct_select(sql_query, params)
        return {
            "answer": _format_ticket_age_answer(
                rows,
                scope_label=scope_label,
                duration_based=duration_based,
            ),
            "sql_query": sql_query,
            "raw_results": rows,
            "table_used": "support_tickets, customers",
            "row_count": len(rows),
            "execution_path": "direct_ticket_age_status",
            "phase_timings_ms": timings,
        }

    status_filter = _extract_status_filter(question)
    severity_filter = _extract_severity_filter(question)

    where_parts = []
    params: list[Any] = []
    if severity_filter:
        where_parts.append("severity_level = %s")
        params.append(severity_filter)
    if status_filter:
        where_parts.append("ticket_status = %s")
        params.append(status_filter)

    if not where_parts:
        return None

    where_sql = " AND ".join(where_parts)
    sql_query = (
        "SELECT ticket_id, customer_id, issue_category, severity_level, ticket_status, "
        "created_at, assigned_team, escalation_flag "
        f"FROM support_tickets WHERE {where_sql} ORDER BY created_at DESC"
    )

    rows, timings = _execute_direct_select(sql_query, params)

    if rows:
        answer = _format_structured_answer(rows, sql_query)
    else:
        filters = []
        if status_filter:
            filters.append(status_filter.lower())
        if severity_filter:
            filters.append(f"{severity_filter} severity")
        answer = "There are currently no " + " ".join(filters).strip() + " tickets."

    return {
        "answer": answer,
        "sql_query": sql_query,
        "raw_results": rows,
        "table_used": "support_tickets",
        "row_count": len(rows),
        "execution_path": "direct_ticket",
        "phase_timings_ms": timings,
    }


def _run_nl2sql_fallback(question: str) -> Dict[str, Any]:
    timings: dict[str, int] = {}

    engine_started = time.perf_counter()
    engine = _get_engine()
    timings["engine_get_ms"] = _elapsed_ms(engine_started)

    retriever = engine.sql_retriever
    query_bundle = QueryBundle(question)

    schema_started = time.perf_counter()
    table_desc_str = retriever._get_table_context(query_bundle)
    timings["schema_context_ms"] = _elapsed_ms(schema_started)
    logger.info("NL2SQL schema context ready chars=%s latency_ms=%s", len(table_desc_str), timings["schema_context_ms"])

    llm_started = time.perf_counter()
    response_str = retriever._llm.predict(
        retriever._text_to_sql_prompt,
        query_str=query_bundle.query_str,
        schema=table_desc_str,
        dialect=retriever._sql_database.dialect,
    )
    timings["llm_sql_generation_ms"] = _elapsed_ms(llm_started)
    if timings["llm_sql_generation_ms"] >= 10000:
        logger.warning(
            "NL2SQL LLM generation was slow latency_ms=%s; Azure OpenAI retry/backoff or quota pressure may be involved.",
            timings["llm_sql_generation_ms"],
        )

    parse_started = time.perf_counter()
    sql_query = retriever._sql_parser.parse_response_to_sql(response_str, query_bundle)
    timings["sql_parse_ms"] = _elapsed_ms(parse_started)
    logger.info("NL2SQL generated query: %s", sql_query or "<not exposed by query engine>")

    normalize_started = time.perf_counter()
    normalized_sql_query = _normalize_generated_sql_text_comparisons(sql_query)
    timings["sql_normalization_ms"] = _elapsed_ms(normalize_started)
    if normalized_sql_query != sql_query:
        logger.info("NL2SQL normalized query for case-insensitive comparisons: %s", normalized_sql_query)
        sql_query = normalized_sql_query

    execute_started = time.perf_counter()
    try:
        _retrieved_nodes, metadata = retriever._sql_retriever.retrieve_with_metadata(sql_query)
    except BaseException as exc:
        if retriever._handle_sql_errors:
            metadata = {}
            timings["generated_sql_error_ms"] = _elapsed_ms(execute_started)
            logger.warning("NL2SQL generated SQL execution failed: %s", exc)
        else:
            raise
    timings["generated_sql_execution_ms"] = _elapsed_ms(execute_started)

    raw = _metadata_value(dict(metadata or {}), "result", "raw_result", "result_rows")
    if raw is None:
        raw_results: List[Any] = []
    elif isinstance(raw, list):
        raw_results = raw
    else:
        raw_results = [raw]

    timings["fallback_total_ms"] = sum(value for value in timings.values() if isinstance(value, int))
    logger.info("NL2SQL phase timings ms: %s", timings)
    return {
        "answer": _format_structured_answer(raw_results, sql_query),
        "sql_query": sql_query,
        "raw_results": raw_results,
        "table_used": _tables_used(sql_query),
        "row_count": len(raw_results),
        "execution_path": "nl2sql_fallback",
        "phase_timings_ms": timings,
    }


def _run_query(question: str) -> Dict[str, Any]:
    direct_incident_result = _try_direct_incident_query(question)
    if direct_incident_result is not None:
        logger.info("Direct structured incident query used: %s", direct_incident_result["sql_query"])
        return direct_incident_result

    direct_customer_result = _try_direct_customer_query(question)
    if direct_customer_result is not None:
        logger.info("Direct structured customer query used: %s", direct_customer_result["sql_query"])
        return direct_customer_result

    direct_result = _try_direct_ticket_query(question)
    if direct_result is not None:
        logger.info("Direct structured ticket query used: %s", direct_result["sql_query"])
        return direct_result

    return _run_nl2sql_fallback(question)


@observe(name="nl2sql_tool", as_type="tool", capture_input=False, capture_output=False)
async def execute_nl2sql(question: str) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(_SQL_EXECUTOR, _run_query, question)
    except Exception as exc:
        logger.exception("NL2SQL retrieval failed")
        result = {
            "answer": "Structured data retrieval failed.",
            "sql_query": "",
            "raw_results": [],
            "table_used": "unknown",
            "row_count": 0,
            "error": str(exc),
            "execution_path": "sql_tool_error",
            "phase_timings_ms": {},
        }
    logger.info(
        "SQL tool completed path=%s latency_ms=%s phase_timings_ms=%s",
        result.get("execution_path", "unknown"),
        _elapsed_ms(started),
        result.get("phase_timings_ms", {}),
    )
    trace_tool_result(tool_name="nl2sql_tool", question=question, result=result, started_at=started)
    return result
