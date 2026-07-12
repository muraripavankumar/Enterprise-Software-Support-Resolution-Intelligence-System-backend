import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

try:
    import redis.asyncio as redis_async
except ImportError:  # pragma: no cover - redis is optional at import time.
    redis_async = None


class CacheRoute(str, Enum):
    RAG = "rag"
    SQL = "sql"
    HYBRID = "hybrid"
    AGENT = "agent"
    SAFETY_CRITICAL = "safety_critical"


@dataclass(frozen=True)
class SemanticCacheHit:
    strategy: str
    response: Any
    exact_hash: str | None = None
    similarity_score: float | None = None
    entry_id: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    raw_entry: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SemanticCacheStoreResult:
    stored: bool
    reason: str
    exact_hash: str | None = None
    entry_id: str | None = None


@dataclass(frozen=True)
class _ProcessCacheEntry:
    response: Any
    attributes: dict[str, Any]
    expires_at: int


SAFETY_CRITICAL_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\boutage\b",
        r"\bproduction\s+(down|impact|outage)\b",
        r"\b(data|security|credential|credentials|privacy|customer\s+data|api\s+key|token|secret)\s+breach(?:ed|es)?\b",
        r"\bbreach(?:ed|es)?\s+(of\s+)?(data|security|credentials|privacy|customer\s+data|api\s+key|token|secret)\b",
        r"\bunauthorized\s+access\b",
        r"\b(customer\s+data|api\s+key|token|secret)\s+(exposed|leaked|compromised)\b",
        r"\bsecurity\s+(vulnerability|incident|alert)\b",
        r"\bdata\s+loss\b",
        r"\bcritical\s+(incident|alert|severity)\b",
        r"\bunresolved\s+critical\s+alert\b",
        r"\bpremium\s+customer\s+outage\b",
        r"\baccount\s+suspension\b",
    ]
]

SQL_ROUTE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\b(ticket|tickets|customer|customers|account|subscription|incident\s+logs?)\b",
        r"\b(open|closed|resolved|status|sla\s+tier|region)\b",
        r"\blist\s+all\b",
        r"\bhow\s+many\b",
    ]
]


def _now() -> int:
    return int(time.time())


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _json_loads(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _normalize_prompt(prompt: str) -> str:
    return " ".join(prompt.casefold().strip().split())


def _safe_attributes(attributes: dict[str, Any] | None) -> dict[str, Any]:
    if not attributes:
        return {}
    safe: dict[str, Any] = {}
    for key, value in attributes.items():
        if key in {"created_at", "expires_at", "exact_hash", "prompt_cache_key", "ttl_seconds"}:
            continue
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            safe[str(key)] = value
        else:
            safe[str(key)] = str(value)
    return safe


def build_prompt_cache_key(prompt: str, route: CacheRoute, attributes: dict[str, Any] | None) -> str:
    """Build a stable, human-debuggable cache key seed for exact cache hits.

    The final storage key is still hashed, but this deterministic payload is
    included in cache metadata so repeated prompts can be inspected and reasoned
    about without depending on Redis semantic matching behavior.
    """

    payload = {
        "prompt": _normalize_prompt(prompt),
        "route": route.value,
        "attributes": _safe_attributes(attributes),
    }
    return _json_dumps(payload)


def _build_exact_hash(prompt: str, route: CacheRoute, attributes: dict[str, Any] | None) -> str:
    return hashlib.sha256(build_prompt_cache_key(prompt, route, attributes).encode("utf-8")).hexdigest()


def _extract_entry_id(entry: dict[str, Any]) -> str | None:
    for key in ("id", "entry_id", "entryId", "cacheEntryId"):
        value = entry.get(key)
        if value:
            return str(value)
    return None


def _extract_entry_attributes(entry: dict[str, Any]) -> dict[str, Any]:
    attrs = entry.get("attributes") or entry.get("metadata") or {}
    return dict(attrs) if isinstance(attrs, dict) else {}


def _extract_similarity(entry: dict[str, Any]) -> float | None:
    for key in ("similarity", "similarity_score", "similarityScore", "score", "distance"):
        value = entry.get(key)
        if value is None:
            continue
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if key == "distance":
            return max(0.0, min(1.0, 1.0 - score))
        return max(0.0, min(1.0, score))
    return None


def _extract_response(entry: dict[str, Any]) -> Any:
    for key in ("response", "completion", "answer", "value"):
        if key in entry:
            value = entry[key]
            return _json_loads(value) if isinstance(value, str) else value
    if "data" in entry:
        return entry["data"]
    return entry


def _entries_from_response(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [dict(item) for item in data if isinstance(item, dict)]
    if not isinstance(data, dict):
        return []
    for key in ("entries", "results", "matches", "data", "items"):
        value = data.get(key)
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, dict)]
    return [data] if any(key in data for key in ("response", "completion", "answer", "value")) else []


def is_safety_critical_query(query: str, metadata: dict[str, Any] | None = None) -> bool:
    if any(pattern.search(query) for pattern in SAFETY_CRITICAL_PATTERNS):
        return True
    metadata = metadata or {}
    if metadata.get("escalation_flag") or metadata.get("safety_critical"):
        return True
    severity = str(metadata.get("severity") or metadata.get("severity_priority") or "").lower()
    return severity in {"critical", "p0"}


def cache_route_for_query(mode: str, query: str, metadata: dict[str, Any] | None = None) -> CacheRoute:
    if is_safety_critical_query(query, metadata):
        return CacheRoute.SAFETY_CRITICAL
    normalized_mode = str(mode or "").lower()
    if normalized_mode == "vector":
        return CacheRoute.RAG
    if normalized_mode == "sql":
        return CacheRoute.SQL
    if normalized_mode == "agent" and any(pattern.search(query) for pattern in SQL_ROUTE_PATTERNS):
        return CacheRoute.SQL
    if normalized_mode == "agent":
        return CacheRoute.RAG
    return CacheRoute(normalized_mode) if normalized_mode in {item.value for item in CacheRoute} else CacheRoute.AGENT


class SemanticCache:
    """Redis-backed exact and semantic cache for retrieval responses."""

    def __init__(self) -> None:
        self._redis_client: Any = None
        self._process_cache: dict[str, _ProcessCacheEntry] = {}

    @property
    def enabled(self) -> bool:
        return bool(settings.enable_semantic_cache)

    @property
    def _langcache_enabled(self) -> bool:
        return bool(settings.redis_api_url and settings.redis_api_key and settings.redis_store_id)

    @property
    def _redis_exact_enabled(self) -> bool:
        return bool(settings.redis_url and redis_async is not None)

    def ttl_for_route(self, route: CacheRoute) -> int:
        if route == CacheRoute.RAG:
            return max(0, settings.cache_rag_ttl_seconds)
        if route == CacheRoute.SQL:
            return max(0, settings.cache_sql_ttl_seconds)
        if route == CacheRoute.SAFETY_CRITICAL:
            return max(0, settings.cache_safety_critical_ttl_seconds)
        return max(0, settings.cache_ttl_seconds)

    def _cache_key(self, exact_hash: str) -> str:
        return f"semantic_cache:exact:{exact_hash}"

    def _get_redis_client(self) -> Any:
        if not self._redis_exact_enabled:
            return None
        if self._redis_client is None:
            self._redis_client = redis_async.Redis.from_url(settings.redis_url, decode_responses=True)
        return self._redis_client

    def _cache_attributes(
        self,
        route: CacheRoute,
        exact_hash: str,
        prompt_cache_key: str,
        ttl_seconds: int,
        attributes: dict[str, Any] | None,
    ) -> dict[str, Any]:
        now = _now()
        return {
            **_safe_attributes(attributes),
            "cache_route": route.value,
            "exact_hash": exact_hash,
            "prompt_cache_key": prompt_cache_key,
            "ttl_seconds": ttl_seconds,
            "created_at": now,
            "expires_at": now + ttl_seconds,
            "redis_service_name": settings.redis_service_name or "",
            "redis_database_name": settings.redis_database_name or "",
        }

    def _get_process_cache(self, exact_hash: str) -> SemanticCacheHit | None:
        entry = self._process_cache.get(exact_hash)
        if entry is None:
            return None
        if entry.expires_at <= _now():
            self._process_cache.pop(exact_hash, None)
            return None
        return SemanticCacheHit(
            strategy="prompt_cache_key",
            response=entry.response,
            exact_hash=exact_hash,
            attributes=entry.attributes,
        )

    def _set_process_cache(self, exact_hash: str, response: Any, attributes: dict[str, Any], ttl_seconds: int) -> None:
        self._process_cache[exact_hash] = _ProcessCacheEntry(
            response=response,
            attributes=attributes,
            expires_at=_now() + ttl_seconds,
        )

    def _entry_is_fresh(self, attributes: dict[str, Any]) -> bool:
        expires_at = attributes.get("expires_at")
        try:
            return int(expires_at) > _now()
        except (TypeError, ValueError):
            return True

    def _langcache_url(self, path: str) -> str:
        base_url = str(settings.redis_api_url or "").rstrip("/") + "/"
        return urljoin(base_url, path.lstrip("/"))

    def _langcache_request(self, path: str, payload: dict[str, Any], method: str = "POST") -> Any:
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            self._langcache_url(path),
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {settings.redis_api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        timeout = max(0.5, settings.semantic_cache_timeout_seconds)
        with urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
            return json.loads(response_body) if response_body else {}

    async def _search_langcache(
        self,
        prompt: str,
        route: CacheRoute,
        attributes: dict[str, Any],
        strategy: str,
    ) -> SemanticCacheHit | None:
        if not self._langcache_enabled:
            return None

        payload = {
            "prompt": prompt,
            "attributes": attributes,
            "searchStrategies": [strategy],
            "similarityThreshold": settings.cache_similarity_threshold,
            "limit": 1,
        }

        paths = [
            f"/v1/caches/{settings.redis_store_id}/entries/search",
            f"/v1/stores/{settings.redis_store_id}/entries/search",
            f"/stores/{settings.redis_store_id}/entries/search",
        ]

        for path in paths:
            try:
                data = await asyncio.to_thread(self._langcache_request, path, payload)
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
                logger.debug("semantic cache search failed path=%s strategy=%s error=%s", path, strategy, exc)
                continue

            for entry in _entries_from_response(data):
                entry_attributes = _extract_entry_attributes(entry)
                if not self._entry_is_fresh(entry_attributes):
                    continue
                if entry_attributes.get("cache_route") != route.value:
                    continue
                if strategy.lower() == "exact" and entry_attributes.get("exact_hash") != attributes.get("exact_hash"):
                    continue
                similarity_score = _extract_similarity(entry)
                if strategy.lower() == "semantic" and similarity_score is not None:
                    if similarity_score < settings.cache_similarity_threshold:
                        continue
                return SemanticCacheHit(
                    strategy=strategy.lower(),
                    response=_extract_response(entry),
                    exact_hash=str(entry_attributes.get("exact_hash") or attributes.get("exact_hash")),
                    similarity_score=similarity_score,
                    entry_id=_extract_entry_id(entry),
                    attributes=entry_attributes,
                    raw_entry=entry,
                )
        return None

    async def _store_langcache(self, prompt: str, response: Any, attributes: dict[str, Any]) -> str | None:
        if not self._langcache_enabled:
            return None

        response_payload = response if isinstance(response, str) else _json_dumps(response)
        payload = {
            "prompt": prompt,
            "response": response_payload,
            "attributes": attributes,
        }

        paths = [
            f"/v1/caches/{settings.redis_store_id}/entries",
            f"/v1/stores/{settings.redis_store_id}/entries",
            f"/stores/{settings.redis_store_id}/entries",
        ]

        for path in paths:
            try:
                data = await asyncio.to_thread(self._langcache_request, path, payload)
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
                logger.debug("semantic cache store failed path=%s error=%s", path, exc)
                continue
            return _extract_entry_id(data) if isinstance(data, dict) else None
        return None

    async def get(
        self,
        prompt: str,
        route: CacheRoute,
        attributes: dict[str, Any] | None = None,
    ) -> SemanticCacheHit | None:
        started = time.perf_counter()
        if not self.enabled:
            logger.info("semantic_cache_bypassed", extra={"reason": "disabled", "route": route.value})
            return None

        ttl_seconds = self.ttl_for_route(route)
        if ttl_seconds <= 0:
            logger.info("semantic_cache_bypassed", extra={"reason": "ttl_disabled", "route": route.value})
            return None

        prompt_cache_key = build_prompt_cache_key(prompt, route, attributes)
        exact_hash = hashlib.sha256(prompt_cache_key.encode("utf-8")).hexdigest()
        cache_attributes = self._cache_attributes(route, exact_hash, prompt_cache_key, ttl_seconds, attributes)

        process_hit = self._get_process_cache(exact_hash)
        if process_hit:
            logger.info(
                "semantic_cache_hit",
                extra={"strategy": process_hit.strategy, "route": route.value, "duration_ms": int((time.perf_counter() - started) * 1000)},
            )
            return process_hit

        redis_client = self._get_redis_client()
        if redis_client is not None:
            try:
                raw = await redis_client.get(self._cache_key(exact_hash))
                if raw:
                    payload = _json_loads(raw)
                    logger.info(
                        "semantic_cache_hit",
                        extra={"strategy": "exact", "route": route.value, "duration_ms": int((time.perf_counter() - started) * 1000)},
                    )
                    return SemanticCacheHit(
                        strategy="exact",
                        response=payload.get("response") if isinstance(payload, dict) else payload,
                        exact_hash=exact_hash,
                        attributes=dict(payload.get("attributes") or {}) if isinstance(payload, dict) else {},
                    )
            except Exception as exc:
                logger.debug("redis exact cache lookup failed error=%s", exc)

        exact_hit = await self._search_langcache(prompt, route, cache_attributes, "exact")
        if exact_hit:
            logger.info(
                "semantic_cache_hit",
                extra={"strategy": exact_hit.strategy, "route": route.value, "duration_ms": int((time.perf_counter() - started) * 1000)},
            )
            return exact_hit

        semantic_hit = await self._search_langcache(prompt, route, cache_attributes, "semantic")
        if semantic_hit:
            logger.info(
                "semantic_cache_hit",
                extra={"strategy": semantic_hit.strategy, "route": route.value, "duration_ms": int((time.perf_counter() - started) * 1000)},
            )
            return semantic_hit
        logger.info("semantic_cache_miss", extra={"route": route.value, "duration_ms": int((time.perf_counter() - started) * 1000)})
        return None

    async def set(
        self,
        prompt: str,
        response: Any,
        route: CacheRoute,
        attributes: dict[str, Any] | None = None,
    ) -> SemanticCacheStoreResult:
        started = time.perf_counter()
        if not self.enabled:
            return SemanticCacheStoreResult(False, "semantic cache disabled")

        ttl_seconds = self.ttl_for_route(route)
        if ttl_seconds <= 0:
            return SemanticCacheStoreResult(False, "ttl disabled for route")

        prompt_cache_key = build_prompt_cache_key(prompt, route, attributes)
        exact_hash = hashlib.sha256(prompt_cache_key.encode("utf-8")).hexdigest()
        cache_attributes = self._cache_attributes(route, exact_hash, prompt_cache_key, ttl_seconds, attributes)
        value = {"response": response, "attributes": cache_attributes}

        self._set_process_cache(exact_hash, response, cache_attributes, ttl_seconds)

        redis_client = self._get_redis_client()
        redis_stored = False
        if redis_client is not None:
            try:
                await redis_client.setex(self._cache_key(exact_hash), ttl_seconds, _json_dumps(value))
                redis_stored = True
            except Exception as exc:
                logger.debug("redis exact cache store failed error=%s", exc)

        entry_id = await self._store_langcache(prompt, response, cache_attributes)
        if redis_stored or entry_id:
            logger.info(
                "semantic_cache_store_succeeded",
                extra={"route": route.value, "duration_ms": int((time.perf_counter() - started) * 1000)},
            )
            return SemanticCacheStoreResult(True, "stored", exact_hash=exact_hash, entry_id=entry_id)
        logger.info(
            "semantic_cache_store_succeeded",
            extra={
                "route": route.value,
                "duration_ms": int((time.perf_counter() - started) * 1000),
                "strategy": "prompt_cache_key",
            },
        )
        logger.debug(
            "semantic_cache_external_store_skipped",
            extra={"route": route.value, "duration_ms": int((time.perf_counter() - started) * 1000)},
        )
        return SemanticCacheStoreResult(True, "stored in process prompt_cache_key fallback", exact_hash=exact_hash)


_SEMANTIC_CACHE: SemanticCache | None = None


def get_semantic_cache() -> SemanticCache:
    global _SEMANTIC_CACHE
    if _SEMANTIC_CACHE is None:
        _SEMANTIC_CACHE = SemanticCache()
    return _SEMANTIC_CACHE
