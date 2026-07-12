"""Import ERIS path-coverage dataset items into Langfuse.

Usage:
    python scripts/import_langfuse_dataset.py --dry-run
    python scripts/import_langfuse_dataset.py
    python scripts/import_langfuse_dataset.py --dataset-name eris-path-coverage-v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langfuse import Langfuse


BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent
DEFAULT_DATASET_PATH = PROJECT_ROOT / "docs" / "software_support_dataset_items.json"
DEFAULT_DATASET_NAME = "eris-software-support-path-coverage"

VALID_SEVERITIES = {"P0", "P1", "P2", "P3"}
VALID_ROUTES = {"rag", "sql", "hybrid", "high_risk", "clarification", "chitchat"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}

INPUT_SCHEMA = {
    "type": "object",
    "required": ["ticket_text"],
    "properties": {
        "ticket_text": {"type": "string"},
        "customer_tier": {"type": "string"},
        "prior_context": {"type": "string"},
    },
    "additionalProperties": True,
}

EXPECTED_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["intent", "route_decision", "severity", "escalate", "reasoning"],
    "properties": {
        "intent": {"type": "string"},
        "route_decision": {"type": "string", "enum": sorted(VALID_ROUTES)},
        "severity": {"type": "string", "enum": sorted(VALID_SEVERITIES)},
        "escalate": {"type": "boolean"},
        "reasoning": {"type": "string"},
    },
    "additionalProperties": True,
}


def _load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(BACKEND_ROOT / ".env")


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _load_items(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON array")
    return [dict(item) for item in payload]


def _validate_item(item: dict[str, Any], index: int) -> None:
    location = f"item[{index}]"
    for key in ("input", "expected_output", "metadata"):
        if not isinstance(item.get(key), dict):
            raise ValueError(f"{location}.{key} must be an object")

    input_payload = item["input"]
    expected_output = item["expected_output"]
    metadata = item["metadata"]

    if not str(input_payload.get("ticket_text") or "").strip():
        raise ValueError(f"{location}.input.ticket_text is required")
    if not str(expected_output.get("intent") or "").strip():
        raise ValueError(f"{location}.expected_output.intent is required")
    if expected_output.get("route_decision") not in VALID_ROUTES:
        raise ValueError(f"{location}.expected_output.route_decision must be one of {sorted(VALID_ROUTES)}")
    if expected_output.get("severity") not in VALID_SEVERITIES:
        raise ValueError(f"{location}.expected_output.severity must be one of {sorted(VALID_SEVERITIES)}")
    if not isinstance(expected_output.get("escalate"), bool):
        raise ValueError(f"{location}.expected_output.escalate must be boolean")
    if not str(expected_output.get("reasoning") or "").strip():
        raise ValueError(f"{location}.expected_output.reasoning is required")
    if not str(metadata.get("path_tested") or "").strip():
        raise ValueError(f"{location}.metadata.path_tested is required")
    if not str(metadata.get("source_doc_or_table") or "").strip():
        raise ValueError(f"{location}.metadata.source_doc_or_table is required")
    if metadata.get("difficulty") not in VALID_DIFFICULTIES:
        raise ValueError(f"{location}.metadata.difficulty must be one of {sorted(VALID_DIFFICULTIES)}")


def _validate_items(items: list[dict[str, Any]]) -> None:
    path_tests: set[str] = set()
    for index, item in enumerate(items):
        _validate_item(item, index)
        path_tested = str(item["metadata"]["path_tested"])
        if path_tested in path_tests:
            raise ValueError(f"Duplicate metadata.path_tested value: {path_tested}")
        path_tests.add(path_tested)


def _stable_item_id(dataset_name: str, path_tested: str) -> str:
    digest = hashlib.sha256(f"{dataset_name}:{path_tested}".encode("utf-8")).hexdigest()
    return digest[:32]


def _create_dataset_if_needed(client: Langfuse, dataset_name: str, description: str) -> None:
    try:
        client.get_dataset(dataset_name)
        print(f"Langfuse dataset already exists: {dataset_name}")
        return
    except Exception as exc:
        message = str(exc).lower()
        if "not found" not in message and "404" not in message:
            raise

    try:
        client.create_dataset(
            name=dataset_name,
            description=description,
            metadata={
                "project": "Enterprise Software Support & Resolution Intelligence System",
                "dataset_type": "langgraph_path_coverage",
            },
            input_schema=INPUT_SCHEMA,
            expected_output_schema=EXPECTED_OUTPUT_SCHEMA,
        )
        print(f"Created Langfuse dataset: {dataset_name}")
    except Exception as exc:
        message = str(exc).lower()
        if "already" in message or "exists" in message or "duplicate" in message or "409" in message:
            print(f"Langfuse dataset already exists: {dataset_name}")
            return
        raise


def _create_dataset_item(
    client: Langfuse,
    *,
    dataset_name: str,
    item: dict[str, Any],
    stable_ids: bool,
) -> bool:
    path_tested = str(item["metadata"]["path_tested"])
    kwargs: dict[str, Any] = {
        "dataset_name": dataset_name,
        "input": item["input"],
        "expected_output": item["expected_output"],
        "metadata": item["metadata"],
    }
    if stable_ids:
        kwargs["id"] = _stable_item_id(dataset_name, path_tested)

    try:
        client.create_dataset_item(**kwargs)
        return True
    except Exception as exc:
        message = str(exc).lower()
        if stable_ids and ("already" in message or "exists" in message or "duplicate" in message or "409" in message):
            print(f"Skipped existing dataset item: {path_tested}")
            return False
        raise


def _client_from_env() -> Langfuse:
    public_key = _require_env("LANGFUSE_PUBLIC_KEY")
    secret_key = _require_env("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST") or "https://cloud.langfuse.com"
    return Langfuse(public_key=public_key, secret_key=secret_key, host=host)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import ERIS dataset items into Langfuse.")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument(
        "--description",
        default="Path-coverage dataset for ERIS LangGraph support orchestration.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and summarize without writing to Langfuse.")
    parser.add_argument(
        "--no-stable-ids",
        action="store_true",
        help="Do not use deterministic item ids. Re-running may create duplicates.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_path = args.dataset_path.resolve()
    items = _load_items(dataset_path)
    _validate_items(items)

    path_tests = [str(item["metadata"]["path_tested"]) for item in items]
    print(f"Validated {len(items)} dataset items from {dataset_path}")
    print(f"Unique path_tested values: {len(set(path_tests))}")

    if args.dry_run:
        return 0

    _load_env()
    client = _client_from_env()
    _create_dataset_if_needed(client, args.dataset_name, args.description)

    created = 0
    skipped = 0
    for item in items:
        if _create_dataset_item(
            client,
            dataset_name=args.dataset_name,
            item=item,
            stable_ids=not args.no_stable_ids,
        ):
            created += 1
        else:
            skipped += 1

    flush = getattr(client, "flush", None)
    if callable(flush):
        flush()

    print(
        "Langfuse dataset import complete: "
        f"dataset={args.dataset_name}, created={created}, skipped={skipped}, total={len(items)}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Dataset import failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
