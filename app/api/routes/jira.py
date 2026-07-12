from datetime import datetime
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.core.auth import AuthenticatedUser, require_permissions
from app.core.config import settings
from app.core.logging import get_logger

router = APIRouter(prefix="/jira", tags=["jira"])
logger = get_logger(__name__)


def _jira_issue_url(issue_key: str) -> str | None:
    if not settings.jira_url or not issue_key:
        return None
    return f"{settings.jira_url.rstrip('/')}/browse/{issue_key}"


def _format_created(value: str | None) -> str | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.date().isoformat()
    except ValueError:
        return value


def _issue_from_jira(raw_issue: dict[str, Any]) -> dict[str, Any]:
    fields = dict(raw_issue.get("fields") or {})
    status_value = dict(fields.get("status") or {})
    assignee = dict(fields.get("assignee") or {})
    priority = dict(fields.get("priority") or {})
    issue_type = dict(fields.get("issuetype") or {})
    issue_key = str(raw_issue.get("key") or "")
    return {
        "key": issue_key,
        "summary": fields.get("summary") or "Untitled Jira issue",
        "status": status_value.get("name") or "Unknown",
        "assignee": assignee.get("displayName") or "Unassigned",
        "created": _format_created(fields.get("created")),
        "priority": priority.get("name"),
        "issue_type": issue_type.get("name"),
        "url": _jira_issue_url(issue_key),
    }


@router.get(
    "/issues",
    summary="List Jira issues",
    description="Return recent Jira issues from the configured escalation project.",
)
async def list_jira_issues(
    limit: int = Query(default=50, ge=1, le=100),
    _current_user: AuthenticatedUser = Depends(require_permissions("view:evaluation")),
) -> dict[str, Any]:
    """Return existing Jira issues for the configured project."""

    if not settings.jira_url or not settings.jira_username or not settings.jira_api_token or not settings.jira_project_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "jira_not_configured",
                "detail": "Jira URL, username, API token, and project key must be configured.",
            },
        )

    jql = f'project = "{settings.jira_project_key}" ORDER BY created DESC'
    params = [
        ("jql", jql),
        ("maxResults", str(limit)),
        ("fields", "summary"),
        ("fields", "status"),
        ("fields", "assignee"),
        ("fields", "created"),
        ("fields", "priority"),
        ("fields", "issuetype"),
    ]
    url = f"{settings.jira_url.rstrip('/')}/rest/api/3/search/jql"

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(
                url,
                params=params,
                auth=(settings.jira_username, settings.jira_api_token),
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.warning("Jira issue list request failed status=%s body=%s", exc.response.status_code, exc.response.text[:500])
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"error": "jira_request_failed", "detail": "Jira rejected the issue list request."},
        ) from exc
    except httpx.HTTPError as exc:
        logger.warning("Jira issue list request failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"error": "jira_unreachable", "detail": "Unable to reach Jira."},
        ) from exc

    payload = response.json()
    issues = [_issue_from_jira(issue) for issue in payload.get("issues", [])]
    return {
        "project_key": settings.jira_project_key,
        "count": len(issues),
        "issues": issues,
    }
