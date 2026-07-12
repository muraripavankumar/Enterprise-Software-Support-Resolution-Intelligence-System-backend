from dataclasses import dataclass, field
import time
from typing import Any, Callable

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient
from jwt.exceptions import InvalidTokenError, PyJWKClientError

from app.core.config import ConfigurationError, settings
from app.core.logging import get_logger

logger = get_logger(__name__)

bearer_scheme = HTTPBearer(auto_error=False)
_JWK_CLIENT: PyJWKClient | None = None

ROLE_PRIORITY = ["customer_user", "support_agent", "support_manager", "admin"]


@dataclass(frozen=True)
class AuthenticatedUser:
    """Verified Auth0 principal extracted from an access token."""

    sub: str
    permissions: list[str] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    primary_role: str = "customer_user"
    token_claims: dict[str, Any] = field(default_factory=dict)


DEV_AUTH_USER = AuthenticatedUser(
    sub="local-dev-user",
    permissions=[
        "ask:support_query",
        "read:tickets",
        "read:incidents",
        "trigger:escalation",
        "view:evaluation",
    ],
    roles=["admin"],
    primary_role="admin",
    token_claims={"auth_disabled": True},
)


def _auth_error(detail: str, status_code: int = status.HTTP_401_UNAUTHORIZED) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"error": "authentication_failed", "detail": detail},
        headers={"WWW-Authenticate": "Bearer"},
    )


def _permission_error(missing: list[str]) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail={
            "error": "insufficient_permissions",
            "detail": "Missing required permission(s): " + ", ".join(missing),
        },
    )


def _get_jwk_client() -> PyJWKClient:
    global _JWK_CLIENT
    settings.validate_for_auth()
    if _JWK_CLIENT is None:
        _JWK_CLIENT = PyJWKClient(settings.auth0_jwks_url)
    return _JWK_CLIENT


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item for item in value.split() if item]
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)] if str(value).strip() else []


def _extract_permissions(claims: dict[str, Any]) -> list[str]:
    permissions = set(_as_list(claims.get(settings.auth0_permissions_claim)))
    permissions.update(_as_list(claims.get("permissions")))
    permissions.update(_as_list(claims.get("scope")))
    return sorted(permissions)


def _extract_roles(claims: dict[str, Any]) -> list[str]:
    roles = set(_as_list(claims.get(settings.auth0_roles_claim)))
    roles.update(_as_list(claims.get("roles")))
    roles.update(_as_list(claims.get("role")))
    return sorted(roles)


def _primary_role(roles: list[str]) -> str:
    normalized = {role.lower() for role in roles}
    for role in reversed(ROLE_PRIORITY):
        if role in normalized:
            return role
    return "customer_user"


def _decode_token(token: str) -> dict[str, Any]:
    try:
        jwks_started = time.perf_counter()
        signing_key = _get_jwk_client().get_signing_key_from_jwt(token)
        logger.info("auth0_jwks_signing_key_resolved", extra={"duration_ms": int((time.perf_counter() - jwks_started) * 1000)})
        return dict(
            jwt.decode(
                token,
                signing_key.key,
                algorithms=settings.auth0_algorithm_list,
                audience=settings.auth0_audience,
                issuer=settings.auth0_issuer,
            )
        )
    except ConfigurationError as exc:
        logger.exception("Auth0 configuration is incomplete")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "auth_configuration_error", "detail": str(exc)},
        ) from exc
    except (InvalidTokenError, PyJWKClientError, ValueError) as exc:
        logger.warning("JWT validation failed: %s", exc)
        raise _auth_error("Invalid or expired access token.") from exc


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> AuthenticatedUser:
    """Validate the bearer token and return the Auth0 principal."""

    if not settings.enable_auth0:
        if not settings.is_local_env:
            logger.error("Auth0 disabled in non-local environment app_env=%s", settings.app_env)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "auth_configuration_error",
                    "detail": "Auth bypass is only allowed in local environments.",
                },
            )
        return DEV_AUTH_USER

    if credentials is None or credentials.scheme.lower() != "bearer" or not credentials.credentials:
        raise _auth_error("Missing bearer access token.")

    claims = _decode_token(credentials.credentials)
    subject = str(claims.get("sub") or "").strip()
    if not subject:
        raise _auth_error("Token is missing subject claim.")

    roles = _extract_roles(claims)
    return AuthenticatedUser(
        sub=subject,
        permissions=_extract_permissions(claims),
        roles=roles,
        primary_role=_primary_role(roles),
        token_claims=claims,
    )


async def validate_token(
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> AuthenticatedUser:
    """Reusable FastAPI dependency that validates an Auth0 access token."""

    return current_user


def require_role(role: str) -> Callable[[AuthenticatedUser], AuthenticatedUser]:
    """FastAPI dependency factory for role-based authorization checks."""

    normalized_required_role = role.strip().lower()

    async def dependency(current_user: AuthenticatedUser = Depends(validate_token)) -> AuthenticatedUser:
        normalized_roles = {available_role.strip().lower() for available_role in current_user.roles}
        if normalized_required_role not in normalized_roles:
            logger.warning(
                "Role check failed sub=%s required=%s available_roles=%s audience=%s",
                current_user.sub,
                normalized_required_role,
                current_user.roles,
                current_user.token_claims.get("aud"),
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "insufficient_role",
                    "detail": f"Missing required role: {normalized_required_role}",
                },
            )
        return current_user

    return dependency


def require_permissions(*required_permissions: str) -> Callable[[AuthenticatedUser], AuthenticatedUser]:
    """FastAPI dependency factory for route-level RBAC permission checks."""

    async def dependency(current_user: AuthenticatedUser = Depends(get_current_user)) -> AuthenticatedUser:
        missing = [
            permission
            for permission in required_permissions
            if permission not in set(current_user.permissions)
        ]
        if missing:
            logger.warning(
                "Permission check failed sub=%s missing=%s available_permissions=%s roles=%s audience=%s scope=%s",
                current_user.sub,
                missing,
                current_user.permissions,
                current_user.roles,
                current_user.token_claims.get("aud"),
                current_user.token_claims.get("scope"),
            )
            raise _permission_error(missing)
        return current_user

    return dependency


def auth_metadata(current_user: AuthenticatedUser) -> dict[str, Any]:
    """Minimal non-secret identity metadata passed into the orchestration state."""

    return {
        "jwt_sub": current_user.sub,
        "user_id": current_user.sub,
        "user_role": current_user.primary_role,
        "roles": current_user.roles,
        "permissions": current_user.permissions,
    }
