from typing import Any

from fastapi import APIRouter, Depends

from app.core.auth import AuthenticatedUser, validate_token
from app.core.config import settings

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get(
    "/me",
    summary="Current principal",
    description="Return the Auth0 principal after backend JWT validation.",
)
async def me(current_user: AuthenticatedUser = Depends(validate_token)) -> dict[str, Any]:
    """Return the backend-validated principal and authorization context."""

    return {
        "sub": current_user.sub,
        "roles": current_user.roles,
        "primary_role": current_user.primary_role,
        "permissions": current_user.permissions,
        "audience": current_user.token_claims.get("aud"),
        "issuer": current_user.token_claims.get("iss"),
        "expires_at": current_user.token_claims.get("exp"),
    }


@router.get(
    "/diagnostics",
    summary="Auth diagnostics",
    description="Return non-secret Auth0 configuration and token-claim diagnostics.",
)
async def diagnostics(current_user: AuthenticatedUser = Depends(validate_token)) -> dict[str, Any]:
    """Non-secret Auth0 diagnostics for local/admin troubleshooting."""

    return {
        "auth0_enabled": settings.enable_auth0,
        "expected_audience": settings.auth0_audience,
        "expected_issuer": settings.auth0_issuer,
        "roles_claim": settings.auth0_roles_claim,
        "permissions_claim": settings.auth0_permissions_claim,
        "principal": {
            "sub": current_user.sub,
            "roles": current_user.roles,
            "primary_role": current_user.primary_role,
            "permissions": current_user.permissions,
            "token_audience": current_user.token_claims.get("aud"),
            "token_scope": current_user.token_claims.get("scope"),
        },
    }
