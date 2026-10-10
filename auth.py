from __future__ import annotations

from functools import lru_cache

import firebase_admin
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from firebase_admin import auth as firebase_auth
from firebase_admin import credentials

from core.config import load_settings

_bearer = HTTPBearer(auto_error=False)


@lru_cache(maxsize=1)
def _init_firebase_app() -> firebase_admin.App:
    s = load_settings()
    if not s.FIREBASE_SERVICE_ACCOUNT_JSON:
        raise RuntimeError(
            "FIREBASE_SERVICE_ACCOUNT_JSON is not set. Download a service-account "
            "key from Firebase console -> Project settings -> Service accounts, "
            "and set FIREBASE_SERVICE_ACCOUNT_JSON to its path in your .env."
        )
    cred = credentials.Certificate(s.FIREBASE_SERVICE_ACCOUNT_JSON)
    return firebase_admin.initialize_app(cred)


def verify_id_token(id_token: str) -> dict:
    """Verify a Firebase ID token and return its decoded claims. Raises
    firebase_admin.auth exceptions (or ValueError) on an invalid/expired token."""
    _init_firebase_app()
    return firebase_auth.verify_id_token(id_token, check_revoked=True)


async def require_auth(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict:
    """FastAPI dependency: verifies the `Authorization: Bearer <firebase-id-token>`
    header and returns the decoded token claims (uid, email, ...)."""
    if creds is None or not creds.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token. Sign in and pass the Firebase ID token as 'Authorization: Bearer <token>'.",
        )
    try:
        return verify_id_token(creds.credentials)
    except RuntimeError as e:
        # Server misconfiguration (no service account configured yet).
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
        )
