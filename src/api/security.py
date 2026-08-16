"""Password hashing and JWT issuance/verification for the API layer."""
from __future__ import annotations

import datetime as dt
from typing import Optional

import jwt
from passlib.context import CryptContext

from src.utils.config import AuthConfig

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return _pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return _pwd_context.verify(password, password_hash)


def create_access_token(subject: str, auth_config: AuthConfig) -> str:
    now = dt.datetime.utcnow()
    payload = {
        "sub": subject,
        "iat": now,
        "exp": now + dt.timedelta(minutes=auth_config.access_token_expire_minutes),
    }
    return jwt.encode(payload, auth_config.jwt_secret, algorithm=auth_config.jwt_algorithm)


def decode_access_token(token: str, auth_config: AuthConfig) -> Optional[str]:
    """Return the token subject (user email) if valid, else None."""
    try:
        payload = jwt.decode(token, auth_config.jwt_secret, algorithms=[auth_config.jwt_algorithm])
        return payload.get("sub")
    except jwt.PyJWTError:
        return None
