"""Shared FastAPI dependencies: DB sessions, current-user auth, settings."""
from __future__ import annotations

from typing import Iterator

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from src.api import security
from src.api.db import User, get_session
from src.utils.config import Settings, get_settings

_bearer_scheme = HTTPBearer(auto_error=False)


def db_session() -> Iterator[Session]:
    session = get_session()
    try:
        yield session
    finally:
        session.close()


def settings_dep() -> Settings:
    return get_settings()


def current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    session: Session = Depends(db_session),
    settings: Settings = Depends(settings_dep),
) -> User:
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Требуется авторизация",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if credentials is None:
        raise unauthorized

    email = security.decode_access_token(credentials.credentials, settings.auth)
    if email is None:
        raise unauthorized

    user = session.query(User).filter_by(email=email).first()
    if user is None:
        raise unauthorized
    return user
