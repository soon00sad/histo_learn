"""Authentication endpoints: login (issues a JWT) and current-user lookup."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.api import security
from src.api.db import User
from src.api.deps import current_user, db_session, settings_dep
from src.api.schemas import LoginRequest, Token, UserOut
from src.utils.config import Settings
from src.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])


def _to_user_out(user: User) -> UserOut:
    return UserOut(email=user.email, full_name=user.full_name, role=user.role)


@router.post("/login", response_model=Token)
def login(
    payload: LoginRequest,
    session: Session = Depends(db_session),
    settings: Settings = Depends(settings_dep),
) -> Token:
    user = session.query(User).filter_by(email=payload.email).first()
    if user is None or not security.verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Неверный логин или пароль")

    token = security.create_access_token(user.email, settings.auth)
    logger.info("User %s logged in", user.email)
    return Token(access_token=token, user=_to_user_out(user))


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(current_user)) -> UserOut:
    return _to_user_out(user)
