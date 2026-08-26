"""Minimal offline admin tool for user accounts — list, create, and reset
passwords directly against the SQLite DB. No self-registration and no
email-based password reset exist in the product (expected for a clinical
system perimeter, see docs) — this script is the "administrator resets a
doctor's password" path, run by whoever operates the deployment, not
exposed as an API endpoint.

Usage:
    python scripts/manage_users.py list
    python scripts/manage_users.py create-user --email doc@clinic.ru --full-name "Иванова И.И." --role doctor
    python scripts/manage_users.py reset-password --email doc@clinic.ru
"""
from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.api import security  # noqa: E402
from src.api.db import User, get_session, init_db  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def _prompt_password(confirm: bool = True) -> str:
    while True:
        password = getpass.getpass("Новый пароль: ")
        if len(password) < 8:
            print("Пароль должен быть не короче 8 символов.")
            continue
        if confirm and getpass.getpass("Повторите пароль: ") != password:
            print("Пароли не совпадают, попробуйте снова.")
            continue
        return password


def cmd_list() -> None:
    session = get_session()
    try:
        users = session.query(User).order_by(User.id).all()
        if not users:
            print("Пользователей нет.")
            return
        for user in users:
            print(f"{user.id:>3}  {user.email:<32}  {user.full_name:<28}  {user.role}")
    finally:
        session.close()


def cmd_create_user(email: str, full_name: str, role: str, password: str | None) -> None:
    session = get_session()
    try:
        if session.query(User).filter_by(email=email).first():
            print(f"Пользователь с email {email} уже существует.")
            return
        password = password or _prompt_password()
        session.add(User(email=email, full_name=full_name, role=role, password_hash=security.hash_password(password)))
        session.commit()
        print(f"Создан пользователь {email} ({role}).")
    finally:
        session.close()


def cmd_reset_password(email: str, password: str | None) -> None:
    session = get_session()
    try:
        user = session.query(User).filter_by(email=email).first()
        if user is None:
            print(f"Пользователь с email {email} не найден.")
            return
        password = password or _prompt_password()
        user.password_hash = security.hash_password(password)
        session.commit()
        print(f"Пароль для {email} обновлён.")
    finally:
        session.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="Список пользователей.")

    p_create = sub.add_parser("create-user", help="Завести нового пользователя (например, врача).")
    p_create.add_argument("--email", required=True)
    p_create.add_argument("--full-name", required=True)
    p_create.add_argument("--role", default="doctor")
    p_create.add_argument("--password", default=None, help="Если не указан — запрошен интерактивно (безопаснее).")

    p_reset = sub.add_parser("reset-password", help="Сбросить пароль существующего пользователя.")
    p_reset.add_argument("--email", required=True)
    p_reset.add_argument("--password", default=None, help="Если не указан — запрошен интерактивно (безопаснее).")

    args = parser.parse_args()
    init_db()

    if args.command == "list":
        cmd_list()
    elif args.command == "create-user":
        cmd_create_user(args.email, args.full_name, args.role, args.password)
    elif args.command == "reset-password":
        cmd_reset_password(args.email, args.password)


if __name__ == "__main__":
    main()
