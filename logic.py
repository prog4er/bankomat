from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import secrets
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Protocol

from ml.infer import build_features  # единый генератор признаков для train/infer

logger = logging.getLogger("atm")

PBKDF2_ITERS = 150_000  # учебное значение, не "платёжная криптография"


class SessionState(str, Enum):
    AWAITING_PIN = "AWAITING_PIN"
    ACTIVE = "ACTIVE"
    CANCELLED = "CANCELLED"
    TIMED_OUT = "TIMED_OUT"
    BLOCKED = "BLOCKED"
    CLOSED = "CLOSED"


class RiskScorer(Protocol):
    def score(self, features: dict[str, Any]) -> dict[str, Any]:
        """Возвращает dict: {risk: float, flagged: bool, model: str, details?: dict}"""
        ...


class ATMError(Exception):
    pass


class NotFoundError(ATMError):
    pass


class AuthError(ATMError):
    pass


class StateError(ATMError):
    pass


class ValidationError(ATMError):
    pass


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def new_salt() -> bytes:
    return secrets.token_bytes(16)


def hash_pin(pin: str, salt: bytes) -> bytes:
    # Хэширование PIN как учебный пример безопасного хранения секрета.
    # Не реализует EMV/PIN-block/ключи и не предназначено для продакшена платежей.
    return hashlib.pbkdf2_hmac("sha256", pin.encode("utf-8"), salt, PBKDF2_ITERS)


def verify_pin(pin: str, salt: bytes, expected_hash: bytes) -> bool:
    return hmac.compare_digest(hash_pin(pin, salt), expected_hash)


def init_db(conn: sqlite3.Connection, schema_sql: str) -> None:
    """Упрощённая миграция: если schema_migrations отсутствует — применяем schema.sql."""
    conn.execute("PRAGMA foreign_keys = ON;")
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_migrations';"
    ).fetchone()
    if row is None:
        conn.executescript(schema_sql)
        conn.execute(
            "INSERT INTO schema_migrations(version, applied_at) VALUES(?, ?);",
            (1, _iso(_utc_now())),
        )
        conn.commit()


def seed_demo_data(conn: sqlite3.Connection) -> None:
    """Создаёт 2 учебных пользователя/карты/счёта для сценариев (включая перевод)."""
    cur = conn.execute("SELECT COUNT(*) AS n FROM users;").fetchone()
    if int(cur["n"]) > 0:
        return

    now = _iso(_utc_now())

    def _create_user(full_name: str, pin: str) -> int:
        salt = new_salt()
        p_hash = hash_pin(pin, salt)
        conn.execute(
            """
            INSERT INTO users(full_name, pin_salt, pin_hash, pin_failed_attempts, is_blocked, created_at)
            VALUES(?, ?, ?, 0, 0, ?);
            """,
            (full_name, salt, p_hash, now),
        )
        return int(conn.execute("SELECT last_insert_rowid() AS id;").fetchone()["id"])

    def _create_card(user_id: int, card_token: str) -> int:
        conn.execute(
            """
            INSERT INTO cards(card_token, user_id, is_active, created_at)
            VALUES(?, ?, 1, ?);
            """,
            (card_token, user_id, now),
        )
        return int(conn.execute("SELECT last_insert_rowid() AS id;").fetchone()["id"])

    def _create_account(user_id: int, balance_cents: int, limit_cents: int) -> int:
        conn.execute(
            """
            INSERT INTO accounts(user_id, currency, balance_cents, daily_withdraw_limit_cents,
                                 daily_withdrawn_cents, daily_withdrawn_date, created_at)
            VALUES(?, 'RUB', ?, ?, 0, '', ?);
            """,
            (user_id, int(balance_cents), int(limit_cents), now),
        )
        return int(conn.execute("SELECT last_insert_rowid() AS id;").fetchone()["id"])

    with conn:
        u1 = _create_user("Учебный пользователь 1", "1234")
        _create_card(u1, "CARD-0001")
        _create_account(u1, balance_cents=50_000, limit_cents=30_000)

        u2 = _create_user("Учебный пользователь 2", "4321")
        _create_card(u2, "CARD-0002")
        _create_account(u2, balance_cents=10_000, limit_cents=30_000)


def _log_event(conn: sqlite3.Connection, session_id: str, event_type: str, payload: dict[str, Any] | None = None) -> None:
    payload = payload or {}
    conn.execute(
        "INSERT INTO events(session_id, event_type, payload_json, created_at) VALUES(?, ?, ?, ?);",
        (session_id, event_type, json.dumps(payload, ensure_ascii=False), _iso(_utc_now())),
    )


def _get_session(conn: sqlite3.Connection, session_id: str) -> sqlite3.Row:
    row = conn.execute(
        """
        SELECT s.session_id, s.card_id, s.authenticated_user_id, s.state, s.created_at,
               s.last_activity_at, s.expires_at, s.closed_at,
               c.user_id AS card_user_id
        FROM sessions s
        JOIN cards c ON c.card_id = s.card_id
        WHERE s.session_id = ?;
        """,
        (session_id,),
    ).fetchone()
    if row is None:
        raise NotFoundError("Сессия не найдена.")
    return row


def _touch_or_timeout(conn: sqlite3.Connection, session_row: sqlite3.Row, ttl_sec: int) -> None:
    now = _utc_now()
    expires_at = datetime.fromisoformat(session_row["expires_at"])
    if now > expires_at and session_row["state"] == SessionState.ACTIVE.value:
        with conn:
            conn.execute(
                "UPDATE sessions SET state=?, closed_at=? WHERE session_id=?;",
                (SessionState.TIMED_OUT.value, _iso(now), session_row["session_id"]),
            )
            _log_event(conn, session_row["session_id"], "SESSION_TIMED_OUT", {})
        raise StateError("Сессия завершена по тайм‑ауту.")

    # sliding TTL
    new_expires = now + timedelta(seconds=ttl_sec)
    conn.execute(
        "UPDATE sessions SET last_activity_at=?, expires_at=? WHERE session_id=?;",
        (_iso(now), _iso(new_expires), session_row["session_id"]),
    )


def start_session(conn: sqlite3.Connection, card_token: str, pin: str, ttl_sec: int) -> dict[str, Any]:
    """Сценарий 'карта+PIN'. PIN не логируем."""
    row = conn.execute(
        """
        SELECT c.card_id, c.user_id, c.is_active,
               u.full_name, u.pin_salt, u.pin_hash, u.is_blocked, u.pin_failed_attempts
        FROM cards c
        JOIN users u ON u.user_id = c.user_id
        WHERE c.card_token = ?;
        """,
        (card_token,),
    ).fetchone()

    if row is None:
        raise NotFoundError("Карта не найдена (учебный токен).")

    if int(row["is_active"]) == 0 or int(row["is_blocked"]) == 1:
        raise AuthError("Карта/пользователь заблокированы.")

    # проверка PIN
    ok = verify_pin(pin, row["pin_salt"], row["pin_hash"])
    if not ok:
        fails = int(row["pin_failed_attempts"]) + 1
        with conn:
            conn.execute(
                "UPDATE users SET pin_failed_attempts=? WHERE user_id=?;",
                (fails, int(row["user_id"])),
            )
            # блокируем после 3 попыток
            if fails >= 3:
                conn.execute("UPDATE users SET is_blocked=1 WHERE user_id=?;", (int(row["user_id"]),))
                conn.execute("UPDATE cards SET is_active=0 WHERE card_id=?;", (int(row["card_id"]),))
        raise AuthError("Неверный PIN. (Учебно) После 3 ошибок карта блокируется.")

    # успешная авторизация
    session_id = uuid.uuid4().hex
    now = _utc_now()
    expires = now + timedelta(seconds=ttl_sec)

    with conn:
        conn.execute("UPDATE users SET pin_failed_attempts=0 WHERE user_id=?;", (int(row["user_id"]),))
        conn.execute(
            """
            INSERT INTO sessions(session_id, card_id, authenticated_user_id, state,
                                 created_at, last_activity_at, expires_at, closed_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, NULL);
            """,
            (session_id, int(row["card_id"]), int(row["user_id"]), SessionState.ACTIVE.value,
             _iso(now), _iso(now), _iso(expires)),
        )
        _log_event(conn, session_id, "SESSION_STARTED", {"card_token": card_token})

    return {"session_id": session_id, "state": SessionState.ACTIVE.value, "message": f"Здравствуйте, {row['full_name']}!"}


def get_session_info(conn: sqlite3.Connection, session_id: str, ttl_sec: int) -> dict[str, Any]:
    s = _get_session(conn, session_id)
    # если ACTIVE — проверяем тайм‑аут и обновляем ttl
    if s["state"] == SessionState.ACTIVE.value:
        with conn:
            _touch_or_timeout(conn, s, ttl_sec)
        s = _get_session(conn, session_id)

    now = _utc_now()
    try:
        expires_at = datetime.fromisoformat(s["expires_at"])
        seconds_left = max(0, int((expires_at - now).total_seconds()))
    except Exception:
        seconds_left = 0

    return {"session_id": session_id, "state": s["state"], "seconds_left": seconds_left}


def _get_account(conn: sqlite3.Connection, user_id: int) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM accounts WHERE user_id=? LIMIT 1;",
        (int(user_id),),
    ).fetchone()
    if row is None:
        raise NotFoundError("Счёт пользователя не найден.")
    return row


def balance(conn: sqlite3.Connection, session_id: str, ttl_sec: int) -> dict[str, Any]:
    s = _get_session(conn, session_id)
    if s["state"] != SessionState.ACTIVE.value:
        raise StateError(f"Операция недоступна в состоянии {s['state']}.")

    with conn:
        _touch_or_timeout(conn, s, ttl_sec)
        _log_event(conn, session_id, "BALANCE_REQUEST", {})

    acc = _get_account(conn, int(s["authenticated_user_id"]))
    return {"balance_cents": int(acc["balance_cents"]), "currency": acc["currency"]}


def _reset_daily_if_needed(conn: sqlite3.Connection, account_row: sqlite3.Row) -> None:
    today = _utc_now().date().isoformat()
    if (account_row["daily_withdrawn_date"] or "") != today:
        conn.execute(
            "UPDATE accounts SET daily_withdrawn_cents=0, daily_withdrawn_date=? WHERE account_id=?;",
            (today, int(account_row["account_id"])),
        )


def _last_txn_time(conn: sqlite3.Connection, user_id: int) -> datetime | None:
    row = conn.execute(
        "SELECT created_at FROM transactions WHERE user_id=? ORDER BY created_at DESC LIMIT 1;",
        (int(user_id),),
    ).fetchone()
    if row is None:
        return None
    try:
        return datetime.fromisoformat(row["created_at"])
    except Exception:
        return None


def withdraw(conn: sqlite3.Connection, session_id: str, amount_cents: int, ttl_sec: int, risk: RiskScorer | None) -> dict[str, Any]:
    if amount_cents <= 0:
        raise ValidationError("Сумма должна быть > 0 (в копейках/центах).")

    s = _get_session(conn, session_id)
    if s["state"] != SessionState.ACTIVE.value:
        raise StateError(f"Операция недоступна в состоянии {s['state']}.")

    user_id = int(s["authenticated_user_id"])
    acc = _get_account(conn, user_id)

    with conn:
        _touch_or_timeout(conn, s, ttl_sec)
        _reset_daily_if_needed(conn, acc)

        # перечитать после reset
        acc = _get_account(conn, user_id)

        balance_before = int(acc["balance_cents"])
        daily_withdrawn = int(acc["daily_withdrawn_cents"])
        daily_limit = int(acc["daily_withdraw_limit_cents"])

        status = "APPROVED"
        reason = "OK"
        if amount_cents > balance_before:
            status, reason = "DECLINED", "Недостаточно средств."
        elif daily_withdrawn + amount_cents > daily_limit:
            status, reason = "DECLINED", "Превышен дневной лимит снятия."

        last_time = _last_txn_time(conn, user_id)
        time_since_prev = int((_utc_now() - last_time).total_seconds()) if last_time else None

        feature_dict = build_features(
            txn_type="WITHDRAW",
            amount_cents=amount_cents,
            hour=_utc_now().hour,
            dow=_utc_now().weekday(),
            balance_before_cents=balance_before,
            daily_withdrawn_cents=daily_withdrawn,
            daily_withdraw_limit_cents=daily_limit,
            time_since_prev_sec=time_since_prev,
        )

        score = {"risk": 0.0, "flagged": False, "model": "none"}
        if risk is not None:
            try:
                score = risk.score(feature_dict)
            except Exception as e:
                logger.warning("ML score failed: %s", e)
                score = {"risk": 0.0, "flagged": False, "model": "ml_error"}

        balance_after = balance_before
        if status == "APPROVED":
            balance_after = balance_before - amount_cents
            conn.execute(
                "UPDATE accounts SET balance_cents=?, daily_withdrawn_cents=daily_withdrawn_cents+? WHERE account_id=?;",
                (balance_after, amount_cents, int(acc["account_id"])),
            )

        conn.execute(
            """
            INSERT INTO transactions(session_id, user_id, account_id, txn_type, amount_cents, status,
                                     balance_before_cents, balance_after_cents, counterparty_account_id,
                                     created_at, ml_model, ml_risk, ml_flagged, features_json)
            VALUES(?, ?, ?, 'WITHDRAW', ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?);
            """,
            (session_id, user_id, int(acc["account_id"]), int(amount_cents), status,
             balance_before, balance_after, _iso(_utc_now()),
             score.get("model"), float(score.get("risk", 0.0)), 1 if score.get("flagged") else 0,
             json.dumps(feature_dict, ensure_ascii=False)),
        )

        _log_event(conn, session_id, "WITHDRAW", {"amount_cents": amount_cents, "status": status})

    msg = "Снятие выполнено." if status == "APPROVED" else f"Снятие отклонено: {reason}"
    if score.get("flagged"):
        msg += " (Учебно) Операция помечена как подозрительная."

    return {"status": status, "message": msg, "balance_cents": balance_after, "ml": score}


def deposit(conn: sqlite3.Connection, session_id: str, amount_cents: int, ttl_sec: int, risk: RiskScorer | None) -> dict[str, Any]:
    if amount_cents <= 0:
        raise ValidationError("Сумма должна быть > 0 (в копейках/центах).")

    s = _get_session(conn, session_id)
    if s["state"] != SessionState.ACTIVE.value:
        raise StateError(f"Операция недоступна в состоянии {s['state']}.")

    user_id = int(s["authenticated_user_id"])
    acc = _get_account(conn, user_id)

    with conn:
        _touch_or_timeout(conn, s, ttl_sec)

        balance_before = int(acc["balance_cents"])
        last_time = _last_txn_time(conn, user_id)
        time_since_prev = int((_utc_now() - last_time).total_seconds()) if last_time else None

        feature_dict = build_features(
            txn_type="DEPOSIT",
            amount_cents=amount_cents,
            hour=_utc_now().hour,
            dow=_utc_now().weekday(),
            balance_before_cents=balance_before,
            daily_withdrawn_cents=int(acc["daily_withdrawn_cents"]),
            daily_withdraw_limit_cents=int(acc["daily_withdraw_limit_cents"]),
            time_since_prev_sec=time_since_prev,
        )

        score = {"risk": 0.0, "flagged": False, "model": "none"}
        if risk is not None:
            try:
                score = risk.score(feature_dict)
            except Exception as e:
                logger.warning("ML score failed: %s", e)
                score = {"risk": 0.0, "flagged": False, "model": "ml_error"}

        balance_after = balance_before + amount_cents
        conn.execute("UPDATE accounts SET balance_cents=? WHERE account_id=?;", (balance_after, int(acc["account_id"])))

        conn.execute(
            """
            INSERT INTO transactions(session_id, user_id, account_id, txn_type, amount_cents, status,
                                     balance_before_cents, balance_after_cents, counterparty_account_id,
                                     created_at, ml_model, ml_risk, ml_flagged, features_json)
            VALUES(?, ?, ?, 'DEPOSIT', ?, 'APPROVED', ?, ?, NULL, ?, ?, ?, ?, ?);
            """,
            (session_id, user_id, int(acc["account_id"]), int(amount_cents),
             balance_before, balance_after, _iso(_utc_now()),
             score.get("model"), float(score.get("risk", 0.0)), 1 if score.get("flagged") else 0,
             json.dumps(feature_dict, ensure_ascii=False)),
        )

        _log_event(conn, session_id, "DEPOSIT", {"amount_cents": amount_cents, "status": "APPROVED"})

    msg = "Пополнение выполнено."
    if score.get("flagged"):
        msg += " (Учебно) Операция помечена как подозрительная."
    return {"status": "APPROVED", "message": msg, "balance_cents": balance_after, "ml": score}


def transfer(conn: sqlite3.Connection, session_id: str, target_card_token: str, amount_cents: int, ttl_sec: int, risk: RiskScorer | None) -> dict[str, Any]:
    if amount_cents <= 0:
        raise ValidationError("Сумма должна быть > 0 (в копейках/центах).")

    s = _get_session(conn, session_id)
    if s["state"] != SessionState.ACTIVE.value:
        raise StateError(f"Операция недоступна в состоянии {s['state']}.")

    src_user_id = int(s["authenticated_user_id"])
    src_acc = _get_account(conn, src_user_id)

    dst = conn.execute(
        """
        SELECT c.user_id, c.is_active, a.account_id, a.balance_cents
        FROM cards c
        JOIN accounts a ON a.user_id = c.user_id
        WHERE c.card_token = ? LIMIT 1;
        """,
        (target_card_token,),
    ).fetchone()
    if dst is None:
        raise NotFoundError("Карта получателя не найдена.")
    if int(dst["is_active"]) == 0:
        raise ValidationError("Карта получателя не активна (учебно).")
    if int(dst["user_id"]) == src_user_id:
        raise ValidationError("Нельзя переводить самому себе (для простоты прототипа).")

    with conn:
        _touch_or_timeout(conn, s, ttl_sec)

        src_balance_before = int(src_acc["balance_cents"])
        if amount_cents > src_balance_before:
            status, reason = "DECLINED", "Недостаточно средств."
            src_balance_after = src_balance_before
        else:
            status, reason = "APPROVED", "OK"
            src_balance_after = src_balance_before - amount_cents

        last_time = _last_txn_time(conn, src_user_id)
        time_since_prev = int((_utc_now() - last_time).total_seconds()) if last_time else None

        feature_dict = build_features(
            txn_type="TRANSFER_OUT",
            amount_cents=amount_cents,
            hour=_utc_now().hour,
            dow=_utc_now().weekday(),
            balance_before_cents=src_balance_before,
            daily_withdrawn_cents=int(src_acc["daily_withdrawn_cents"]),
            daily_withdraw_limit_cents=int(src_acc["daily_withdraw_limit_cents"]),
            time_since_prev_sec=time_since_prev,
        )

        score = {"risk": 0.0, "flagged": False, "model": "none"}
        if risk is not None:
            try:
                score = risk.score(feature_dict)
            except Exception as e:
                logger.warning("ML score failed: %s", e)
                score = {"risk": 0.0, "flagged": False, "model": "ml_error"}

        if status == "APPROVED":
            # списание у источника
            conn.execute(
                "UPDATE accounts SET balance_cents=? WHERE account_id=?;",
                (src_balance_after, int(src_acc["account_id"])),
            )
            # зачисление получателю
            dst_balance_before = int(dst["balance_cents"])
            dst_balance_after = dst_balance_before + amount_cents
            conn.execute(
                "UPDATE accounts SET balance_cents=? WHERE account_id=?;",
                (dst_balance_after, int(dst["account_id"])),
            )

            # запись входящей транзакции получателя (упрощённо)
            conn.execute(
                """
                INSERT INTO transactions(session_id, user_id, account_id, txn_type, amount_cents, status,
                                         balance_before_cents, balance_after_cents, counterparty_account_id,
                                         created_at, ml_model, ml_risk, ml_flagged, features_json)
                VALUES(?, ?, ?, 'TRANSFER_IN', ?, 'APPROVED', ?, ?, ?, ?, NULL, NULL, 0, '{}');
                """,
                (session_id, int(dst["user_id"]), int(dst["account_id"]), int(amount_cents),
                 dst_balance_before, dst_balance_after, int(src_acc["account_id"]), _iso(_utc_now())),
            )

        # исходящая транзакция источника пишется всегда (даже при отказе) — удобно для аналитики
        conn.execute(
            """
            INSERT INTO transactions(session_id, user_id, account_id, txn_type, amount_cents, status,
                                     balance_before_cents, balance_after_cents, counterparty_account_id,
                                     created_at, ml_model, ml_risk, ml_flagged, features_json)
            VALUES(?, ?, ?, 'TRANSFER_OUT', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (session_id, src_user_id, int(src_acc["account_id"]), int(amount_cents), status,
             src_balance_before, src_balance_after, int(dst["account_id"]), _iso(_utc_now()),
             score.get("model"), float(score.get("risk", 0.0)), 1 if score.get("flagged") else 0,
             json.dumps(feature_dict, ensure_ascii=False)),
        )

        _log_event(conn, session_id, "TRANSFER", {"amount_cents": amount_cents, "target_card_token": target_card_token, "status": status})

    msg = "Перевод выполнен." if status == "APPROVED" else f"Перевод отклонён: {reason}"
    if score.get("flagged"):
        msg += " (Учебно) Операция помечена как подозрительная."
    return {"status": status, "message": msg, "balance_cents": src_balance_after, "ml": score}


def cancel_session(conn: sqlite3.Connection, session_id: str) -> dict[str, Any]:
    s = _get_session(conn, session_id)
    if s["state"] in (SessionState.CANCELLED.value, SessionState.CLOSED.value, SessionState.TIMED_OUT.value):
        return {"state": s["state"], "message": "Сессия уже завершена."}

    with conn:
        conn.execute(
            "UPDATE sessions SET state=?, closed_at=? WHERE session_id=?;",
            (SessionState.CANCELLED.value, _iso(_utc_now()), session_id),
        )
        _log_event(conn, session_id, "SESSION_CANCELLED", {})
    return {"state": SessionState.CANCELLED.value, "message": "Операция отменена. Карта извлечена (учебно)."}
