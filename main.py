from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from core.logic import (
    ATMError,
    AuthError,
    NotFoundError,
    StateError,
    ValidationError,
    balance,
    cancel_session,
    init_db,
    seed_demo_data,
    start_session,
    get_session_info,
    withdraw,
    deposit,
    transfer,
)
from ml.infer import RiskScoringService

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("atm")

DEFAULT_DB_PATH = os.getenv("ATM_DB_PATH", str(Path("data") / "atm.db"))
DEFAULT_TTL_SEC = int(os.getenv("ATM_SESSION_TTL_SEC", "60"))
DEFAULT_SCHEMA_PATH = os.getenv("ATM_SCHEMA_PATH", str(Path("storage") / "schema.sql"))
DEFAULT_ARTIFACTS_DIR = os.getenv("ATM_ARTIFACTS_DIR", str(Path("artifacts")))


class StartSessionRequest(BaseModel):
    card_token: str = Field(min_length=1, max_length=64)
    pin: str = Field(min_length=4, max_length=12)


class AmountRequest(BaseModel):
    amount_cents: int = Field(gt=0, description="Сумма в копейках/центах (целое число)")


class TransferRequest(BaseModel):
    target_card_token: str = Field(min_length=1, max_length=64)
    amount_cents: int = Field(gt=0)


def create_app(
    db_path: str = DEFAULT_DB_PATH,
    schema_path: str = DEFAULT_SCHEMA_PATH,
    artifacts_dir: str = DEFAULT_ARTIFACTS_DIR,
    ttl_sec: int = DEFAULT_TTL_SEC,
) -> FastAPI:
    app = FastAPI(title="ATM Simulator (Educational)")

    app.state.db_path = db_path
    app.state.schema_path = schema_path
    app.state.ttl_sec = ttl_sec
    app.state.artifacts_dir = artifacts_dir
    app.state.risk = RiskScoringService(artifacts_dir=artifacts_dir)

    def get_conn() -> sqlite3.Connection:
        Path(app.state.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(app.state.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def conn_dep():
        conn = get_conn()
        try:
            yield conn
        finally:
            conn.close()

    @app.on_event("startup")
    def _startup():
        Path(app.state.artifacts_dir).mkdir(parents=True, exist_ok=True)
        schema_sql = Path(app.state.schema_path).read_text(encoding="utf-8")
        conn = get_conn()
        try:
            init_db(conn, schema_sql)
            seed_demo_data(conn)
            logger.info("DB ready at %s, artifacts at %s", app.state.db_path, app.state.artifacts_dir)
        finally:
            conn.close()

    def _map_error(e: Exception):
        if isinstance(e, NotFoundError):
            raise HTTPException(status_code=404, detail=str(e))
        if isinstance(e, AuthError):
            raise HTTPException(status_code=401, detail=str(e))
        if isinstance(e, ValidationError):
            raise HTTPException(status_code=400, detail=str(e))
        if isinstance(e, StateError):
            # можно использовать 409 Conflict как "invalid state"
            raise HTTPException(status_code=409, detail=str(e))
        if isinstance(e, ATMError):
            raise HTTPException(status_code=400, detail=str(e))
        raise

    @app.post("/api/v1/auth/start")
    def api_start(req: StartSessionRequest, conn=Depends(conn_dep)):
        try:
            return start_session(conn, req.card_token, req.pin, app.state.ttl_sec)
        except Exception as e:
            _map_error(e)

    @app.get("/api/v1/session/{session_id}")
    def api_session_info(session_id: str, conn=Depends(conn_dep)):
        try:
            return get_session_info(conn, session_id, app.state.ttl_sec)
        except Exception as e:
            _map_error(e)

    @app.post("/api/v1/session/{session_id}/balance")
    def api_balance(session_id: str, conn=Depends(conn_dep)):
        try:
            return balance(conn, session_id, app.state.ttl_sec)
        except Exception as e:
            _map_error(e)

    @app.post("/api/v1/session/{session_id}/withdraw")
    def api_withdraw(session_id: str, req: AmountRequest, conn=Depends(conn_dep)):
        try:
            return withdraw(conn, session_id, req.amount_cents, app.state.ttl_sec, app.state.risk)
        except Exception as e:
            _map_error(e)

    @app.post("/api/v1/session/{session_id}/deposit")
    def api_deposit(session_id: str, req: AmountRequest, conn=Depends(conn_dep)):
        try:
            return deposit(conn, session_id, req.amount_cents, app.state.ttl_sec, app.state.risk)
        except Exception as e:
            _map_error(e)

    @app.post("/api/v1/session/{session_id}/transfer")
    def api_transfer(session_id: str, req: TransferRequest, conn=Depends(conn_dep)):
        try:
            return transfer(conn, session_id, req.target_card_token, req.amount_cents, app.state.ttl_sec, app.state.risk)
        except Exception as e:
            _map_error(e)

    @app.post("/api/v1/session/{session_id}/cancel")
    def api_cancel(session_id: str, conn=Depends(conn_dep)):
        try:
            return cancel_session(conn, session_id)
        except Exception as e:
            _map_error(e)

    return app


app = create_app()
