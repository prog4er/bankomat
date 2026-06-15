from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np


def build_features(
    txn_type: str,
    amount_cents: int,
    hour: int,
    dow: int,
    balance_before_cents: int,
    daily_withdrawn_cents: int,
    daily_withdraw_limit_cents: int,
    time_since_prev_sec: int | None,
) -> dict[str, Any]:
    """
    Единый генератор признаков для train/infer.
    Важно: признаки учебные, не содержат персональных данных и не используют реальные платежные реквизиты.
    """
    limit = max(1, int(daily_withdraw_limit_cents))
    ratio = float(daily_withdrawn_cents) / limit

    return {
        "txn_type": txn_type,  # категориальный
        "amount_log": math.log1p(max(0, int(amount_cents))) / 10.0,
        "hour": int(hour),
        "dow": int(dow),
        "is_night": 1 if int(hour) in (0, 1, 2, 3, 4, 5) else 0,
        "balance_log": math.log1p(max(0, int(balance_before_cents))) / 10.0,
        "daily_withdraw_ratio": ratio,
        "time_since_prev_log": math.log1p(time_since_prev_sec) / 10.0 if time_since_prev_sec is not None else 0.0,
    }


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + math.exp(-x)))


class RiskScoringService:
    """
    Загружает модели из artifacts_dir:
      - model_rf.joblib
      - model_logreg.joblib
      - model_isof.joblib
    Если моделей нет — всегда риск=0.
    """
    def __init__(self, artifacts_dir: str = "artifacts", primary: str = "rf", threshold: float = 0.80):
        self.dir = Path(artifacts_dir)
        self.primary = primary
        self.threshold = threshold

        self.models: dict[str, Any] = {}
        self._try_load("rf", "model_rf.joblib")
        self._try_load("logreg", "model_logreg.joblib")
        self._try_load("isof", "model_isof.joblib")

    def _try_load(self, key: str, filename: str) -> None:
        path = self.dir / filename
        if path.exists():
            self.models[key] = joblib.load(path)

    def score(self, features: dict[str, Any]) -> dict[str, Any]:
        # порядок предпочтений
        order = [self.primary, "rf", "logreg", "isof"]
        for key in order:
            if key == "isof":
                k = "isof"
            else:
                k = key
            if k in self.models:
                return self._score_with(k, features)

        return {"risk": 0.0, "flagged": False, "model": "none", "details": {"reason": "no_models"}}

    def _score_with(self, key: str, features: dict[str, Any]) -> dict[str, Any]:
        model = self.models[key]

        X = [features]  # scikit-learn pipeline (DictVectorizer внутри) ожидает list[dict]
        if key in ("rf", "logreg"):
            proba = float(model.predict_proba(X)[0, 1])
            flagged = proba >= self.threshold
            return {"risk": proba, "flagged": flagged, "model": key, "details": {}}

        if key == "isof":
            # decision_function: чем выше, тем "нормальнее" (см. смысл decision function у IsolationForest) [13]
            df = float(model.decision_function(X)[0])
            # преобразуем в "риск": чем меньше df, тем больше риск
            risk = _sigmoid(-5.0 * df)
            flagged = risk >= self.threshold
            return {"risk": risk, "flagged": flagged, "model": "isof", "details": {"decision_function": df}}

        return {"risk": 0.0, "flagged": False, "model": "unknown", "details": {}}
