from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml.infer import build_features


def generate_synthetic_dataset(
    n_users: int,
    txns_per_user: int,
    seed: int,
    suspicious_rate: float = 0.02,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """
    Генератор синтетических пользователей/транзакций.
    Возвращает:
      X: list[feature_dict]
      y: np.ndarray labels (0/1)

    Идея:
      - каждому пользователю задаём "типичный" объём/время операций
      - нормальные операции вокруг типичных параметров
      - подозрительные: большие суммы, ночь, высокая частота и т.п.
    """
    rng = np.random.default_rng(seed)

    X: list[dict[str, Any]] = []
    y: list[int] = []

    for _u in range(n_users):
        typical_amount = float(rng.lognormal(mean=8.5, sigma=0.6))  # в "центах"
        typical_hour = int(rng.integers(8, 20))
        balance = int(rng.integers(30_000, 200_000))
        daily_limit = int(rng.integers(20_000, 60_000))
        daily_withdrawn = 0

        prev_gap = int(rng.integers(60, 3600))

        for _t in range(txns_per_user):
            txn_type = rng.choice(["WITHDRAW", "DEPOSIT", "TRANSFER_OUT"], p=[0.45, 0.20, 0.35]).item()
            is_susp = 1 if rng.random() < suspicious_rate else 0

            # базовые параметры
            hour = int(np.clip(rng.normal(typical_hour, 3.0), 0, 23))
            dow = int(rng.integers(0, 7))
            gap = int(rng.integers(10, 7200))
            prev_gap = gap

            amount = float(rng.lognormal(mean=np.log(max(100.0, typical_amount)), sigma=0.7))
            amount_cents = int(np.clip(amount, 100, 200_000))

            # инъекция "подозрительности"
            if is_susp:
                hour = int(rng.choice([0, 1, 2, 3, 4, 5, 23]))
                amount_cents = int(np.clip(amount_cents * rng.uniform(2.5, 8.0), 100, 400_000))
                gap = int(rng.integers(1, 30))

            # имитация динамики баланса (очень упрощённо)
            balance_before = balance
            if txn_type in ("WITHDRAW", "TRANSFER_OUT"):
                if amount_cents > balance:
                    # если средств не хватает — делаем "сжатие" суммы, но помечаем как подозрительную иногда
                    amount_cents = max(100, int(balance * rng.uniform(0.2, 0.9)))
                    if rng.random() < 0.3:
                        is_susp = 1
                balance -= amount_cents
                daily_withdrawn += amount_cents
            else:
                balance += amount_cents

            f = build_features(
                txn_type=txn_type,
                amount_cents=amount_cents,
                hour=hour,
                dow=dow,
                balance_before_cents=balance_before,
                daily_withdrawn_cents=daily_withdrawn,
                daily_withdraw_limit_cents=daily_limit,
                time_since_prev_sec=gap,
            )
            X.append(f)
            y.append(is_susp)

    return X, np.array(y, dtype=int)


def eval_binary(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    roc = roc_auc_score(y_true, y_score)  # ROC AUC [14]

    prec, rec, _ = precision_recall_curve(y_true, y_score)  # PR curve [16]
    pr_auc = auc(rec, prec)

    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "roc_auc": float(roc),
        "pr_auc": float(pr_auc),
    }


def plot_curves(y_true: np.ndarray, y_score: np.ndarray, out_dir: Path, prefix: str) -> None:
    # ROC curve [15]
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC: {prefix}")
    plt.savefig(out_dir / f"{prefix}_roc.png", dpi=160, bbox_inches="tight")
    plt.close()

    # PR curve [16]
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR: {prefix}")
    plt.savefig(out_dir / f"{prefix}_pr.png", dpi=160, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="artifacts")
    ap.add_argument("--n-users", type=int, default=300)
    ap.add_argument("--txns-per-user", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=0.80, help="Порог для метрик (для примера)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y = generate_synthetic_dataset(args.n_users, args.txns_per_user, args.seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=args.seed, stratify=y)

    results: dict[str, Any] = {}

    # 1) LogisticRegression [11]
    logreg = Pipeline(
        steps=[
            ("vec", DictVectorizer(sparse=True)),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")),
        ]
    )
    logreg.fit(X_train, y_train)
    y_score_lr = logreg.predict_proba(X_test)[:, 1]
    results["logreg"] = eval_binary(y_test, y_score_lr, threshold=args.threshold)
    plot_curves(y_test, y_score_lr, out_dir, "logreg")
    joblib.dump(logreg, out_dir / "model_logreg.joblib")

    # 2) RandomForestClassifier [12]
    rf = Pipeline(
        steps=[
            ("vec", DictVectorizer(sparse=True)),
            ("model", RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=args.seed,
            )),
        ]
    )
    rf.fit(X_train, y_train)
    y_score_rf = rf.predict_proba(X_test)[:, 1]
    results["rf"] = eval_binary(y_test, y_score_rf, threshold=args.threshold)
    plot_curves(y_test, y_score_rf, out_dir, "rf")
    joblib.dump(rf, out_dir / "model_rf.joblib")

    # 3) IsolationForest (обучаем на "норме", оценка как аномалия) [13]
    contamination = float(max(0.001, min(0.20, y_train.mean() if y_train.mean() > 0 else 0.02)))
    isof = Pipeline(
        steps=[
            ("vec", DictVectorizer(sparse=True)),
            ("model", IsolationForest(
                n_estimators=200,
                contamination=contamination,
                random_state=args.seed,
            )),
        ]
    )
    X_train_norm = [x for x, yy in zip(X_train, y_train) if yy == 0]
    isof.fit(X_train_norm)

    # decision_function: больше => более "нормально"; риск ~ -decision_function [13]
    df = isof.decision_function(X_test)
    y_score_isof = 1.0 / (1.0 + np.exp(5.0 * df))  # сигмоид от (-df)
    results["isof"] = eval_binary(y_test, y_score_isof, threshold=args.threshold)
    plot_curves(y_test, y_score_isof, out_dir, "isof")
    joblib.dump(isof, out_dir / "model_isof.joblib")

    # Сводка
    (out_dir / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved models and metrics to:", out_dir.resolve())
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
