"""
src/attribution.py — пять моделей атрибуции поверх touches + payments.

Три инварианта, которые проверяются, а не молчаливо предполагаются
(задача 6):
1. Веса на одну покупку всегда сходятся в единицу.
2. Заказы без единого касания не выбрасываются, а идут в канал "organic".
3. "organic" — явный канал, а не свалка для ошибок разметки.
"""
import sqlite3
from collections import defaultdict

MODELS = ["first_touch", "last_touch", "linear", "time_decay", "position_based"]


def _weights_first_touch(touches: list[dict]) -> dict[str, float]:
    return {touches[0]["channel_name"]: 1.0}


def _weights_last_touch(touches: list[dict]) -> dict[str, float]:
    return {touches[-1]["channel_name"]: 1.0}


def _weights_linear(touches: list[dict]) -> dict[str, float]:
    w = defaultdict(float)
    share = 1.0 / len(touches)
    for t in touches:
        w[t["channel_name"]] += share
    return dict(w)


def _weights_time_decay(touches: list[dict], half_life_days: float = 3.0) -> dict[str, float]:
    w = defaultdict(float)
    order_ts = touches[-1]["_order_ts"]
    raw = []
    for t in touches:
        age_days = (order_ts - t["ts"]).total_seconds() / 86400
        raw.append(0.5 ** (age_days / half_life_days))
    total = sum(raw)
    for t, r in zip(touches, raw):
        w[t["channel_name"]] += r / total
    return dict(w)


def _weights_position_based(touches: list[dict]) -> dict[str, float]:
    w = defaultdict(float)
    if len(touches) == 1:
        return {touches[0]["channel_name"]: 1.0}
    w[touches[0]["channel_name"]] += 0.4
    w[touches[-1]["channel_name"]] += 0.4
    middle = touches[1:-1]
    if middle:
        share = 0.2 / len(middle)
        for t in middle:
            w[t["channel_name"]] += share
    else:
        # только два касания — 20% "середины" уходит поровну первому/последнему
        w[touches[0]["channel_name"]] += 0.1
        w[touches[-1]["channel_name"]] += 0.1
    return dict(w)


_MODEL_FUNCS = {
    "first_touch": _weights_first_touch,
    "last_touch": _weights_last_touch,
    "linear": _weights_linear,
    "time_decay": _weights_time_decay,
    "position_based": _weights_position_based,
}


def _load_orders_with_touches(conn: sqlite3.Connection):
    """Заказ = одна строка payments с уникальным (student_id, ts) —
    правило склейки из задачи 1, здесь берём уже агрегированную сумму
    на уровне первой строки заказа, чтобы не задваивать пакетные заказы."""
    import pandas as pd

    payments = pd.read_sql(
        "SELECT hashed_user_id, ts, SUM(amount) as amount "
        "FROM payments GROUP BY hashed_user_id, ts", conn,
    )
    payments["ts"] = pd.to_datetime(payments["ts"])

    touches = pd.read_sql(
        "SELECT hashed_user_id, channel_name, ts FROM touches ORDER BY hashed_user_id, ts", conn,
    )
    touches["ts"] = pd.to_datetime(touches["ts"])
    touches_by_user = defaultdict(list)
    for _, row in touches.iterrows():
        touches_by_user[row["hashed_user_id"]].append(
            {"channel_name": row["channel_name"], "ts": row["ts"]}
        )
    return payments, touches_by_user


def run_all_models(db_path: str) -> dict[str, dict[str, float]]:
    """Возвращает {модель: {канал: attributed_revenue}} для всех 5 моделей,
    плюс канал "organic" для заказов без касаний."""
    conn = sqlite3.connect(db_path)
    payments, touches_by_user = _load_orders_with_touches(conn)
    conn.close()

    results = {m: defaultdict(float) for m in MODELS}

    for _, order in payments.iterrows():
        user_touches = [t for t in touches_by_user.get(order["hashed_user_id"], [])
                         if t["ts"] <= order["ts"]]
        if not user_touches:
            for m in MODELS:
                results[m]["organic"] += order["amount"]
            continue

        user_touches = sorted(user_touches, key=lambda t: t["ts"])
        for t in user_touches:
            t["_order_ts"] = order["ts"]

        for model in MODELS:
            weights = _MODEL_FUNCS[model](user_touches)
            assert abs(sum(weights.values()) - 1.0) < 1e-9, \
                f"веса не сходятся в 1.0 для модели {model}: {weights}"
            for channel, w in weights.items():
                results[model][channel] += w * order["amount"]

    return {m: dict(channels) for m, channels in results.items()}


if __name__ == "__main__":
    revenue_by_model = run_all_models("db/demo.db")
    for model in MODELS:
        print(f"\n=== {model} ===")
        for channel, revenue in sorted(revenue_by_model[model].items(), key=lambda x: -x[1]):
            print(f"  {channel:15s} {revenue:>12,.0f} ₽".replace(",", " "))
