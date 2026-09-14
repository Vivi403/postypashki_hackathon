"""
Задача 7. Attribution != Incrementality.

Отвечает не на вопрос "кому приписать продажу" (это attribution.py),
а на вопрос "правда ли реклама добавила продажи, а не просто совпала
по времени с их естественным ростом".
"""
import hashlib

import numpy as np
import pandas as pd
from scipy import stats


def load_daily_orders(path: str) -> pd.DataFrame:
    """Дневной ряд заказов с признаками курса (ПРО / СТАРТ)."""
    df = pd.read_excel(path)
    df.columns = ["student_id", "amount", "course", "ts"]
    orders = df.groupby(["student_id", "ts"], as_index=False).agg(
        courses=("course", lambda x: list(x))
    )
    orders["date"] = pd.to_datetime(orders["ts"].dt.date)
    orders["is_pro"] = orders["courses"].apply(lambda cs: any("про" in c.lower() for c in cs))
    orders["is_start"] = orders["courses"].apply(lambda cs: any("старт" in c.lower() for c in cs))
    return orders


def diff_in_diff(orders: pd.DataFrame, pre_start: str, pre_end: str,
                  treat_start: str, treat_end: str) -> dict:
    """Сравнивает рост продаж у продвигаемой линейки (ПРО) с ростом у
    непродвигаемой (СТАРТ) в то же самое время. Разница разностей —
    это эффект, очищенный от общего тренда/сезонности периода."""
    full_range = pd.date_range(orders["date"].min(), orders["date"].max(), freq="D")
    daily_pro = orders[orders["is_pro"]].groupby("date").size().reindex(full_range, fill_value=0)
    daily_start = orders[orders["is_start"]].groupby("date").size().reindex(full_range, fill_value=0)

    pre = pd.date_range(pre_start, pre_end)
    treat = pd.date_range(treat_start, treat_end)

    pro_pre, pro_treat = daily_pro[pre].mean(), daily_pro[treat].mean()
    start_pre, start_treat = daily_start[pre].mean(), daily_start[treat].mean()

    return {
        "pro_pre": pro_pre, "pro_treat": pro_treat, "pro_delta": pro_treat - pro_pre,
        "start_pre": start_pre, "start_treat": start_treat, "start_delta": start_treat - start_pre,
        "effect_per_day": (pro_treat - pro_pre) - (start_treat - start_pre),
    }


def weekend_confound_range(orders: pd.DataFrame, spike_date: str) -> dict:
    """Для дня, где акция и выходной совпали, честно раскладывает
    всплеск на вилку, а не на одно число — потому что разделить эффект
    скидки и эффект выходного дня на этих данных нельзя."""
    full_range = pd.date_range(orders["date"].min(), orders["date"].max(), freq="D")
    daily = orders.groupby("date").size().reindex(full_range, fill_value=0)
    daily.index.name = "date"
    d = daily.reset_index(name="orders")
    d["weekday"] = d["date"].dt.day_name()
    spike = int(d.loc[d["date"] == spike_date, "orders"].iloc[0])

    weekday_baseline = d.loc[~d["weekday"].isin(["Saturday", "Sunday"]), "orders"].mean()
    weekend_baseline = d.loc[
        d["weekday"].isin(["Saturday", "Sunday"]) & (d["date"] != spike_date), "orders"
    ].mean()

    return {
        "spike_orders": spike,
        "lower_bound": spike - weekend_baseline,  # если бы день вёл себя как обычный выходной
        "upper_bound": spike - weekday_baseline,  # если бы день вёл себя как обычный будний
    }


def required_sample_size(baseline_rate: float, relative_lift: float,
                          alpha: float = 0.05, power: float = 0.8) -> int:
    """Сколько человек нужно на группу в holdout-тесте, чтобы поймать
    заданный относительный прирост конверсии с заданной надёжностью."""
    p1 = baseline_rate
    p2 = baseline_rate * (1 + relative_lift)
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)
    pooled_var = p1 * (1 - p1) + p2 * (1 - p2)
    n = (z_alpha + z_power) ** 2 * pooled_var / (p2 - p1) ** 2
    return int(np.ceil(n))


def assign_group(user_id: str, experiment_name: str) -> str:
    """Детерминированное назначение в группу теста/контроля: один и тот
    же пользователь всегда попадает в одну и ту же группу, без
    хранения отдельной таблицы назначений на каждый пересчёт."""
    h = hashlib.sha256(f"{experiment_name}:{user_id}".encode()).hexdigest()
    return "treatment" if int(h[:8], 16) % 2 == 0 else "control"


def power_simulation(n_per_group: int, baseline_rate: float, relative_lift: float,
                      n_sims: int = 5000, alpha: float = 0.05, seed: int = 42) -> float:
    """Честная проверка метода: как часто holdout-тест ЗАМЕТИТ реальный
    эффект при данном размере выборки. Показывает, где тесту доверять
    можно, а где отсутствие результата ничего не доказывает."""
    rng = np.random.default_rng(seed)
    p1, p2 = baseline_rate, baseline_rate * (1 + relative_lift)
    detected = 0
    z_crit = stats.norm.ppf(1 - alpha / 2)
    for _ in range(n_sims):
        a = rng.binomial(n_per_group, p1)
        b = rng.binomial(n_per_group, p2)
        p_pool = (a + b) / (2 * n_per_group)
        se = np.sqrt(p_pool * (1 - p_pool) * (2 / n_per_group)) if 0 < p_pool < 1 else 1e-9
        z = (b / n_per_group - a / n_per_group) / se
        if abs(z) > z_crit:
            detected += 1
    return detected / n_sims


if __name__ == "__main__":
    orders = load_daily_orders("/mnt/user-data/uploads/base.xlsx")

    print("=== DiD: продвижение ПРО, 20-23 августа, контроль — СТАРТ ===")
    did = diff_in_diff(orders, "2026-08-15", "2026-08-19", "2026-08-20", "2026-08-23")
    for k, v in did.items():
        print(f"  {k}: {v:.2f}")

    print("\n=== Вилка для всплеска 9 августа (скидка + выходной) ===")
    rng_result = weekend_confound_range(orders, "2026-08-09")
    for k, v in rng_result.items():
        print(f"  {k}: {v:.1f}")

    print("\n=== Размер выборки для holdout (baseline 5%, лифт +50%) ===")
    n = required_sample_size(0.05, 0.5)
    print(f"  нужно на группу: {n}")

    print("\n=== Проверка мощности метода (истинный эффект +30%) ===")
    for n_per_group in (600, 40_000):
        p = power_simulation(n_per_group, 0.05, 0.30)
        print(f"  n={n_per_group}: эффект обнаружен в {p:.1%} симуляций")
