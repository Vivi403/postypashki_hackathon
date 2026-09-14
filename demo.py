"""
demo.py — сквозной прогон: продажи -> касания -> атрибуция -> ROMI.

Запуск:
    pip install -r requirements.txt
    python demo.py
"""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from bot import simulator
from src.attribution import run_all_models, MODELS
from src.romi import romi_by_channel, format_romi
from src.incrementality import (
    load_daily_orders,
    diff_in_diff,
    weekend_confound_range,
    required_sample_size,
    power_simulation,
)
from src.forecast import load_daily_orders as load_forecast_orders, backtest, mae, mape

DB_PATH = "demo.db"
BASE_XLSX = "data/base.xlsx"
SCHEMA_PATH = "bd/schema_sqlite.sql"


def section(title: str):
    print(f"\n--- {title} ---")


def step1_build_and_simulate():
    section("1. Продажи + касания (bot/simulator.py)")
    simulator.run(db_path=DB_PATH, base_xlsx_path=BASE_XLSX, schema_path=SCHEMA_PATH)


def step2_attribution():
    section("2. Атрибуция по 5 моделям")
    revenue_by_model = run_all_models(DB_PATH)
    for model in MODELS:
        print(f"\n{model}:")
        for channel, revenue in sorted(
            revenue_by_model[model].items(), key=lambda x: -x[1]
        ):
            print(f"  {channel:15s} {revenue:>12,.0f} \u20bd".replace(",", " "))
    return revenue_by_model


def step3_romi(revenue_by_model: dict):
    section("3. ROMI и CAC (position_based)")
    conn = sqlite3.connect(DB_PATH)
    cost_rows = conn.execute(
        "SELECT channel_name, SUM(cost) as total_cost, MAX(cost_is_estimate) as is_estimate "
        "FROM placements GROUP BY channel_name"
    ).fetchall()
    conn.close()
    cost = {ch: c for ch, c, _ in cost_rows}
    estimated = {ch for ch, _, is_est in cost_rows if is_est}

    revenue = dict(revenue_by_model["position_based"])
    revenue.setdefault("organic", 0.0)
    cost.setdefault("organic", 0.0)

    orders_placeholder = {ch: 0 for ch in revenue}
    results = romi_by_channel(
        revenue, cost, orders_placeholder, margin=0.7, estimated_channels=estimated
    )
    for r in results:
        tag = (
            " (оценка стоимости)"
            if r.cost_is_estimate
            else (" (затрат нет)" if r.cost == 0 else "")
        )
        print(
            f"  {r.channel:15s} revenue={r.revenue:>12,.0f} \u20bd  "
            f"cost={r.cost:>8,.0f} \u20bd  ROMI={format_romi(r.romi)}{tag}".replace(
                ",", " "
            )
        )
    print("\nЗатраты — сумма 4 условных размещений за период, не один пост.")
    print("Это по-прежнему рыночная оценка, а не реальные счета.")


def step4_real_data_blocks():
    section("4. Incrementality и прогноз (реальные данные, без синтетики)")

    print("\nDiD: продвижение ПРО 20-23.08 vs контроль СТАРТ")
    orders = load_daily_orders(BASE_XLSX)
    did = diff_in_diff(orders, "2026-08-15", "2026-08-19", "2026-08-20", "2026-08-23")
    print(
        f"  ПРО:   {did['pro_pre']:.2f} -> {did['pro_treat']:.2f} заказов/день (+{did['pro_delta']:.2f})"
    )
    print(
        f"  СТАРТ: {did['start_pre']:.2f} -> {did['start_treat']:.2f} (+{did['start_delta']:.2f})"
    )
    print(f"  Эффект рекламы: {did['effect_per_day']:+.2f} заказов/день")

    rng = weekend_confound_range(orders, "2026-08-09")
    print(
        f"\nВсплеск 9 августа: {rng['spike_orders']:.0f} заказов, "
        f"скидка и выходной неразличимы — вилка от {rng['lower_bound']:.0f} до {rng['upper_bound']:.0f}"
    )

    n = required_sample_size(0.05, 0.5)
    print(f"\nHoldout на будущее: {n} человек на группу (baseline 5%, лифт +50%)")

    print("\nПрогноз: backtest бейзлайнов (walk-forward)")
    daily = load_forecast_orders(BASE_XLSX)
    y = daily["n_orders"].values
    bt = backtest(y, daily["date"])
    for m in ["naive", "seasonal7", "ma7", "wd_profile"]:
        print(
            f"  {m:12s} MAE={mae(bt['actual'].values, bt[m].values):5.2f}  "
            f"MAPE={mape(bt['actual'].values, bt[m].values):6.1f}%"
        )


def final_summary():
    section("Реальное vs синтетика")
    print("""реальное: продажи (base.xlsx), incrementality, backtest прогноза
синтетика: каналы, размещения, сами touches и их стоимость

Когда появится реальный трекинг, шаг 1 заменяется на настоящие данные
бота — шаги 2-4 не меняются.""")


if __name__ == "__main__":
    step1_build_and_simulate()
    revenue_by_model = step2_attribution()
    step3_romi(revenue_by_model)
    step4_real_data_blocks()
    final_summary()
