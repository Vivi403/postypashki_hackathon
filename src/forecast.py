"""
Задача 9. Прогноз продаж.

Честная валидация важнее сложности модели — при 37 полных днях истории
любая модель сложнее baseline рискует переобучиться на шум.

Запуск: python forecast.py
Вход: base.xlsx (795 строк, 04.08–10.09.2026)
"""

import numpy as np
import pandas as pd

LAST_FULL_DAY = pd.Timestamp(
    "2026-09-09"
)  # 10 сентября обрезан (1 заказ в 06:09) — исключаем


def load_daily_orders(path: str) -> pd.DataFrame:
    """Строит дневной ряд числа заказов. Заказ = строки одного student_id
    с одинаковым timestamp (правило склейки из задачи 1, защищено разрывом
    167×0с / 26 минут между соседними покупками)."""
    df = pd.read_excel(path)
    df.columns = ["student_id", "amount", "course", "ts"]
    orders = df.groupby(["student_id", "ts"], as_index=False).agg(
        amount=("amount", "sum")
    )
    orders["date"] = pd.to_datetime(orders["ts"].dt.date)

    full_range = pd.date_range(orders["date"].min(), orders["date"].max(), freq="D")
    daily = (
        orders.groupby("date")
        .agg(n_orders=("amount", "count"), revenue=("amount", "sum"))
        .reindex(full_range, fill_value=0)
    )
    daily.index.name = "date"
    daily = daily.reset_index()
    return daily[daily["date"] <= LAST_FULL_DAY].reset_index(drop=True)


def backtest(y: np.ndarray, dates: pd.Series, start: int = 14) -> pd.DataFrame:
    """Walk-forward, one-step-ahead backtest трёх baseline-моделей.
    start=14 — минимум истории, чтобы seasonal7 и weekday-profile
    успели накопить хотя бы одно полное наблюдение по каждому дню недели."""
    weekday = dates.dt.dayofweek.values
    rows = []
    for t in range(start, len(y)):
        naive = y[t - 1]
        seasonal7 = y[t - 7] if t >= 7 else np.nan
        ma7 = y[max(0, t - 7) : t].mean()
        mask = weekday[:t] == weekday[t]
        wd_profile = y[:t][mask].mean() if mask.sum() > 0 else y[:t].mean() # type: ignore
        rows.append(
            {
                "date": dates[t],
                "actual": y[t],
                "naive": naive,
                "seasonal7": seasonal7,
                "ma7": ma7,
                "wd_profile": wd_profile,
            }
        )
    return pd.DataFrame(rows)


def mae(a, f):
    return float(np.mean(np.abs(a - f)))


def mape(a, f):
    mask = a > 0
    return float(np.mean(np.abs(a[mask] - f[mask]) / a[mask]) * 100)


def forecast_uncertainty(y: np.ndarray, horizons=(1, 7, 14, 30)) -> pd.DataFrame:
    """Ширина 95% доверительного интервала суммарного прогноза на горизонте h,
    в предположении iid-шума вокруг среднего (нижняя граница неопределённости —
    реальная будет не меньше, т.к. дни не независимы при наличии кампаний)."""
    mean_d, std_d = y.mean(), y.std(ddof=1)
    rows = []
    for h in horizons:
        mean_h = mean_d * h
        ci95 = 1.96 * std_d * np.sqrt(h)
        rows.append(
            {
                "horizon_days": h,
                "forecast_sum": round(mean_h),
                "ci95_width": round(ci95),
                "ci95_pct_of_forecast": round(ci95 / mean_h * 100, 1),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    daily = load_daily_orders("/mnt/user-data/uploads/base.xlsx")
    y = daily["n_orders"].values

    print(
        f"Дней в ряду: {len(daily)} ({daily['date'].min().date()} — {daily['date'].max().date()})"
    )
    print(
        f"mean={y.mean():.2f}  std={y.std(ddof=1):.2f}  CV={y.std(ddof=1)/y.mean():.2f}\n" # type: ignore
    )

    bt = backtest(y, daily["date"]) # type: ignore
    print("Backtest (one-step-ahead, walk-forward):")
    for m in ["naive", "seasonal7", "ma7", "wd_profile"]:
        print(
            f"  {m:12s} MAE={mae(bt['actual'].values, bt[m].values):5.2f}  "
            f"MAPE={mape(bt['actual'].values, bt[m].values):6.1f}%"
        )

    print("\nШирина доверительного интервала по горизонтам:")
    print(forecast_uncertainty(y).to_string(index=False)) # type: ignore
