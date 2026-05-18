#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import duckdb
import numpy as np

from compute import ema_np, get_strategy_profiles, normalize_code, sma_np, rolling_max_np
from settings import MARKET_DB, SMART_DB


OUT_DIR = ROOT / "analysis"
SUMMARY_CSV = OUT_DIR / "strategy_audit_summary.csv"
TRADES_CSV = OUT_DIR / "strategy_recent_trades.csv"
REPORT_MD = OUT_DIR / "strategy_audit_report.md"

HISTORY_LOOKBACK_SIGNALS = 60
RECENT_TRADE_LIMIT = 20
MIN_PRIOR_WIN_RATE = 0.48
MIN_PRIOR_CALMAR = 0.50
PRICE_MODES = ("qfq_next_close", "current_model_vwap")


@dataclass
class Trade:
    profile_id: str
    profile_name: str
    price_mode: str
    code: str
    name: str
    industry: str
    signal_date: str
    buy_date: str
    sell_date: str
    holding_days: int
    ret: float
    max_dd: float
    prior_n: int
    prior_win_rate: float | None
    prior_avg_ret: float | None
    prior_avg_dd: float | None
    prior_calmar: float | None
    amt_r20: float
    price60: float
    selected: bool


def load_market_data() -> tuple[dict[str, np.ndarray], dict[str, tuple[str, str]]]:
    con = duckdb.connect(str(MARKET_DB), read_only=True)
    try:
        con.execute(f"ATTACH '{SMART_DB}' AS sm (READ_ONLY)")
        raw = con.execute(
            """
            SELECT k.code, k.date, k.low, k.close, k.volume, k.amount
            FROM v_price_kline_qfq k
            INNER JOIN sm.dim_active_a_stock s ON k.code = s.stock_code
            ORDER BY k.code, k.date
            """
        ).fetchnumpy()
        meta_rows = con.execute(
            """
            SELECT s.stock_code, s.stock_name, COALESCE(a.tdx_l1_name, '未知') AS industry
            FROM sm.dim_active_a_stock s
            LEFT JOIN sm.dim_stock_archetype_latest a ON s.stock_code = a.stock_code
            """
        ).fetchall()
    except duckdb.IOException as e:
        print(f"Warning: SMART_DB locked/unavailable, falling back to all market codes: {e}")
        raw = con.execute(
            """
            SELECT code, date, low, close, volume, amount
            FROM v_price_kline_qfq
            ORDER BY code, date
            """
        ).fetchnumpy()
        meta_rows = []
    con.close()

    codes = raw["code"]
    unique_codes, counts = np.unique(codes, return_counts=True)
    by_code: dict[str, np.ndarray] = {}
    idx = 0
    for code_raw, cnt in zip(unique_codes, counts):
        code = normalize_code(code_raw)
        sl = slice(idx, idx + cnt)
        by_code[code] = {
            "date": raw["date"][sl],
            "low": raw["low"][sl].astype(np.float64),
            "close": raw["close"][sl].astype(np.float64),
            "volume": raw["volume"][sl].astype(np.float64),
            "amount": raw["amount"][sl].astype(np.float64),
        }
        idx += cnt

    meta = {normalize_code(code): (str(name), str(industry)) for code, name, industry in meta_rows}
    return by_code, meta


def prior_metrics(rets: list[float], dds: list[float]) -> tuple[int, float | None, float | None, float | None, float | None]:
    if not rets:
        return 0, None, None, None, None
    rr = np.asarray(rets[-HISTORY_LOOKBACK_SIGNALS:], dtype=np.float64)
    dd = np.asarray(dds[-HISTORY_LOOKBACK_SIGNALS:], dtype=np.float64)
    avg_ret = float(np.mean(rr))
    avg_dd = float(np.mean(dd))
    win_rate = float(np.mean(rr > 0))
    calmar = avg_ret / max(abs(avg_dd), 0.005)
    return len(rr), win_rate, avg_ret, avg_dd, calmar


def profile_trades(profile: dict[str, Any], by_code: dict[str, dict[str, np.ndarray]], meta: dict[str, tuple[str, str]]) -> list[Trade]:
    fast = int(profile["macd_fast"])
    slow = int(profile["macd_slow"])
    sig = int(profile["macd_signal"])
    holding_days = int(profile["holding_days"])
    min_signals = int(profile.get("min_signals", 1))
    vol_min = float(profile.get("vol_ratio_min", 1.0))
    amt_min = float(profile.get("amt_ratio_min", 1.0))
    price_max = float(profile.get("price_pos_max", 1.0))
    require_dif_positive = bool(profile.get("dif_positive", False))

    trades: list[Trade] = []
    warmup = slow + sig + max(60, holding_days) + 2

    for code, rows in by_code.items():
        dates = rows["date"]
        low = rows["low"]
        close = rows["close"]
        volume = rows["volume"]
        amount = rows["amount"]
        n = len(close)
        if n < warmup:
            continue

        dif = ema_np(close, fast) - ema_np(close, slow)
        dea = ema_np(dif, sig)
        vol_ma20 = sma_np(volume, 20)
        amt_ma20 = sma_np(amount, 20)
        max60 = rolling_max_np(close, 60)
        crosses = np.where((dif[:-1] < dea[:-1]) & (dif[1:] > dea[1:]))[0] + 1

        past: dict[str, list[tuple[int, float, float]]] = {mode: [] for mode in PRICE_MODES}
        name, industry = meta.get(code, ("", "未知"))

        for si in crosses:
            buy_i = si + 1
            sell_i = buy_i + holding_days
            if sell_i >= n:
                continue
            if volume[buy_i] <= 0 or amount[buy_i] <= 0:
                continue
            if (
                vol_ma20[si] <= 0
                or np.isnan(vol_ma20[si])
                or amt_ma20[si] <= 0
                or np.isnan(amt_ma20[si])
                or max60[si] <= 0
            ):
                continue

            vol_r20 = float(volume[si] / vol_ma20[si])
            amt_r20 = float(amount[si] / amt_ma20[si])
            price60 = float(close[si] / max60[si])
            sell_price = float(close[sell_i])
            hold_low = float(np.min(low[buy_i : sell_i + 1]))
            prices = {
                "qfq_next_close": float(close[buy_i]),
                "current_model_vwap": float(amount[buy_i] / (volume[buy_i] * 100)),
            }
            for mode, buy_price in prices.items():
                if buy_price <= 0:
                    continue
                ret = (sell_price - buy_price) / buy_price
                max_dd = min(0.0, (hold_low - buy_price) / buy_price)
                prior_completed = [row for row in past[mode] if row[0] < si]
                past_rets = [row[1] for row in prior_completed]
                past_dds = [row[2] for row in prior_completed]

                prior_n, prior_wr, prior_ret, prior_dd, prior_calmar = prior_metrics(past_rets, past_dds)
                selected = bool(
                    vol_r20 >= vol_min
                    and amt_r20 >= amt_min
                    and price60 <= price_max
                    and (not require_dif_positive or dif[si] > 0)
                    and prior_n >= min_signals
                    and (prior_wr or 0.0) >= MIN_PRIOR_WIN_RATE
                    and (prior_calmar or 0.0) >= MIN_PRIOR_CALMAR
                )

                trades.append(
                    Trade(
                        profile_id=str(profile["id"]),
                        profile_name=str(profile["name"]),
                        price_mode=mode,
                        code=code,
                        name=name,
                        industry=industry,
                        signal_date=str(dates[si]),
                        buy_date=str(dates[buy_i]),
                        sell_date=str(dates[sell_i]),
                        holding_days=holding_days,
                        ret=ret,
                        max_dd=max_dd,
                        prior_n=prior_n,
                        prior_win_rate=prior_wr,
                        prior_avg_ret=prior_ret,
                        prior_avg_dd=prior_dd,
                        prior_calmar=prior_calmar,
                        amt_r20=amt_r20,
                        price60=price60,
                        selected=selected,
                    )
                )
                past[mode].append((sell_i, ret, max_dd))

    return trades


def metric_row(profile: dict[str, Any], price_mode: str, trades: list[Trade]) -> dict[str, Any]:
    mode_trades = [t for t in trades if t.price_mode == price_mode]
    selected = [t for t in mode_trades if t.selected]
    recent = sorted(selected, key=lambda t: (t.buy_date, t.code), reverse=True)[:RECENT_TRADE_LIMIT]

    def stats(rows: list[Trade], prefix: str) -> dict[str, Any]:
        if not rows:
            return {
                f"{prefix}_n": 0,
                f"{prefix}_win_rate": None,
                f"{prefix}_avg_ret": None,
                f"{prefix}_median_ret": None,
                f"{prefix}_avg_dd": None,
                f"{prefix}_calmar": None,
            }
        rets = np.asarray([t.ret for t in rows], dtype=np.float64)
        dds = np.asarray([t.max_dd for t in rows], dtype=np.float64)
        avg_ret = float(np.mean(rets))
        avg_dd = float(np.mean(dds))
        return {
            f"{prefix}_n": len(rows),
            f"{prefix}_win_rate": float(np.mean(rets > 0)),
            f"{prefix}_avg_ret": avg_ret,
            f"{prefix}_median_ret": float(np.median(rets)),
            f"{prefix}_avg_dd": avg_dd,
            f"{prefix}_calmar": avg_ret / max(abs(avg_dd), 0.005),
        }

    out = {
        "profile_id": profile["id"],
        "profile_name": profile["name"],
        "price_mode": price_mode,
        "macd_fast": profile["macd_fast"],
        "macd_slow": profile["macd_slow"],
        "macd_signal": profile["macd_signal"],
        "holding_days": profile["holding_days"],
        "vol_ratio_min": profile.get("vol_ratio_min", 1.0),
        "amt_ratio_min": profile.get("amt_ratio_min"),
        "price_pos_max": profile.get("price_pos_max"),
        "dif_positive": profile.get("dif_positive", False),
        "min_signals": profile.get("min_signals", 1),
        "raw_signals": len(mode_trades),
    }
    out.update(stats(selected, "selected_all"))
    out.update(stats(recent, "recent"))
    return out


def fmt_pct(v: Any) -> str:
    if v is None or v == "":
        return "-"
    return f"{float(v) * 100:.2f}%"


def write_outputs(summary: list[dict[str, Any]], trades: list[Trade]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    recent_rows = sorted([t for t in trades if t.selected], key=lambda t: (t.buy_date, t.profile_id, t.code), reverse=True)
    fieldnames = list(Trade.__dataclass_fields__.keys())
    with TRADES_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in recent_rows:
            writer.writerow(t.__dict__)

    ranked = sorted(summary, key=lambda r: ((r["recent_calmar"] or -999), (r["recent_avg_ret"] or -999)), reverse=True)
    lines = [
        "# Strategy Audit",
        "",
        "Method: walk-forward MACD golden-cross trades. Each trade uses only prior completed trades for that stock to decide whether historical quality passes the gate.",
        f"Selection gate: feature filter + prior_n >= profile min_signals + prior_win_rate >= {MIN_PRIOR_WIN_RATE:.2f} + prior_calmar >= {MIN_PRIOR_CALMAR:.2f}.",
        f"Recent window: latest {RECENT_TRADE_LIMIT} selected completed trades per strategy.",
        "",
        "| Rank | Strategy | Price Mode | Recent N | Recent Avg Ret | Recent Win | Recent Calmar | All Selected N | All Avg Ret | All Win |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(ranked, 1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(i),
                    str(row["profile_id"]),
                    str(row["price_mode"]),
                    str(row["recent_n"]),
                    fmt_pct(row["recent_avg_ret"]),
                    fmt_pct(row["recent_win_rate"]),
                    "-" if row["recent_calmar"] is None else f"{float(row['recent_calmar']):.2f}",
                    str(row["selected_all_n"]),
                    fmt_pct(row["selected_all_avg_ret"]),
                    fmt_pct(row["selected_all_win_rate"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Recent Cases",
            "",
            "| Strategy | Price Mode | Code | Name | Buy | Sell | Ret | Prior N | Prior Win | Prior Calmar | AmtR20 | Price60 |",
            "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for t in recent_rows[:40]:
        lines.append(
            "| "
            + " | ".join(
                [
                    t.profile_id,
                    t.price_mode,
                    t.code,
                    t.name,
                    t.buy_date,
                    t.sell_date,
                    fmt_pct(t.ret),
                    str(t.prior_n),
                    fmt_pct(t.prior_win_rate),
                    "-" if t.prior_calmar is None else f"{t.prior_calmar:.2f}",
                    f"{t.amt_r20:.2f}",
                    f"{t.price60:.2f}",
                ]
            )
            + " |"
        )

    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def auditable_profiles() -> list[dict[str, Any]]:
    profiles = get_strategy_profiles()
    out = []
    for pid, profile in profiles.items():
        if profile.get("formula_filter_mode"):
            continue
        out.append(profile)
    return out


def main() -> None:
    print("Loading market data...")
    by_code, meta = load_market_data()
    print(f"Loaded {len(by_code)} stocks")

    all_trades: list[Trade] = []
    summary: list[dict[str, Any]] = []
    for profile in auditable_profiles():
        print(f"Auditing {profile['id']}...")
        trades = profile_trades(profile, by_code, meta)
        all_trades.extend(trades)
        for mode in PRICE_MODES:
            summary.append(metric_row(profile, mode, trades))

    write_outputs(summary, all_trades)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {TRADES_CSV}")
    print(f"Wrote {REPORT_MD}")


if __name__ == "__main__":
    main()
