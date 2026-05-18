#!/usr/bin/env python3
"""
MACD 金叉回测 — 通达信公式移植版

策略:
  - 信号: DIF 上穿 DEA (MACD 金叉), 参数 EMA(12/26/9)
  - 买入: 金叉次日均价 (amount/volume = VWAP)
  - 卖出: 持股 N 交易日后收盘价
  - 回测: 持股 5/10/15/20/30/40/60 交易日, 找最优 N

输出:
  - macd_gcross_holding_period_summary.csv: 各持股期全市场统计
  - macd_gcross_top_stocks.csv: 最优持股期下 top 50 股票
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent
MARKET_DB = BASE / "data/market.duckdb"
SMART_DB  = BASE / "data/smartmoney.duckdb"
OUT_DIR   = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# 回测参数
# ---------------------------------------------------------------------------
HOLDING_PERIODS = [5, 10, 15, 20, 30, 40, 60]
MIN_SIGNALS     = 5       # 每只股票至少需要这么多次有效信号才纳入统计
MACD_FAST       = 12
MACD_SLOW       = 26
MACD_SIG        = 9
MIN_WIN_RATE    = 0.55    # top 50 筛选时最低胜率要求


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (kline_df, name_df)."""
    print("  连接 market.duckdb …")
    mkt_con = duckdb.connect(str(MARKET_DB), read_only=True)
    mkt_con.execute(f"ATTACH '{SMART_DB}' AS sm (READ_ONLY)")

    kline = mkt_con.execute("""
        SELECT
            k.code,
            k.date,
            k.open,
            k.high,
            k.low,
            k.close,
            k.volume,
            k.amount
        FROM v_price_kline_qfq k
        INNER JOIN sm.dim_active_a_stock s ON k.code = s.stock_code
        ORDER BY k.code, k.date
    """).df()

    names = mkt_con.execute(
        "SELECT stock_code, stock_name FROM sm.dim_active_a_stock"
    ).df()

    mkt_con.close()
    kline["date"] = pd.to_datetime(kline["date"])
    return kline, names


# ---------------------------------------------------------------------------
# MACD 计算 (通达信 EMA 公式: adjust=False, span=N)
# ---------------------------------------------------------------------------
def compute_macd(close: pd.Series):
    ema12 = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema26 = close.ewm(span=MACD_SLOW, adjust=False).mean()
    dif   = ema12 - ema26
    dea   = dif.ewm(span=MACD_SIG, adjust=False).mean()
    return dif, dea


def golden_cross_mask(dif: pd.Series, dea: pd.Series) -> pd.Series:
    """DIF 从下方上穿 DEA."""
    return (dif.shift(1) < dea.shift(1)) & (dif > dea)


# ---------------------------------------------------------------------------
# 单股回测
# ---------------------------------------------------------------------------
def backtest_stock(grp: pd.DataFrame) -> list[dict]:
    """返回该股所有 (holding_period, metrics) 记录."""
    grp = grp.reset_index(drop=True)
    n   = len(grp)

    # 需要足够的预热期 + 至少一次持股
    if n < MACD_SLOW + MACD_SIG + max(HOLDING_PERIODS) + 2:
        return []

    dif, dea = compute_macd(grp["close"])
    cross_mask = golden_cross_mask(dif, dea)
    signal_positions = grp.index[cross_mask].tolist()

    if not signal_positions:
        return []

    out = []
    for hp in HOLDING_PERIODS:
        trades_ret: list[float] = []
        trades_dd:  list[float] = []

        for sig_i in signal_positions:
            buy_i  = sig_i + 1
            sell_i = sig_i + 1 + hp
            if sell_i >= n:
                continue

            vol = grp.at[buy_i, "volume"]
            amt = grp.at[buy_i, "amount"]
            if vol <= 0 or amt <= 0:
                continue  # 停牌

            buy_price  = amt / (vol * 100)    # 次日 VWAP (volume 单位: 手=100股)
            sell_price = grp.at[sell_i, "close"]

            # 持仓期最低价 → 最大浮亏
            hold_low = grp.loc[buy_i:sell_i, "low"].min()
            max_dd   = (hold_low - buy_price) / buy_price   # ≤ 0

            ret = (sell_price - buy_price) / buy_price

            trades_ret.append(ret)
            trades_dd.append(max_dd)

        if len(trades_ret) < MIN_SIGNALS:
            continue

        avg_ret  = float(np.mean(trades_ret))
        avg_dd   = float(np.mean(trades_dd))     # 负数或 0
        win_rate = float(np.mean([r > 0 for r in trades_ret]))
        # 0.5% 回撤下限，防止 Calmar 因极小回撤而爆表
        calmar   = avg_ret / max(abs(avg_dd), 0.005)

        out.append({
            "holding_days":  hp,
            "signal_count":  len(trades_ret),
            "avg_return":    avg_ret,
            "avg_max_dd":    avg_dd,
            "win_rate":      win_rate,
            "calmar":        calmar,
        })

    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    t_total = time.time()

    print("=== MACD 金叉回测 ===")
    print(f"参数: EMA({MACD_FAST}/{MACD_SLOW}/{MACD_SIG}), "
          f"持股期 {HOLDING_PERIODS} 交易日\n")

    print("[1/4] 加载数据")
    kline, names = load_data()
    n_stocks = kline["code"].nunique()
    print(f"  {len(kline):,} 行, {n_stocks} 只股票\n")

    print("[2/4] 逐股计算 MACD 金叉信号并回测 …")
    t0 = time.time()
    records: list[dict] = []
    done = 0

    for code, grp in kline.groupby("code", sort=False):
        for rec in backtest_stock(grp):
            rec["code"] = code
            records.append(rec)
        done += 1
        if done % 500 == 0:
            elapsed = time.time() - t0
            pct = done / n_stocks * 100
            print(f"  {done}/{n_stocks} ({pct:.0f}%)  {elapsed:.0f}s")

    print(f"  完成: {len(records)} 条 (stock × holding_period) 记录  "
          f"耗时 {time.time()-t0:.1f}s\n")

    if not records:
        print("无有效信号，请检查数据。")
        sys.exit(1)

    res = pd.DataFrame(records)

    # -----------------------------------------------------------------------
    # [3/4] 各持股期全市场汇总
    # -----------------------------------------------------------------------
    print("[3/4] 汇总各持股期全市场统计")
    summary = (
        res.groupby("holding_days")
        .agg(
            stock_count    = ("code",        "nunique"),
            signal_count   = ("signal_count","sum"),
            median_calmar  = ("calmar",      "median"),
            median_win_rate= ("win_rate",    "median"),
            median_return  = ("avg_return",  "median"),
            median_max_dd  = ("avg_max_dd",  "median"),
        )
        .reset_index()
    )

    print()
    print("持股天数  股票数  信号总数  中位Calmar  中位胜率  中位收益  中位最大回撤")
    for _, row in summary.iterrows():
        print(f"  {int(row.holding_days):>4}天  "
              f"{int(row.stock_count):>5}  "
              f"{int(row.signal_count):>8}  "
              f"{row.median_calmar:>10.3f}  "
              f"{row.median_win_rate:>8.1%}  "
              f"{row.median_return:>8.2%}  "
              f"{row.median_max_dd:>12.2%}")

    # 最优持股期: 中位 calmar 最高
    best_row = summary.loc[summary["median_calmar"].idxmax()]
    best_hp  = int(best_row["holding_days"])
    print(f"\n★ 最优持股期: {best_hp} 交易日  "
          f"(中位 Calmar={best_row['median_calmar']:.3f}, "
          f"中位收益={best_row['median_return']:.2%}, "
          f"中位最大回撤={best_row['median_max_dd']:.2%})\n")

    # -----------------------------------------------------------------------
    # [4/4] Top 50 最适合股票
    # -----------------------------------------------------------------------
    print(f"[4/4] 筛选最优持股期 ({best_hp}天) 下 Top 50 股票")
    best_df = res[res["holding_days"] == best_hp].copy()
    best_df = best_df[best_df["win_rate"] >= MIN_WIN_RATE]
    best_df = best_df.dropna(subset=["calmar"])
    best_df = best_df.sort_values("calmar", ascending=False).head(50)
    best_df = best_df.merge(names, left_on="code", right_on="stock_code", how="left")

    display_cols = ["code", "stock_name", "signal_count",
                    "win_rate", "avg_return", "avg_max_dd", "calmar"]
    if not best_df.empty:
        print()
        print(f"{'代码':>8}  {'名称':^8}  {'信号':>4}  "
              f"{'胜率':>6}  {'均收益':>7}  {'均回撤':>7}  {'Calmar':>7}")
        for _, r in best_df[display_cols].iterrows():
            name_str = str(r.get("stock_name", ""))[:8]
            print(f"  {r['code']:>6}  {name_str:<8}  {int(r['signal_count']):>4}  "
                  f"{r['win_rate']:>6.1%}  {r['avg_return']:>7.2%}  "
                  f"{r['avg_max_dd']:>7.2%}  {r['calmar']:>7.3f}")
    else:
        print(f"  (无满足 win_rate≥{MIN_WIN_RATE:.0%} 且 calmar 有效的股票)")

    # -----------------------------------------------------------------------
    # 保存 CSV
    # -----------------------------------------------------------------------
    summary_path = OUT_DIR / "macd_gcross_holding_period_summary.csv"
    top_path     = OUT_DIR / "macd_gcross_top_stocks.csv"

    summary.to_csv(summary_path, index=False)
    if not best_df.empty:
        best_df[display_cols + ["holding_days"]].to_csv(top_path, index=False)

    print(f"\nCSV 已保存:")
    print(f"  {summary_path}")
    print(f"  {top_path}")
    print(f"\n总耗时: {time.time()-t_total:.1f}s")


if __name__ == "__main__":
    main()
