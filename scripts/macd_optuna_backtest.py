#!/usr/bin/env python3
"""
macd_optuna_backtest.py — DuckDB + numpy + Optuna，无 pandas

流程:
  1. DuckDB 读取全量日线 (numpy)
  2. 计算 3 组 MACD 参数下的金叉信号 + 4 个因子特征
  3. 信号写入 in-memory DuckDB 表
  4. Optuna 300 trials: 搜索 MACD 参数 + 过滤条件 + 持股天数
  5. 最优参数下每只股票打分，Top 10 + 原因分析

因子说明:
  dif_val    金叉日 DIF 值，正 = 零轴上方（强势金叉），负 = 底部反弹型
  vol_r20    金叉日成交量 / 前 20 日均量（量能放大倍数）
  amt_r20    金叉日成交额 / 前 20 日均额（换手率变化代理）
  price60    金叉日收盘 / 前 60 日最高（价格位置，1=历史新高，<0.7=低位）
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import optuna
from numpy.lib.stride_tricks import sliding_window_view

optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE      = Path(__file__).resolve().parent.parent
MARKET_DB = BASE / "data/market.duckdb"
SMART_DB  = BASE / "data/smartmoney.duckdb"
OUT_DIR   = Path(__file__).resolve().parent

HOLDING_PERIODS = [5, 10, 15, 20, 30, 40, 60]
MIN_SIG_PER_STOCK  = 2     # 每只股票至少几次有效信号（排名阶段）
MIN_SIG_GLOBAL     = 400   # Optuna trial 最少总信号数
MIN_STOCKS_GLOBAL  = 40    # Optuna trial 最少覆盖股票数（防止过拟合到少数极端股）
OPTUNA_TRIALS      = 300

# 3 组 MACD 参数: (fast, slow, signal)
MACD_COMBOS: dict[str, tuple[int, int, int]] = {
    "S": (10, 22, 8),   # 短周期
    "M": (12, 26, 9),   # 通达信默认
    "L": (14, 30, 11),  # 长周期
}


# ---------------------------------------------------------------------------
# 数值计算工具 (纯 numpy，无 pandas)
# ---------------------------------------------------------------------------

def ema_np(arr: np.ndarray, span: int) -> np.ndarray:
    """EMA，等价于 pandas ewm(span=span, adjust=False)."""
    alpha = 2.0 / (span + 1)
    c     = 1.0 - alpha
    out   = np.empty(len(arr), dtype=np.float64)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + c * out[i - 1]
    return out


def sma_np(arr: np.ndarray, window: int) -> np.ndarray:
    """简单移动均值，前 window-1 个位置为 NaN."""
    n   = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out
    kernel    = np.ones(window, dtype=np.float64) / window
    out[window - 1:] = np.convolve(arr, kernel, mode="valid")
    return out


def rolling_max_np(arr: np.ndarray, window: int) -> np.ndarray:
    """滚动最大值（向前 window 日含当日），使用 stride_tricks，O(n*w) 但全在 C 层."""
    n      = len(arr)
    padded = np.pad(arr, (window - 1, 0), mode="edge")
    # sliding_window_view: shape (n, window)
    return sliding_window_view(padded, window).max(axis=1)


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_data() -> tuple[dict[str, np.ndarray], dict]:
    """返回 (ohlcv_numpy_dict, meta_dict)."""
    print("  连接 market.duckdb ...")
    mkt = duckdb.connect(str(MARKET_DB), read_only=True)
    mkt.execute(f"ATTACH '{SMART_DB}' AS sm (READ_ONLY)")

    ohlcv = mkt.execute("""
        SELECT k.code,
               k.date,
               k.open,
               k.high,
               k.low,
               k.close,
               k.volume,
               k.amount
        FROM   v_price_kline_qfq k
        INNER JOIN sm.dim_active_a_stock s ON k.code = s.stock_code
        ORDER  BY k.code, k.date
    """).fetchnumpy()

    meta_rows = mkt.execute("""
        SELECT s.stock_code,
               s.stock_name,
               COALESCE(a.tdx_l1_name,  '未知')          AS industry,
               COALESCE(a.stock_archetype, '未知')        AS archetype,
               COALESCE(f.holder_count_change_pct, 0.0)   AS holder_chg_pct
        FROM   sm.dim_active_a_stock s
        LEFT JOIN sm.dim_stock_archetype_latest a ON s.stock_code = a.stock_code
        LEFT JOIN sm.dim_financial_latest       f ON s.stock_code = f.stock_code
    """).fetchall()

    mkt.close()

    meta = {r[0]: r[1:] for r in meta_rows}   # code → (name, industry, archetype, holder_chg_pct)
    return ohlcv, meta


# ---------------------------------------------------------------------------
# 单股信号特征计算
# ---------------------------------------------------------------------------

def signals_for_stock(
    close:   np.ndarray,
    high:    np.ndarray,
    low:     np.ndarray,
    volume:  np.ndarray,
    amount:  np.ndarray,
    fast: int, slow: int, sig: int,
) -> list[tuple]:
    """
    返回该股在此 MACD 参数下的信号特征列表，每行对应一次金叉:
    (dif_val, vol_r20, amt_r20, price60,
     ret_5, ret_10, ret_15, ret_20, ret_30, ret_40, ret_60,
     dd_5,  dd_10,  dd_15,  dd_20,  dd_30,  dd_40,  dd_60)
    """
    n = len(close)
    warmup = slow + sig + max(HOLDING_PERIODS) + 2
    if n < warmup:
        return []

    close64  = close.astype(np.float64)
    volume64 = volume.astype(np.float64)
    amount64 = amount.astype(np.float64)
    low64    = low.astype(np.float64)

    dif = ema_np(close64, fast) - ema_np(close64, slow)
    dea = ema_np(dif, sig)

    vol_ma20  = sma_np(volume64, 20)
    amt_ma20  = sma_np(amount64, 20)
    max60     = rolling_max_np(close64, 60)

    # 金叉: DIF[t-1] < DEA[t-1] 且 DIF[t] > DEA[t]
    cross_mask = (dif[:-1] < dea[:-1]) & (dif[1:] > dea[1:])
    sig_indices = np.where(cross_mask)[0] + 1   # 金叉发生在第 t 天 (dif[t] > dea[t])

    rows: list[tuple] = []
    for si in sig_indices:
        buy_i = si + 1
        if buy_i >= n:
            continue
        # 特征合法性检查
        if (vol_ma20[si] <= 0 or np.isnan(vol_ma20[si])
                or amt_ma20[si] <= 0 or np.isnan(amt_ma20[si])
                or max60[si] <= 0):
            continue
        if volume64[buy_i] <= 0 or amount64[buy_i] <= 0:
            continue  # 停牌

        # 因子
        dif_val  = float(dif[si])
        vol_r20  = float(volume64[si] / vol_ma20[si])
        amt_r20  = float(amount64[si] / amt_ma20[si])
        price60  = float(close64[si] / max60[si])

        # 买入均价 (次日 VWAP; volume 单位: 手=100股)
        buy_price = float(amount64[buy_i] / (volume64[buy_i] * 100))

        rets: list[float] = []
        dds:  list[float] = []
        for hp in HOLDING_PERIODS:
            sell_i = buy_i + hp
            if sell_i >= n:
                rets.append(None)   # DuckDB NULL，不进入 AVG
                dds.append(None)
            else:
                sell  = float(close[sell_i])
                lo    = float(np.min(low64[buy_i: sell_i + 1]))
                rets.append((sell - buy_price) / buy_price)
                dds.append((lo   - buy_price) / buy_price)

        rows.append((dif_val, vol_r20, amt_r20, price60, *rets, *dds))

    return rows


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    t0_total = time.time()
    print("=== MACD 金叉 Optuna 参数寻优 (DuckDB+numpy, 无pandas) ===\n")

    # ------------------------------------------------------------------
    # Phase 1: 加载数据
    # ------------------------------------------------------------------
    print("[1/4] 加载数据")
    ohlcv, meta = load_data()

    codes   = ohlcv["code"]
    closes  = ohlcv["close"]
    highs   = ohlcv["high"]
    lows    = ohlcv["low"]
    volumes = ohlcv["volume"]
    amounts = ohlcv["amount"]

    unique_codes, counts = np.unique(codes, return_counts=True)
    n_stocks = len(unique_codes)
    print(f"  {len(codes):,} 行, {n_stocks} 只股票\n")

    # ------------------------------------------------------------------
    # Phase 2: 计算信号特征 → DuckDB in-memory
    # ------------------------------------------------------------------
    print(f"[2/4] 计算信号 ({len(MACD_COMBOS)} 组 MACD × {n_stocks} 只股票) ...")

    mem = duckdb.connect(":memory:")
    hp_col_defs = ", ".join(
        f"ret_{hp} DOUBLE, dd_{hp} DOUBLE" for hp in HOLDING_PERIODS
    )
    mem.execute(f"""
        CREATE TABLE signals (
            code    VARCHAR,
            combo   VARCHAR,
            dif_val DOUBLE,
            vol_r20 DOUBLE,
            amt_r20 DOUBLE,
            price60 DOUBLE,
            {hp_col_defs}
        )
    """)

    n_cols  = 6 + 2 * len(HOLDING_PERIODS)
    placeholders = ",".join(["?"] * n_cols)
    insert_sql   = f"INSERT INTO signals VALUES ({placeholders})"
    BATCH        = 8_000
    batch: list[tuple] = []
    t0 = time.time()
    idx = 0

    for ci, (code, cnt) in enumerate(zip(unique_codes, counts)):
        sl = slice(idx, idx + cnt)
        c  = closes[sl].astype(np.float64)
        h  = highs[sl].astype(np.float64)
        lo = lows[sl].astype(np.float64)
        v  = volumes[sl].astype(np.float64)
        a  = amounts[sl].astype(np.float64)

        for cname, (fast, slow, sig) in MACD_COMBOS.items():
            for row in signals_for_stock(c, h, lo, v, a, fast, slow, sig):
                batch.append((code, cname) + row)

        if len(batch) >= BATCH:
            mem.executemany(insert_sql, batch)
            batch.clear()

        if (ci + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  {ci+1}/{n_stocks} ({(ci+1)/n_stocks:.0%})  "
                  f"{elapsed:.0f}s")
        idx += cnt

    if batch:
        mem.executemany(insert_sql, batch)

    total_sigs = mem.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    print(f"  完成: {total_sigs:,} 条信号  耗时 {time.time()-t0:.1f}s\n")

    if total_sigs == 0:
        print("无信号，检查数据.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Phase 3: Optuna 寻优
    # ------------------------------------------------------------------
    print(f"[3/4] Optuna 寻优 ({OPTUNA_TRIALS} trials) ...")

    def objective(trial: optuna.Trial) -> float:
        combo    = trial.suggest_categorical("combo",        list(MACD_COMBOS.keys()))
        hp       = trial.suggest_categorical("holding_days", HOLDING_PERIODS)
        vol_min  = trial.suggest_float("vol_ratio_min",  0.3,  3.0)
        amt_min  = trial.suggest_float("amt_ratio_min",  0.3,  2.5)
        price_mx = trial.suggest_float("price_pos_max",  0.55, 1.0)
        dif_pos  = trial.suggest_categorical("dif_positive", [0, 1])

        where = (
            f"combo = '{combo}'"
            f" AND vol_r20 >= {vol_min}"
            f" AND amt_r20 >= {amt_min}"
            f" AND price60 <= {price_mx}"
            f" AND ret_{hp} IS NOT NULL"
            f" AND  dd_{hp} IS NOT NULL"
        )
        if dif_pos:
            where += " AND dif_val > 0"

        row = mem.execute(f"""
            SELECT COUNT(DISTINCT code)                                   AS n_stocks,
                   COUNT(*)                                               AS n,
                   AVG(CASE WHEN ret_{hp}>0 THEN 1.0 ELSE 0.0 END)      AS win_rate,
                   MEDIAN(ret_{hp})                                       AS med_ret,
                   MEDIAN(dd_{hp})                                        AS med_dd
            FROM   signals
            WHERE  {where}
        """).fetchone()

        n_stocks, n, win_rate, med_ret, med_dd = row
        if (n is None or n < MIN_SIG_GLOBAL
                or n_stocks is None or n_stocks < MIN_STOCKS_GLOBAL
                or med_ret is None):
            return -999.0
        calmar = med_ret / max(abs(med_dd or 0.0), 0.005)
        result = calmar * win_rate
        if not np.isfinite(result):
            return -999.0
        return float(result)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)

    bp    = study.best_params
    score = study.best_value
    print(f"\n  最佳参数: {bp}")
    print(f"  最佳得分 (Calmar×WinRate): {score:.4f}\n")

    best_combo  = bp["combo"]
    best_hp     = bp["holding_days"]
    best_vmin   = bp["vol_ratio_min"]
    best_amin   = bp["amt_ratio_min"]
    best_pmx    = bp["price_pos_max"]
    best_dif    = bp["dif_positive"]
    fast, slow, sig = MACD_COMBOS[best_combo]

    best_where = (
        f"combo = '{best_combo}'"
        f" AND vol_r20 >= {best_vmin}"
        f" AND amt_r20 >= {best_amin}"
        f" AND price60 <= {best_pmx}"
        f" AND ret_{best_hp} IS NOT NULL"
        f" AND  dd_{best_hp} IS NOT NULL"
    )
    if best_dif:
        best_where += " AND dif_val > 0"

    # ------------------------------------------------------------------
    # Phase 4: 最优参数下每股打分 → Top 10
    # ------------------------------------------------------------------
    print(f"[4/4] 最优参数下各股排名 (EMA {fast}/{slow}/{sig}, 持股 {best_hp}天) ...")

    per_stock = mem.execute(f"""
        SELECT code,
               COUNT(*)                                                  AS n,
               AVG(CASE WHEN ret_{best_hp}>0 THEN 1.0 ELSE 0.0 END)    AS win_rate,
               AVG(ret_{best_hp})                                        AS avg_ret,
               AVG(dd_{best_hp})                                         AS avg_dd,
               AVG(vol_r20)                                              AS avg_vol_r20,
               AVG(amt_r20)                                              AS avg_amt_r20,
               AVG(price60)                                              AS avg_price60,
               AVG(dif_val)                                              AS avg_dif_val
        FROM   signals
        WHERE  {best_where}
        GROUP  BY code
        HAVING COUNT(*) >= {MIN_SIG_PER_STOCK}
    """).fetchall()

    if not per_stock:
        print("  无满足条件的股票，请放宽过滤阈值。")
        sys.exit(1)

    stock_list = []
    for code, n, win_rate, avg_ret, avg_dd, avg_vr, avg_ar, avg_p, avg_dif in per_stock:
        calmar = avg_ret / max(abs(avg_dd or 0.0), 0.005)
        m      = meta.get(code, ("", "未知", "未知", 0.0))
        stock_list.append({
            "code":       code,
            "name":       m[0],
            "industry":   m[1],
            "archetype":  m[2],
            "holder_chg": float(m[3]),
            "n":          int(n),
            "win_rate":   float(win_rate),
            "avg_ret":    float(avg_ret),
            "avg_dd":     float(avg_dd or 0.0),
            "calmar":     float(calmar),
            "score":      float(calmar * win_rate),
            "avg_vol_r20":float(avg_vr  or 0.0),
            "avg_amt_r20":float(avg_ar  or 0.0),
            "avg_price60":float(avg_p   or 0.0),
            "avg_dif_val":float(avg_dif or 0.0),
            "macd_combo": best_combo,
            "holding_days": best_hp,
        })

    stock_list.sort(key=lambda x: x["score"], reverse=True)
    top10 = stock_list[:10]

    # ------------------------------------------------------------------
    # 全局过滤后统计
    # ------------------------------------------------------------------
    g_row = mem.execute(f"""
        SELECT COUNT(DISTINCT code) AS n_stk,
               COUNT(*)             AS n_sig,
               AVG(CASE WHEN ret_{best_hp}>0 THEN 1.0 ELSE 0.0 END) AS win_rate,
               MEDIAN(ret_{best_hp})  AS med_ret,
               MEDIAN(dd_{best_hp})   AS med_dd
        FROM   signals
        WHERE  {best_where}
    """).fetchone()
    g_stk, g_sig, g_wr, g_ret, g_dd = g_row

    # ------------------------------------------------------------------
    # 输出
    # ------------------------------------------------------------------
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  最优 MACD 参数: EMA({fast}/{slow}/{sig})  持股: {best_hp} 交易日")
    dif_desc = "要求零轴上方 (DIF>0)" if best_dif else "零轴上下均可"
    print(f"  过滤条件: 量比≥{best_vmin:.2f}x  额比≥{best_amin:.2f}x  "
          f"价格位置≤{best_pmx:.0%}  {dif_desc}")
    print(f"  过滤后信号: {g_sig} 条 / {g_stk} 只股票  "
          f"整体胜率 {g_wr:.1%}  中位收益 {g_ret:+.2%}  中位回撤 {g_dd:.2%}")
    print(f"{sep}\n")

    for rank, s in enumerate(top10, 1):
        tag = f"#{rank}"
        print(f"{tag}  {s['code']}  {s['name'][:8]:<8}  "
              f"[{s['industry'][:6]:<6}]  [{s['archetype'][:8]:<8}]")
        conf = "⚠小样本" if s["n"] < 4 else ""
        print(f"    信号{s['n']}次{conf}  胜率{s['win_rate']:.1%}  "
              f"均收益{s['avg_ret']:+.2%}  均最大回撤{s['avg_dd']:.2%}  "
              f"Calmar{s['calmar']:.2f}")

        reasons: list[str] = []

        # 1. MACD 位置
        if s["avg_dif_val"] > 0:
            reasons.append("金叉多发生在零轴上方（趋势延续型强势信号）")
        else:
            reasons.append("金叉多发生在零轴下方（底部反弹型，弹性大）")

        # 2. 量能特征
        if s["avg_vol_r20"] > 1.8:
            reasons.append(f"金叉时成交量显著放大（均量比 {s['avg_vol_r20']:.1f}x），资金主动介入")
        elif s["avg_vol_r20"] > 1.2:
            reasons.append(f"金叉时有温和放量（均量比 {s['avg_vol_r20']:.1f}x）")

        # 3. 换手率变化
        if s["avg_amt_r20"] > 1.5:
            reasons.append(f"额比 {s['avg_amt_r20']:.1f}x，换手率明显提升")

        # 4. 价格位置
        if s["avg_price60"] < 0.70:
            reasons.append(f"金叉均在近 60 日低位区域（均价格位置 {s['avg_price60']:.0%}），安全边际高")
        elif s["avg_price60"] < 0.85:
            reasons.append(f"价格位置适中（{s['avg_price60']:.0%}），兼顾上涨空间与风险")
        else:
            reasons.append(f"价格位置偏高（{s['avg_price60']:.0%}），强势股惯性延续")

        # 5. 机构/筹码
        arch = s["archetype"]
        if arch == "高质量稳健型":
            reasons.append("基本面属高质量稳健型，机构长期持有偏好强")
        elif arch == "成长兑现型":
            reasons.append("成长兑现型标的，业绩弹性驱动技术信号有效")
        hc = s["holder_chg"]
        if hc < -0.05:
            reasons.append(f"股东数减少 {abs(hc):.1%}，筹码持续集中")
        elif hc > 0.10:
            reasons.append(f"股东数增加 {hc:.1%}，筹码趋于分散（关注出货风险）")

        for i, r in enumerate(reasons, 1):
            prefix = "    ▸" if i > 1 else "    原因:"
            print(f"{prefix} {r}")
        print()

    # ------------------------------------------------------------------
    # 保存 CSV
    # ------------------------------------------------------------------
    out_path = OUT_DIR / "macd_optuna_top10.csv"
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(top10[0].keys()))
        writer.writeheader()
        writer.writerows(top10)

    print(f"CSV 已保存: {out_path}")
    print(f"总耗时: {time.time()-t0_total:.1f}s")


if __name__ == "__main__":
    main()
