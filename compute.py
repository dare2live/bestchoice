"""
compute.py — MACD 选股后端计算引擎（多策略版）

职责:
  1. 历史回测指标（win_rate / avg_ret / calmar）缓存到 duckdb
  2. 当前 MACD 状态（刚金叉 / 持仓期 / 即将金叉 / 刚死叉 / 等待）
  3. 合并返回给前端，包括通达信三公式 f1/f3/f5 命中
  4. 支持多策略切换：内置参数、Optuna 最优参数、通达信参数
"""
from __future__ import annotations

from collections import OrderedDict
import csv
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

import json

import duckdb
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# ---------------------------------------------------------------------------
# 路径与缓存
# ---------------------------------------------------------------------------
_CHUNKY = Path(__file__).resolve().parent.parent / "chunkymonkey"
MARKET_DB = _CHUNKY / "data/market.duckdb"
SMART_DB  = _CHUNKY / "data/smartmoney.duckdb"
# 优化脚本与 CSV 统一放在 bestchoice/scripts/（已从 chunkymonkey 迁移过来）
SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
CACHE_DIR = Path(__file__).resolve().parent
CACHE_MAX_AGE = 86400  # 缓存有效期 24h
HOLDING_PERIODS = [5, 10, 15, 20, 30, 60]  # 多持股期回测天数


# ---------------------------------------------------------------------------
# 业务常量
# ---------------------------------------------------------------------------
CROSS_WINDOW = 5
IMMINENT_DAYS = 5
IMMINENT_GAP = 0.012

S_JUST = "刚金叉"
S_HOLD = "持仓期"
S_IMMIN = "即将金叉"
S_DEATH = "刚死叉"
S_WAIT = "等待"

STATUS_ORDER = {S_JUST: 1, S_IMMIN: 2, S_HOLD: 3, S_DEATH: 4, S_WAIT: 5}
STATUS_COLOR = {
    S_JUST: "green",
    S_IMMIN: "yellow",
    S_HOLD: "blue",
    S_DEATH: "red",
    S_WAIT: "gray",
}


# ---------------------------------------------------------------------------
# 参数说明（高 / 低含义）
# ---------------------------------------------------------------------------
PARAM_DESCRIPTIONS = {
    "macd_fast": {
        "label": "快线周期",
        "desc": "EMA 快线周期，越小越接近现价，反应更快。",
        "low_hint": "更敏感，抓得更早，但噪音更明显。",
        "high_hint": "更平滑，信号更少但更稳定。",
    },
    "macd_slow": {
        "label": "慢线周期",
        "desc": "EMA 慢线周期，定义中期趋势。",
        "low_hint": "更快识别趋势反转，切入更积极。",
        "high_hint": "趋势定义更稳健，反应更慢。",
    },
    "macd_signal": {
        "label": "信号线周期",
        "desc": "DEA 平滑参数，直接决定金叉/死叉密度。",
        "low_hint": "交叉更密，容易出现短线震荡。",
        "high_hint": "交叉更稀，交易更集中。",
    },
    "holding_days": {
        "label": "持股天数",
        "desc": "买入后固定持有周期，控制兑现节奏。",
        "low_hint": "更快回收，降低回撤压力。",
        "high_hint": "更容易吃完整波段，但回撤扩张。",
    },
    "amt_ratio_min": {
        "label": "额比阈值",
        "desc": "金叉日成交额/20 日均额，用于判断主力资金参与度。",
        "low_hint": "入场机会更多，噪音样本上升。",
        "high_hint": "只留更强资金窗口，筛选更严。",
    },
    "price_pos_max": {
        "label": "价格位置上限",
        "desc": "金叉当日价格与近 60 日高点比例，越低说明越低位。",
        "low_hint": "偏向抄底附近，追高风险更小。",
        "high_hint": "临近高位区域，后续上行空间受限。",
    },
}


# ---------------------------------------------------------------------------
# 策略配置
# ---------------------------------------------------------------------------
DEFAULT_PROFILES = {
    "first_dark_legacy": {
        "id": "first_dark_legacy",
        "name": "深色第一版复现 · EMA(10,22,8)",
        "source": "recovered first dark version",
        "strategy_type": "first_dark_legacy",
        "macd_fast": 10,
        "macd_slow": 22,
        "macd_signal": 8,
        "holding_days": 15,
        "amt_ratio_min": 1.42,
        "price_pos_max": 0.60,
        "min_signals": 2,
    },
    "macd_10_22_8_h15": {
        "id": "macd_10_22_8_h15",
        "name": "基准策略 · EMA(10,22,8)",
        "source": "内置基准",
        "macd_fast": 10,
        "macd_slow": 22,
        "macd_signal": 8,
        "holding_days": 15,
        "amt_ratio_min": 1.42,
        "price_pos_max": 0.60,
        "min_signals": 1,
    },
}

MACD_COMBOS = {
    "S": (10, 22, 8),
    "M": (12, 26, 9),
    "L": (14, 30, 11),
}

FORMULA_PROFILE_RULES = {
    "f1_hit": {
        "name": "通达信公式一（F1）",
        "desc": "MA5 长期低于 MA90 后突破 MA145，且近 45 天 MA5 持续上行。",
        "low_hint": "放宽时命中更快，适合扩大候选池。",
        "high_hint": "提高严格度可显著减少噪音信号。",
        "rule": (
            "关键条件：MA5 在 MA90 下方 ≥45 天；"
            "最近 10 天 MA5 上涨天数≥7；"
            "10 天内突破 MA145 且价格持续在 MA145 上方。"
        ),
    },
    "f3_hit": {
        "name": "通达信公式三（F3）",
        "desc": "多级买卖信号迭代 + 均线多头排列 + 回撤约束。",
        "low_hint": "减少参数约束后可得到更多“刚形成”的中短线机会。",
        "high_hint": "加强约束可过滤高波动样本，降低假信号。",
        "rule": (
            "关键条件：近 45 天内处于卖出后 3 天以内；"
            "90 天内快速反弹率≥40%；均线多头排列，且回撤在可控区间。"
        ),
    },
    "f5_hit": {
        "name": "通达信公式五（F5）",
        "desc": "连跌后首日止跌回升且 MACD 金叉，DIFF ≥ 0。",
        "low_hint": "条件放宽可提高早期介入机会。",
        "high_hint": "要求严格可提升信号可靠度，但样本会变少。",
        "rule": (
            "关键条件：最近 4 日内由下跌转为上涨；"
            "MACD 发生金叉，且 DIFF 非负。"
        ),
    },
    "formula_any": {
        "name": "公式策略 · 任一命中",
        "desc": "至少一个公式（F1/F3/F5）命中后才保留。",
        "low_hint": "放宽后更容易命中，适合扩大样本。",
        "high_hint": "收紧后更偏向稳定样本。",
        "rule": "条件：F1、F3、F5 任一为真。",
    },
}

# hist_cache_profile_id: formula profiles share historical cache with the base MACD profile
# (same EMA params → identical historical computation → no need to recompute)
_FORMULA_HIST_CACHE = "optuna_best"  # will be the default/first computed profile

FORMULA_PROFILES = {
    "f1_only": {
        "id": "formula_f1",
        "name": "公式策略 · F1 命中",
        "source": "chunky-monkey/screening_engine.py",
        "formula_filter_mode": "single",
        "formula_keys": ("f1_hit",),
        "formula_min_hits": 1,
        "formula_rule_id": "f1_hit",
        "macd_fast": 10,
        "macd_slow": 22,
        "macd_signal": 8,
        "holding_days": 15,
        "amt_ratio_min": 1.0,
        "price_pos_max": 1.0,
        "min_signals": 1,
        "hist_cache_profile_id": _FORMULA_HIST_CACHE,
    },
    "f3_only": {
        "id": "formula_f3",
        "name": "公式策略 · F3 命中",
        "source": "chunky-monkey/screening_engine.py",
        "formula_filter_mode": "single",
        "formula_keys": ("f3_hit",),
        "formula_min_hits": 1,
        "formula_rule_id": "f3_hit",
        "macd_fast": 10,
        "macd_slow": 22,
        "macd_signal": 8,
        "holding_days": 15,
        "amt_ratio_min": 1.0,
        "price_pos_max": 1.0,
        "min_signals": 1,
        "hist_cache_profile_id": _FORMULA_HIST_CACHE,
    },
    "f5_only": {
        "id": "formula_f5",
        "name": "公式策略 · F5 命中",
        "source": "chunky-monkey/screening_engine.py",
        "formula_filter_mode": "single",
        "formula_keys": ("f5_hit",),
        "formula_min_hits": 1,
        "formula_rule_id": "f5_hit",
        "macd_fast": 10,
        "macd_slow": 22,
        "macd_signal": 8,
        "holding_days": 15,
        "amt_ratio_min": 1.0,
        "price_pos_max": 1.0,
        "min_signals": 1,
        "hist_cache_profile_id": _FORMULA_HIST_CACHE,
    },
    "f123_any": {
        "id": "formula_any",
        "name": "公式策略 · F1/F3/F5 任一命中",
        "source": "chunky-monkey/screening_engine.py",
        "formula_filter_mode": "any",
        "formula_keys": ("f1_hit", "f3_hit", "f5_hit"),
        "formula_min_hits": 1,
        "formula_rule_id": "formula_any",
        "macd_fast": 10,
        "macd_slow": 22,
        "macd_signal": 8,
        "holding_days": 15,
        "amt_ratio_min": 1.0,
        "price_pos_max": 1.0,
        "min_signals": 1,
        "hist_cache_profile_id": _FORMULA_HIST_CACHE,
    },
}


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def normalize_code(v: Any) -> str:
    if isinstance(v, (bytes, bytearray, memoryview)):
        return bytes(v).decode("utf-8", errors="ignore")
    if v is None:
        return ""
    return str(v)


# Cache the latest data date so we don't re-query on every computation
_LATEST_DATA_DATE: Optional[str] = None
_LATEST_DATA_DATE_LOCK = threading.Lock()

def get_latest_data_date() -> str:
    """Return the latest trading date (cached after first call)."""
    global _LATEST_DATA_DATE
    if _LATEST_DATA_DATE is not None:
        return _LATEST_DATA_DATE
    with _LATEST_DATA_DATE_LOCK:
        if _LATEST_DATA_DATE is not None:
            return _LATEST_DATA_DATE
        try:
            con = duckdb.connect(str(MARKET_DB), read_only=True)
            row = con.execute("SELECT MAX(date) FROM v_price_kline_qfq").fetchone()
            con.close()
            _LATEST_DATA_DATE = str(row[0]) if row and row[0] else "未知"
        except Exception:
            _LATEST_DATA_DATE = "未知"
    return _LATEST_DATA_DATE


def _attach_smart_db(con) -> None:
    """Attach smartmoney.duckdb as 'sm'. Safe to call from multiple threads —
    DuckDB in-process connections share the catalog, so 'sm' might already
    be attached by another thread."""
    try:
        con.execute(f"ATTACH '{SMART_DB}' AS sm (READ_ONLY)")
    except Exception as e:
        if "already exists" not in str(e).lower():
            raise  # real error, propagate


def _to_float(v: Any) -> Optional[float]:
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _to_int(v: Any, default: int = 0) -> int:
    x = _to_float(v)
    return default if x is None else int(x)


def _to_bool(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, bool):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return bool(int(v))
    s = str(v).strip().lower()
    return s in {"1", "true", "t", "y", "yes", "是", "命中"}


def ema_np(arr: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1)
    c = 1.0 - alpha
    out = np.empty(len(arr), dtype=np.float64)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + c * out[i - 1]
    return out


def sma_np(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(arr), np.nan, dtype=np.float64)
    if len(arr) >= window:
        kernel = np.ones(window, dtype=np.float64) / window
        out[window - 1 :] = np.convolve(arr, kernel, mode="valid")
    return out


def rolling_max_np(arr: np.ndarray, window: int) -> np.ndarray:
    padded = np.pad(arr.astype(np.float64), (window - 1, 0), mode="edge")
    return sliding_window_view(padded, window).max(axis=1)


# ---------------------------------------------------------------------------
# 当前 MACD 状态
# ---------------------------------------------------------------------------
def current_status(dif: np.ndarray, dea: np.ndarray, close: np.ndarray) -> tuple[str, Optional[int], float]:
    n = len(dif)
    if n < 3:
        return S_WAIT, None, 0.0

    gap_arr = dif - dea
    gap_now = float(gap_arr[-1])

    if gap_now > 0:
        for i in range(1, min(n, 30)):
            if gap_arr[-(i + 1)] <= 0:
                if i <= CROSS_WINDOW:
                    return S_JUST, i, gap_now
                return S_HOLD, i, gap_now
        return S_HOLD, None, gap_now

    for i in range(1, min(n, 10)):
        if gap_arr[-(i + 1)] >= 0:
            if i <= CROSS_WINDOW:
                return S_DEATH, i, gap_now
            break

    if n >= 3:
        rate = float(gap_arr[-1] - gap_arr[-2])
        rate2 = float(gap_arr[-2] - gap_arr[-3])
        converging = rate > 0 and rate2 > 0
        gap_ratio = abs(gap_now) / max(abs(float(close[-1])), 0.001)
        if converging and gap_ratio < IMMINENT_GAP:
            days_est = int(-gap_now / rate) if rate > 1e-8 else 99
            if days_est <= IMMINENT_DAYS:
                return S_IMMIN, days_est, gap_now

    return S_WAIT, None, gap_now


# ---------------------------------------------------------------------------
# 策略加载
# ---------------------------------------------------------------------------
def _parse_optuna_profile() -> Optional[dict[str, Any]]:
    csv_path = SCRIPTS_DIR / "macd_optuna_top10.csv"
    if not csv_path.exists():
        return None

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None

    def score_fn(r: dict[str, str]) -> float:
        calmar = _to_float(r.get("calmar"))
        score = _to_float(r.get("score"))
        if score is not None:
            return score
        if calmar is not None:
            return calmar
        return 0.0

    row = max(rows, key=score_fn)
    combo_key = str(row.get("macd_combo", "S")).strip().upper() or "S"
    fast, slow, signal = MACD_COMBOS.get(combo_key, (10, 22, 8))

    return {
        "id": "optuna_best",
        "name": f"Optuna 最优 · EMA({fast}/{slow}/{signal})",
        "source": "chunkymonkey/macd_optuna_top10.csv",
        "strategy_type": "macd_optuna",
        "macd_fast": fast,
        "macd_slow": slow,
        "macd_signal": signal,
        "holding_days": _to_int(row.get("holding_days"), 15),
        "amt_ratio_min": _to_float(row.get("avg_amt_r20")) or 1.5,
        "price_pos_max": _to_float(row.get("avg_price60")) or 0.60,
        "min_signals": 1,
        "optuna_score": _to_float(row.get("score")) or 0.0,
        "optuna_n": _to_int(row.get("n"), 0),
        "formula_rule_id": "optuna_macd",
    }


def _parse_golden_profile() -> Optional[dict[str, Any]]:
    summary = SCRIPTS_DIR / "macd_gcross_holding_period_summary.csv"
    if not summary.exists():
        return None

    best_holding = 15
    best_calmar = None

    with summary.open("r", encoding="utf-8", newline="") as f:
        rows = csv.DictReader(f)
        for r in rows:
            cal = _to_float(r.get("median_calmar"))
            if cal is None:
                continue
            if best_calmar is None or cal > best_calmar:
                best_calmar = cal
                best_holding = _to_int(r.get("holding_days"), best_holding)

    if best_calmar is None:
        return None

    return {
        "id": "tdx_12_26_9",
        "name": "通达信参数 · EMA(12,26,9)",
        "source": "chunkymonkey/macd_gcross_holding_period_summary.csv",
        "strategy_type": "macd_golden",
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "holding_days": best_holding,
        "amt_ratio_min": 1.0,
        "price_pos_max": 0.70,
        "min_signals": 1,
        "formula_rule_id": "golden_cross",
        "best_calmar": best_calmar,
    }


def _passes_formula_filter(hits: dict[str, bool], profile: dict[str, Any]) -> bool:
    mode = profile.get("formula_filter_mode")
    if not mode:
        return True

    keys = tuple(profile.get("formula_keys") or ())
    if not keys:
        return True

    values = [bool(hits.get(k, False)) for k in keys]
    if mode == "single":
        # keys 通常只有一个
        return any(values)

    if mode == "all":
        return all(values)

    if mode == "count":
        min_hits = int(profile.get("formula_min_hits", 1))
        return sum(values) >= min_hits

    # any
    return any(values)


def get_strategy_profiles() -> dict[str, dict[str, Any]]:
    profiles: OrderedDict[str, dict[str, Any]] = OrderedDict()

    # 先放置基准 + MACD 三套常用参数
    profiles.update({k: dict(v) for k, v in DEFAULT_PROFILES.items()})

    for name, (f, s, sig) in MACD_COMBOS.items():
        pid = f"macd_{f}_{s}_{sig}"
        if pid not in profiles:
            profiles[pid] = {
                "id": pid,
                "name": f"参数组 {name} · EMA({f},{s},{sig})",
                "source": "chunkymonkey 常规参数",
                "macd_fast": f,
                "macd_slow": s,
                "macd_signal": sig,
                "holding_days": 15,
                "amt_ratio_min": 1.42,
                "price_pos_max": 0.60,
                "min_signals": 1,
            }

    for k, v in FORMULA_PROFILES.items():
        if v["id"] not in profiles:
            profiles[v["id"]] = dict(v)

    # 再读取 Optuna 与通达信参数，用于“更好的买入时机/持股周期”探索
    opt = _parse_optuna_profile()
    if opt is not None:
        profiles[opt["id"]] = opt

    gdx = _parse_golden_profile()
    if gdx is not None:
        profiles[gdx["id"]] = gdx

    return dict(profiles)


def get_default_profile_id(profiles: dict[str, dict[str, Any]]) -> str:
    if "optuna_best" in profiles:
        return "optuna_best"
    if "macd_10_22_8_h15" in profiles:
        return "macd_10_22_8_h15"
    return next(iter(profiles.keys()))


def _safe_cache_path(profile_id: str) -> Path:
    safe = "".join(c if c.isalnum() or c in {"_", "-"} else "_" for c in profile_id)
    return CACHE_DIR / f"cache_{safe}.duckdb"


def _cache_fresh(profile_id: str) -> bool:
    p = _safe_cache_path(profile_id)
    if not p.exists():
        return False
    if (time.time() - p.stat().st_mtime) >= CACHE_MAX_AGE:
        return False
    # Invalidate cache if it predates the multi-horizon schema
    try:
        con = duckdb.connect(str(p), read_only=True)
        info = con.execute("PRAGMA table_info(hist_metrics)").fetchall()
        con.close()
        return any(r[1] == "horizons_json" for r in info)
    except Exception:
        return False


def _load_cache(profile_id: str) -> dict[str, dict[str, Any]]:
    db = _safe_cache_path(profile_id)
    con = duckdb.connect(str(db), read_only=True)
    try:
        info = con.execute("PRAGMA table_info(hist_metrics)").fetchall()
    except Exception:
        con.close()
        return {}
    col_names = {r[1] for r in info}
    has_status   = "history_status"   in col_names
    has_best_hp  = "best_holding_days" in col_names
    has_horizons = "horizons_json"     in col_names

    cols = "code, signal_count, win_rate, avg_ret, avg_dd, calmar"
    if has_status:   cols += ", history_status"
    if has_best_hp:  cols += ", best_holding_days"
    if has_horizons: cols += ", horizons_json"

    rows = con.execute(f"SELECT {cols} FROM hist_metrics").fetchall()
    con.close()

    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        i = 0
        code = normalize_code(row[i]); i += 1
        sc   = row[i];                 i += 1
        wr   = row[i];                 i += 1
        ar   = row[i];                 i += 1
        ad   = row[i];                 i += 1
        cal  = row[i];                 i += 1
        status   = row[i] if has_status   else "ok"; i += 1 if has_status   else 0
        best_hp  = row[i] if has_best_hp  else None; i += 1 if has_best_hp  else 0
        hjson    = row[i] if has_horizons else "{}"; i += 1 if has_horizons else 0
        try:
            horizons = {int(k): v for k, v in json.loads(hjson or "{}").items()}
        except Exception:
            horizons = {}
        out[code] = {
            "signal_count":     int(sc),
            "win_rate":         _to_float(wr),
            "avg_ret":          _to_float(ar),
            "avg_dd":           _to_float(ad),
            "calmar":           _to_float(cal),
            "history_status":   status or "ok",
            "best_holding_days": int(best_hp) if best_hp is not None else None,
            "horizons":         horizons,
        }
    return out


def _save_cache(profile_id: str, metrics: dict[str, dict[str, Any]]) -> None:
    db = _safe_cache_path(profile_id)
    con = duckdb.connect(str(db))
    con.execute("DROP TABLE IF EXISTS hist_metrics")
    con.execute(
        """
        CREATE TABLE hist_metrics (
            code              VARCHAR PRIMARY KEY,
            signal_count      INTEGER,
            win_rate          DOUBLE,
            avg_ret           DOUBLE,
            avg_dd            DOUBLE,
            calmar            DOUBLE,
            history_status    VARCHAR,
            best_holding_days INTEGER,
            horizons_json     TEXT
        )
        """
    )
    rows = []
    for code, v in metrics.items():
        h_raw = v.get("horizons") or {}
        rows.append((
            code,
            int(v.get("signal_count", 0)),
            _to_float(v.get("win_rate")),
            _to_float(v.get("avg_ret")),
            _to_float(v.get("avg_dd")),
            _to_float(v.get("calmar")),
            str(v.get("history_status", "ok") or "ok"),
            int(v.get("best_holding_days") or 15),
            json.dumps({str(k): val for k, val in h_raw.items()}),
        ))
    if rows:
        con.executemany("INSERT INTO hist_metrics VALUES (?,?,?,?,?,?,?,?,?)", rows)
    con.close()


# ---------------------------------------------------------------------------
# 历史回测（带缓存）
# ---------------------------------------------------------------------------
def compute_historical(profile: dict[str, Any], progress_cb=None) -> dict[str, dict[str, Any]]:
    profile_id = profile["id"]
    cache_pid = profile.get("hist_cache_profile_id", profile_id)
    if _cache_fresh(cache_pid):
        return _load_cache(cache_pid)

    mkt = duckdb.connect(str(MARKET_DB), read_only=True)
    _attach_smart_db(mkt)
    raw = mkt.execute(
        """
        SELECT k.code, k.low, k.close, k.volume, k.amount
        FROM v_price_kline_qfq k
        INNER JOIN sm.dim_active_a_stock s ON k.code = s.stock_code
        ORDER BY k.code, k.date
        """
    ).fetchnumpy()
    mkt.close()

    if len(raw["code"]) == 0:
        _save_cache(cache_pid, {})
        return {}

    codes   = raw["code"]
    closes  = raw["close"].astype(np.float64)
    lows    = raw["low"].astype(np.float64)
    volumes = raw["volume"].astype(np.float64)
    amounts = raw["amount"].astype(np.float64)

    unique_codes, counts = np.unique(codes, return_counts=True)
    n_total = len(unique_codes)

    fast = int(profile["macd_fast"])
    slow = int(profile["macd_slow"])
    sig  = int(profile["macd_signal"])
    hp   = int(profile["holding_days"])
    min_signals = int(profile.get("min_signals", 1))

    all_periods = sorted(set(HOLDING_PERIODS + [hp]))
    warmup = slow + sig + max(all_periods) + 2
    metrics: dict[str, dict[str, Any]] = {}

    idx = 0
    for ci, (code_raw, cnt) in enumerate(zip(unique_codes, counts)):
        code = normalize_code(code_raw)
        sl   = slice(idx, idx + cnt)
        cls  = closes[sl]
        lo   = lows[sl]
        vol  = volumes[sl]
        amt  = amounts[sl]
        n    = len(cls)

        if n < warmup:
            metrics[code] = {
                "signal_count": 0, "win_rate": None, "avg_ret": None,
                "avg_dd": None, "calmar": None,
                "history_status": "insufficient_history",
                "best_holding_days": hp, "horizons": {},
            }
            idx += cnt
            continue

        dif = ema_np(cls, fast) - ema_np(cls, slow)
        dea = ema_np(dif, sig)

        cross    = (dif[:-1] < dea[:-1]) & (dif[1:] > dea[1:])
        sig_idxs = np.where(cross)[0] + 1

        h_rets: dict[int, list] = {h: [] for h in all_periods}
        h_dds:  dict[int, list] = {h: [] for h in all_periods}

        max_h = max(all_periods)
        for si in sig_idxs:
            buy_i = si + 1          # T+1: 金叉后第一个交易日买入
            if buy_i >= n or vol[buy_i] <= 0 or amt[buy_i] <= 0:
                continue
            buy_price = float(amt[buy_i] / (vol[buy_i] * 100))

            end_i    = min(buy_i + max_h + 1, n)
            lo_slice = lo[buy_i:end_i]
            cl_slice = cls[buy_i:end_i]
            cum_min  = np.minimum.accumulate(lo_slice)

            for h in all_periods:
                if h >= len(cl_slice):
                    continue
                sell_price = float(cl_slice[h])
                hold_low   = float(cum_min[h])
                r  = (sell_price - buy_price) / buy_price
                dd = (hold_low   - buy_price) / buy_price
                h_rets[h].append(r)
                h_dds[h].append(dd)

        # Per-horizon summary (only for HOLDING_PERIODS, not the extra hp if different)
        horizons: dict[int, dict] = {}
        for h in HOLDING_PERIODS:
            rr = h_rets.get(h, [])
            if rr:
                wr_  = float(np.mean([r > 0 for r in rr]))
                ar_  = float(np.mean(rr))
                ad_  = float(np.mean(h_dds[h]))
                cal_ = ar_ / max(abs(ad_), 0.005)
                horizons[h] = {
                    "win_rate": round(wr_, 4), "avg_ret": round(ar_, 4),
                    "avg_dd":   round(ad_, 4), "calmar":  round(cal_, 4),
                    "n": len(rr),
                }

        # Best holding period (highest calmar)
        best_hp  = hp
        best_cal = None
        for h, hm in horizons.items():
            if best_cal is None or hm["calmar"] > best_cal:
                best_cal, best_hp = hm["calmar"], h

        # Main metrics at profile's holding_days
        rets = h_rets.get(hp, [])
        dds  = h_dds.get(hp, [])

        if len(rets) >= min_signals:
            win_rate = float(np.mean([r > 0 for r in rets]))
            avg_ret  = float(np.mean(rets))
            avg_dd   = float(np.mean(dds))
            calmar   = avg_ret / max(abs(avg_dd), 0.005)
            metrics[code] = {
                "signal_count": len(rets),
                "win_rate": win_rate, "avg_ret": avg_ret,
                "avg_dd": avg_dd,     "calmar": calmar,
                "history_status": "ok",
                "best_holding_days": best_hp, "horizons": horizons,
            }
        else:
            metrics[code] = {
                "signal_count": int(len(rets)),
                "win_rate":  float(np.mean([r > 0 for r in rets])) if rets else None,
                "avg_ret":   float(np.mean(rets)) if rets else None,
                "avg_dd":    float(np.mean(dds))  if dds  else None,
                "calmar":    None,
                "history_status": "too_few_signals" if rets else "no_signal",
                "best_holding_days": best_hp if horizons else hp,
                "horizons":  horizons,
            }

        if progress_cb and (ci + 1) % 200 == 0:
            progress_cb(ci + 1, n_total)

        idx += cnt

    _save_cache(cache_pid, metrics)
    return metrics


# ---------------------------------------------------------------------------
# 当前状态（全量）
# ---------------------------------------------------------------------------
def _load_formula_hits() -> dict[str, dict[str, bool]]:
    out: dict[str, dict[str, bool]] = {}
    try:
        con = duckdb.connect(str(SMART_DB), read_only=True)
        rows = con.execute(
            """
            SELECT stock_code, f1_hit, f3_hit, f5_hit
            FROM (
                SELECT
                    stock_code,
                    f1_hit,
                    f3_hit,
                    f5_hit,
                    ROW_NUMBER() OVER (PARTITION BY stock_code ORDER BY screen_date DESC) AS rn
                FROM mart_stock_screening
            )
            WHERE rn = 1
            """
        ).fetchall()
        con.close()
    except Exception:
        return out

    for stock_code, f1, f3, f5 in rows:
        code = normalize_code(stock_code)
        out[code] = {
            "f1_hit": _to_bool(f1),
            "f3_hit": _to_bool(f3),
            "f5_hit": _to_bool(f5),
        }
    return out


def compute_current(meta: dict[str, tuple], profile: dict[str, Any], formula_hits: dict[str, dict[str, bool]]) -> list[dict[str, Any]]:
    mkt = duckdb.connect(str(MARKET_DB), read_only=True)
    _attach_smart_db(mkt)
    raw = mkt.execute(
        """
        WITH ranked AS (
            SELECT k.code, k.date, k.close, k.volume, k.amount,
                   ROW_NUMBER() OVER (PARTITION BY k.code ORDER BY k.date DESC) AS rn
            FROM v_price_kline_qfq k
            INNER JOIN sm.dim_active_a_stock s ON k.code = s.stock_code
        )
        SELECT code, date, close, volume, amount
        FROM ranked
        WHERE rn <= 220
        ORDER BY code, date
        """
    ).fetchnumpy()
    mkt.close()

    if len(raw["code"]) == 0:
        return []

    codes = raw["code"]
    dates = raw["date"]
    closes = raw["close"].astype(np.float64)
    volumes = raw["volume"].astype(np.float64)
    amounts = raw["amount"].astype(np.float64)

    unique_codes, counts = np.unique(codes, return_counts=True)
    fast = int(profile["macd_fast"])
    slow = int(profile["macd_slow"])
    sig = int(profile["macd_signal"])
    holding_days = int(profile["holding_days"])

    idx = 0
    results: list[dict[str, Any]] = []

    for code_raw, cnt in zip(unique_codes, counts):
        code = normalize_code(code_raw)
        sl = slice(idx, idx + cnt)
        date_arr = dates[sl]
        cls = closes[sl]
        vol = volumes[sl]
        amt = amounts[sl]
        n = len(cls)

        if n < slow + sig + 2:
            idx += cnt
            continue

        dif = ema_np(cls, fast) - ema_np(cls, slow)
        dea = ema_np(dif, sig)

        status, days_ev, gap = current_status(dif, dea, cls)

        amt_ma20 = sma_np(amt, 20)
        max60_arr = rolling_max_np(cls, 60)
        cur_amt_r20 = float(amt[-1] / amt_ma20[-1]) if (amt_ma20[-1] > 0 and not np.isnan(amt_ma20[-1])) else 0.0
        cur_price60 = float(cls[-1] / max60_arr[-1]) if max60_arr[-1] > 0 else 1.0

        last_gc_date = None
        sell_hint = None
        for i in range(1, min(n, 60)):
            if dif[-i] > dea[-i] and (i + 1 <= n) and dif[-(i + 1)] <= dea[-(i + 1)]:
                last_gc_date = str(date_arr[-i])
                days_held = i - 1
                remain = holding_days - days_held
                if remain > 0:
                    sell_hint = f"建议再持 {remain} 天"
                elif remain == 0:
                    sell_hint = "今日为建议卖出日"
                else:
                    sell_hint = f"已超持股期 {abs(remain)} 天"
                break

        hits = formula_hits.get(code, {"f1_hit": False, "f3_hit": False, "f5_hit": False})
        if not _passes_formula_filter(hits, profile):
            idx += cnt
            continue
        meta_val = meta.get(code, ("", "未知", "未知", 0.0))

        results.append(
            {
                "code": code,
                "name": meta_val[0],
                "industry": meta_val[1],
                "archetype": meta_val[2],
                "holder_chg": float(meta_val[3]) if meta_val[3] else 0.0,
                "status": status,
                "status_order": STATUS_ORDER.get(status, 9),
                "status_color": STATUS_COLOR.get(status, "gray"),
                "days_event": days_ev,
                "gap": round(gap, 6),
                "cur_dif": round(float(dif[-1]), 6),
                "cur_dea": round(float(dea[-1]), 6),
                "cur_close": round(float(cls[-1]), 2),
                "cur_date": str(date_arr[-1]),
                "dif_positive": float(dif[-1]) > 0,
                "cur_amt_r20": round(cur_amt_r20, 2),
                "cur_price60": round(cur_price60, 3),
                "filter_pass": (cur_amt_r20 >= float(profile.get("amt_ratio_min", 1.0)))
                and (cur_price60 <= float(profile.get("price_pos_max", 1.0))),
                "last_gc_date": last_gc_date,
                "sell_hint": sell_hint,
                "f1_hit": hits.get("f1_hit", False),
                "f3_hit": hits.get("f3_hit", False),
                "f5_hit": hits.get("f5_hit", False),
                "formula_hit_count": int(hits.get("f1_hit", False))
                + int(hits.get("f3_hit", False))
                + int(hits.get("f5_hit", False)),
                "history_status": "pending",
            }
        )

        idx += cnt

    return results


# ---------------------------------------------------------------------------
# 图表数据
# ---------------------------------------------------------------------------
def get_chart_data(code: str, profile: dict[str, Any]) -> dict[str, Any]:
    mkt = duckdb.connect(str(MARKET_DB), read_only=True)
    raw = mkt.execute(
        """
        SELECT date, open, high, low, close, volume, amount
        FROM v_price_kline_qfq
        WHERE code = ?
        ORDER BY date DESC
        LIMIT 220
        """,
        [normalize_code(code)],
    ).fetchnumpy()
    mkt.close()

    if len(raw["date"]) == 0:
        return {}

    dates = raw["date"][::-1]
    opens = raw["open"][::-1].astype(np.float64)
    highs = raw["high"][::-1].astype(np.float64)
    lows = raw["low"][::-1].astype(np.float64)
    closes = raw["close"][::-1].astype(np.float64)
    volumes = raw["volume"][::-1].astype(np.float64)
    n = len(closes)

    fast = int(profile["macd_fast"])
    slow = int(profile["macd_slow"])
    sig = int(profile["macd_signal"])

    dif = ema_np(closes, fast) - ema_np(closes, slow)
    dea = ema_np(dif, sig)
    bar = (dif - dea) * 2

    crosses = []
    for i in range(1, n):
        if dif[i] > dea[i] and dif[i - 1] <= dea[i - 1]:
            crosses.append({"idx": i, "type": "golden", "date": str(dates[i]), "close": round(float(closes[i]), 2)})
        elif dif[i] < dea[i] and dif[i - 1] >= dea[i - 1]:
            crosses.append({"idx": i, "type": "death", "date": str(dates[i]), "close": round(float(closes[i]), 2)})

    start = max(0, n - 90)
    status, days_ev, gap = current_status(dif, dea, closes)

    return {
        "dates": [str(d) for d in dates[start:]],
        "open": [round(float(v), 2) for v in opens[start:]],
        "high": [round(float(v), 2) for v in highs[start:]],
        "low": [round(float(v), 2) for v in lows[start:]],
        "close": [round(float(v), 2) for v in closes[start:]],
        "volume": [round(float(v) / 100, 0) for v in volumes[start:]],
        "dif": [round(float(v), 6) for v in dif[start:]],
        "dea": [round(float(v), 6) for v in dea[start:]],
        "bar": [round(float(v), 6) for v in bar[start:]],
        "crosses": [
            {**c, "idx": c["idx"] - start}
            for c in crosses
            if c["idx"] >= start
        ],
        "status": status,
        "days_event": days_ev,
        "gap": round(float(gap), 6),
        "profile_id": profile["id"],
    }


# ---------------------------------------------------------------------------
# 引擎
# ---------------------------------------------------------------------------
class ComputeEngine:
    def __init__(self):
        self._lock = threading.Lock()
        self._ready = False
        self._started = False
        self._message = "等待启动"
        # Multi-profile cache: each profile's computed result stored separately
        self._data_cache: dict[str, dict[str, Any]] = {}
        # Track which profiles are currently being computed
        self._computing: set[str] = set()

        self._profiles = get_strategy_profiles()
        start_profile = os.environ.get("BESTCHOICE_START_PROFILE")
        self._active_profile_id = (
            start_profile if start_profile in self._profiles else get_default_profile_id(self._profiles)
        )

    def profiles(self) -> dict[str, dict[str, Any]]:
        return self._profiles

    def active_profile_id(self) -> str:
        return self._active_profile_id

    def active_profile(self) -> dict[str, Any]:
        return self._profiles[self._active_profile_id]

    def set_profile(self, profile_id: str) -> dict[str, Any]:
        if profile_id not in self._profiles:
            raise KeyError(profile_id)

        with self._lock:
            self._active_profile_id = profile_id
            already_done = profile_id in self._data_cache
            already_computing = profile_id in self._computing

        if already_done:
            # Serve from cache instantly — no recompute needed
            with self._lock:
                self._ready = True
                self._message = f"就绪（{self._profiles[profile_id]['name']}）"
        else:
            # Not in cache yet (either computing or not started at all)
            with self._lock:
                self._ready = False
                self._message = f"计算策略（{self._profiles[profile_id]['name']}）..."
            if not already_computing:
                threading.Thread(target=self.start, args=(profile_id,), kwargs={"force": True}, daemon=True).start()

        return self._profiles[profile_id]

    def _build_profile_payload(self, profile: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": profile["id"],
            "name": profile["name"],
            "source": profile.get("source", "内置"),
            "macd": f"EMA({profile['macd_fast']}/{profile['macd_slow']}/{profile['macd_signal']})",
            "holding_days": profile["holding_days"],
            "amt_ratio_min": profile["amt_ratio_min"],
            "price_pos_max": profile["price_pos_max"],
            "min_signals": profile.get("min_signals", 1),
            "fast": profile["macd_fast"],
            "slow": profile["macd_slow"],
            "signal": profile["macd_signal"],
            "strategy_type": profile.get("strategy_type", "macd"),
            "formula_filter_mode": profile.get("formula_filter_mode"),
            "formula_rule_id": profile.get("formula_rule_id"),
        }

    def _build_param_desc_payload(self, profile: dict[str, Any]) -> dict[str, Any]:
        out = {
            "macd_fast": {
                **PARAM_DESCRIPTIONS["macd_fast"],
                "value": profile["macd_fast"],
            },
            "macd_slow": {
                **PARAM_DESCRIPTIONS["macd_slow"],
                "value": profile["macd_slow"],
            },
            "macd_signal": {
                **PARAM_DESCRIPTIONS["macd_signal"],
                "value": profile["macd_signal"],
            },
            "holding_days": {
                **PARAM_DESCRIPTIONS["holding_days"],
                "value": profile["holding_days"],
            },
            "amt_ratio_min": {
                **PARAM_DESCRIPTIONS["amt_ratio_min"],
                "value": profile["amt_ratio_min"],
            },
            "price_pos_max": {
                **PARAM_DESCRIPTIONS["price_pos_max"],
                "value": profile["price_pos_max"],
            },
            "min_signals": {
                "label": "历史信号下限",
                "desc": "至少保留多少个历史有效金叉才作为有效样本。",
                "low_hint": "调低更容易拿到结果，适合先做候选广度分析。",
                "high_hint": "调高可增强历史样本稳定性，但会出现更多空缺。",
                "value": int(profile.get("min_signals", 1)),
            },
        }

        rule_id = str(profile.get("formula_rule_id", ""))
        rule_meta = FORMULA_PROFILE_RULES.get(rule_id)
        if rule_meta:
            out.update(
                {
                    "formula_rule": {
                        "label": rule_meta.get("name", "公式规则"),
                        "desc": rule_meta.get("desc", ""),
                        "low_hint": rule_meta.get("low_hint", ""),
                        "high_hint": rule_meta.get("high_hint", ""),
                        "value": rule_meta.get("rule", ""),
                    },
                    "formula_rule_raw": {
                        "label": "命中模式",
                        "desc": "命中逻辑与筛选组合方式，可理解为策略对样本保守程度的控制。",
                        "low_hint": "放宽模式（任意命中）可显著扩大样本。",
                        "high_hint": "收紧模式（单公式/交集）可减少噪音。",
                        "value": profile.get("formula_filter_mode", "none"),
                    },
                }
            )

        if profile.get("strategy_type") == "macd_optuna":
            score = profile.get("optuna_score")
            out["optuna_n"] = {
                "label": "Optuna 样本量",
                "desc": "在候选参数下满足过滤条件的总信号样本。",
                "low_hint": "样本更少时，排名更容易受极端行情影响。",
                "high_hint": "样本更多通常更稳健，但可能包含更多弱信号。",
                "value": int(profile.get("optuna_n", 0)),
            }
            if score is not None:
                out["optuna_score"] = {
                    "label": "Optuna 目标分数",
                    "desc": "按 Calmar × 胜率定义的综合得分。",
                    "low_hint": "分数高并不代表样本无偏，但整体更均衡。",
                    "high_hint": "分数更高可作为更优买入与持仓参数的参考。",
                    "value": score,
                }

            if "best_calmar" in profile:
                out["best_calmar"] = {
                    "label": "最佳全局中位 Calmar",
                    "desc": "通达信参数扫描选出的全市场中位 Calmar。",
                    "low_hint": "值较低时，持仓期可能更偏快进快出。",
                    "high_hint": "值较高时，持仓窗口更有历史有效性。",
                    "value": profile["best_calmar"],
                }

        return out

    def warmup_all(self) -> None:
        """After default profile is ready, pre-warm all other profiles in the background."""
        for pid in self._profiles:
            if pid != self._active_profile_id:
                with self._lock:
                    already = pid in self._data_cache or pid in self._computing
                if not already:
                    threading.Thread(target=self.start, args=(pid,), kwargs={"force": False}, daemon=True).start()

    def start(self, profile_id: str | None = None, force: bool = False, clear_cache: bool = False) -> None:
        pid = profile_id or self._active_profile_id

        with self._lock:
            if pid not in self._profiles:
                raise KeyError(pid)
            # Prevent duplicate computation
            if pid in self._computing:
                return
            if pid in self._data_cache and not force:
                # Already computed; if this is now the active profile, mark ready
                if pid == self._active_profile_id:
                    self._ready = True
                    self._message = f"就绪（{self._profiles[pid]['name']}）"
                return
            self._computing.add(pid)
            self._started = True  # legacy flag

        profile = self._profiles[pid]
        is_active = pid == self._active_profile_id
        if is_active:
            self._ready = False
            self._message = f"准备计算（{profile['name']}）"
        t0 = time.time()

        if clear_cache:
            cache_pid_for_hist = profile.get("hist_cache_profile_id", pid)
            cache_file = _safe_cache_path(cache_pid_for_hist)
            if cache_file.exists():
                cache_file.unlink()

        def _msg(m: str) -> None:
            """Only update status message when computing the active profile."""
            if pid == self._active_profile_id:
                self._message = m

        try:
            _msg("读取元数据...")
            mkt = duckdb.connect(str(MARKET_DB), read_only=True)
            _attach_smart_db(mkt)
            meta_rows = mkt.execute(
                """
                SELECT s.stock_code, s.stock_name,
                       COALESCE(a.tdx_l1_name, '未知') AS industry,
                       COALESCE(a.stock_archetype, '未知') AS archetype,
                       COALESCE(f.holder_count_change_pct, 0.0) AS holder_chg_pct
                FROM sm.dim_active_a_stock s
                LEFT JOIN sm.dim_stock_archetype_latest a ON s.stock_code = a.stock_code
                LEFT JOIN sm.dim_financial_latest f ON s.stock_code = f.stock_code
                """
            ).fetchall()
            mkt.close()

            meta = {normalize_code(code): tuple(row) for code, *row in meta_rows}

            _msg("加载选股公式命中字段...")
            formula_hits = _load_formula_hits()

            _msg("计算历史回测指标...")

            def prog(done, total):
                _msg(f"历史回测 {done}/{total} ({done*100//max(total,1)}%)")

            hist = compute_historical(profile, progress_cb=prog)

            _msg("计算当前 MACD 状态...")
            current = compute_current(meta, profile, formula_hits)

            _msg("合并历史与当前数据...")
            for row in current:
                h = hist.get(row["code"])
                if h and h.get("history_status") == "ok":
                    row.update(h)
                    row["has_history"] = True
                else:
                    row["has_history"] = False
                    row["signal_count"] = 0
                    row["win_rate"] = None
                    row["avg_ret"] = None
                    row["avg_dd"] = None
                    row["calmar"] = None
                if h:
                    row["history_status"]   = h.get("history_status") or "pending"
                    row["horizons"]         = h.get("horizons") or {}
                    row["best_holding_days"] = h.get("best_holding_days")
                else:
                    row["history_status"]   = row.get("history_status") or "none"
                    row["horizons"]         = {}
                    row["best_holding_days"] = None

                # Compute buy-point signal and composite score after merging hist data
                is_just  = row["status"] == S_JUST
                days_ev  = row.get("days_event") or 99
                fp       = row.get("filter_pass", False)
                wr       = row.get("win_rate") or 0.0
                cal      = row.get("calmar")   or 0.0
                has_hist = row["has_history"]

                # is_buy_point: 刚金叉 within 3 days + good history
                # filter_pass is shown as context but NOT required (too strict as a hard gate)
                row["is_buy_point"] = bool(
                    is_just and days_ev <= 3 and has_hist and wr >= 0.48 and cal >= 0.5
                )

                # buy_score: 0-100 composite for ranking today's picks
                score = 0.0
                if is_just:
                    score += 40 if days_ev <= 1 else (30 if days_ev <= 2 else 18)
                elif row["status"] == S_IMMIN:
                    score += 8
                if fp:
                    score += 15   # bonus for passing volume/position filter
                if has_hist:
                    score += min(wr * 25, 25)
                    score += min(cal * 4, 10)
                    if (row.get("avg_ret") or 0) > 0:
                        score += 5
                    if (row.get("avg_dd") or -1) > -0.08:
                        score += 5
                row["buy_score"] = round(score, 1)

            current.sort(
                key=lambda x: (
                    x["status_order"],
                    -(x["calmar"] if x["calmar"] is not None else -999),
                    x["code"],
                )
            )

            industries = sorted({r["industry"] for r in current if r["industry"] != "未知"})
            archetypes = sorted({r["archetype"] for r in current if r["archetype"] != "未知"})

            summary = {
                "total": len(current),
                "just_cross": sum(1 for r in current if r["status"] == S_JUST),
                "imminent": sum(1 for r in current if r["status"] == S_IMMIN),
                "holding": sum(1 for r in current if r["status"] == S_HOLD),
                "death": sum(1 for r in current if r["status"] == S_DEATH),
                "waiting": sum(1 for r in current if r["status"] == S_WAIT),
                "with_history": sum(1 for r in current if r["has_history"]),
                "today_picks": sum(1 for r in current if r.get("is_buy_point")),
                "f1_hits": sum(1 for r in current if r["f1_hit"]),
                "f3_hits": sum(1 for r in current if r["f3_hit"]),
                "f5_hits": sum(1 for r in current if r["f5_hit"]),
                "elapsed": round(time.time() - t0, 1),
            }

            data = {
                "stocks": current,
                "summary": summary,
                "industries": industries,
                "archetypes": archetypes,
                "params": {
                    "macd": f"EMA({profile['macd_fast']}/{profile['macd_slow']}/{profile['macd_signal']})",
                    "holding_days": profile["holding_days"],
                    "amt_min": profile["amt_ratio_min"],
                    "price_max": profile["price_pos_max"],
                    "min_signals": profile.get("min_signals", 1),
                    "source": profile.get("source", "内置"),
                },
                "param_descriptions": self._build_param_desc_payload(profile),
                "profile": self._build_profile_payload(profile),
                "profile_id": profile["id"],
                "computed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "latest_data_date": get_latest_data_date(),
            }

            with self._lock:
                self._data_cache[pid] = data
                self._computing.discard(pid)
                self._started = False
                if pid == self._active_profile_id:
                    self._ready = True
                    self._message = f"就绪（{profile['name']}）耗时 {summary['elapsed']} 秒"

            # After the default profile finishes, warm up all others in background
            if (
                pid == get_default_profile_id(self._profiles)
                and os.environ.get("BESTCHOICE_SKIP_WARMUP") != "1"
            ):
                self.warmup_all()

        except Exception as e:  # pragma: no cover
            self._message = f"计算出错: {e}"
            with self._lock:
                self._ready = False
                self._computing.discard(pid)
                self._started = False
            raise

    def restart(self, profile_id: str | None = None, clear_cache: bool = False) -> None:
        pid = profile_id or self._active_profile_id
        if pid not in self._profiles:
            raise KeyError(pid)

        self._active_profile_id = pid
        # Evict stale in-memory cache for this profile so it recomputes
        with self._lock:
            self._data_cache.pop(pid, None)
            self._ready = False
        threading.Thread(target=self.start, args=(pid,), kwargs={"force": True, "clear_cache": clear_cache}, daemon=True).start()

    def status(self) -> dict[str, Any]:
        with self._lock:
            computing = list(self._computing)
        return {
            "ready": self._ready,
            "message": self._message,
            "active_profile_id": self._active_profile_id,
            "computing_profiles": computing,
        }

    def data(self) -> Optional[dict[str, Any]]:
        return self._data_cache.get(self._active_profile_id)
