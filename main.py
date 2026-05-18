"""
main.py — MACD 金叉选股 FastAPI 服务
"""
from __future__ import annotations

import subprocess
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from compute import ComputeEngine, get_chart_data, SCRIPTS_DIR, get_latest_data_date

engine = ComputeEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=engine.start, daemon=True).start()
    yield


app = FastAPI(title="MACD 金叉选股", lifespan=lifespan)

HTML_FILE = Path(__file__).parent / "index.html"


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML_FILE.read_text(encoding="utf-8")


@app.get("/api/status")
async def api_status(strategy: Optional[str] = None):
    try:
        s = engine.status(strategy)
    except KeyError:
        raise HTTPException(404, f"未找到策略: {strategy}")
    s["latest_data_date"] = get_latest_data_date()
    return s


@app.get("/api/strategies")
async def api_strategies():
    return {
        "profiles": engine.profiles(),
        "active_profile_id": engine.active_profile_id(),
        "default_profile_id": engine.default_profile_id(),
    }


@app.get("/api/data")
async def api_data(strategy: Optional[str] = None):
    if strategy:
        try:
            engine.ensure_profile(strategy)
        except KeyError:
            raise HTTPException(404, f"未找到策略: {strategy}")

    d = engine.data_for_profile(strategy)
    if d is None:
        raise HTTPException(503, "数据尚未就绪，请稍候")
    return JSONResponse(d)


@app.get("/api/ready/{strategy}")
async def api_ready(strategy: str):
    """Check if a specific strategy's data is ready (cached in memory)."""
    profiles = engine.profiles()
    if strategy not in profiles:
        raise HTTPException(404, f"未找到策略: {strategy}")
    d = engine.data_for_profile(strategy)
    return {"ready": d is not None, "strategy": strategy}


@app.get("/api/chart/{code}")
async def api_chart(code: str, strategy: Optional[str] = None):
    try:
        profile = engine.active_profile() if strategy is None else engine.profiles()[strategy]
    except KeyError:
        raise HTTPException(404, f"未找到策略: {strategy}")

    return JSONResponse(get_chart_data(code, profile))


@app.post("/api/refresh")
async def api_refresh(strategy: Optional[str] = None):
    try:
        engine.restart(strategy, clear_cache=True, activate=(strategy is None))
    except KeyError:
        raise HTTPException(404, f"未找到策略: {strategy}")
    return {"ok": True, "message": "已触发重新计算"}


_optimize_lock = threading.Lock()
_optimize_running = False


@app.post("/api/optimize")
async def api_optimize(job: str = "optuna"):
    """
    后台重新运行优化脚本，完成后自动更新 optuna_best 策略。
    job=optuna  → 重跑 Optuna 参数搜索（~15 min）
    job=gcross  → 重跑 MACD 金叉持股期回测（生成持股期汇总 CSV）
    """
    global _optimize_running
    scripts = {
        "optuna": SCRIPTS_DIR / "macd_optuna_backtest.py",
        "gcross": SCRIPTS_DIR / "macd_golden_cross_backtest.py",
    }
    script = scripts.get(job)
    if not script:
        raise HTTPException(400, f"未知任务: {job}，可选: optuna / gcross")
    if not script.exists():
        raise HTTPException(404, f"脚本不存在: {script}")

    with _optimize_lock:
        if _optimize_running:
            raise HTTPException(409, "已有优化任务在运行，请稍候")
        _optimize_running = True

    def _run():
        global _optimize_running
        try:
            subprocess.run([sys.executable, str(script)], check=False,
                           cwd=str(SCRIPTS_DIR))
            # 优化完成后自动刷新 optuna_best 策略
            if job == "optuna":
                engine.restart("optuna_best", clear_cache=True)
        finally:
            _optimize_running = False

    threading.Thread(target=_run, daemon=True).start()
    return {"ok": True, "message": f"已启动 {job} 优化任务（后台运行），完成后自动更新数据"}
