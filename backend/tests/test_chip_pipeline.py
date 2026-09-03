"""筹码分布流水线（chip_pipeline.py）测试。"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from app.services.chip_pipeline import (
    advance_chip_table,
    build_chip_table,
    compute_chip_table_full,
    compute_chip_table_incremental,
)


def _frame(symbol: str, close, high, low, turnover, dates) -> pl.DataFrame:
    n = len(close)
    return pl.DataFrame({
        "symbol": [symbol] * n,
        "date": dates,
        "high": list(high),
        "low": list(low),
        "close": list(close),
        "turnover_rate": list(turnover),
    })


def _dates(n: int) -> list[date]:
    return [date(2026, 1, 1) + timedelta(days=i) for i in range(n)]


# --------------------------------------------------------------------------- #
# 全量 build_chip_table
# --------------------------------------------------------------------------- #


def test_build_chip_table_shape_and_schema():
    dates = _dates(30)
    frames = []
    for sym in ("000001.SZ", "600000.SH"):
        close = np.linspace(10.0, 11.0, 30)
        frames.append(_frame(sym, close, close * 1.01, close * 0.99, np.full(30, 10.0), dates))
    enriched = pl.concat(frames)

    chip = build_chip_table(enriched)

    assert chip.height == 2
    assert set(chip["symbol"].to_list()) == {"000001.SZ", "600000.SH"}
    assert chip.schema["price_grid"] == pl.List(pl.Float64)
    assert chip.schema["distribution"] == pl.List(pl.Float64)
    assert chip.schema["is_low_position"] == pl.Boolean
    for d in chip["distribution"].to_list():
        assert abs(sum(d) - 1.0) < 1e-9
    # 因子数值范围合理
    assert chip["profit_ratio"].min() >= 0.0
    assert chip["concentration_90"].min() >= 0.0


def test_build_chip_table_empty():
    empty = pl.DataFrame({
        "symbol": pl.Series([], dtype=pl.Utf8),
        "date": pl.Series([], dtype=pl.Date),
        "high": pl.Series([], dtype=pl.Float64),
        "low": pl.Series([], dtype=pl.Float64),
        "close": pl.Series([], dtype=pl.Float64),
        "turnover_rate": pl.Series([], dtype=pl.Float64),
    })
    chip = build_chip_table(empty)
    assert chip.height == 0
    assert "distribution" in chip.columns


# --------------------------------------------------------------------------- #
# 增量 advance_chip_table
# --------------------------------------------------------------------------- #


def test_advance_matches_full_build():
    n = 60
    dates = _dates(n)
    close = np.full(n, 10.0)
    high = np.full(n, 10.1)
    low = np.full(n, 9.9)
    turnover = np.full(n, 5.0)
    enriched = _frame("600000.SH", close, high, low, turnover, dates)

    # 固定窗口（min_cum_turnover 永不达标）→ 网格不变，advance 应严格等于全量
    kwargs = {"min_cum_turnover": 1e9, "max_lookback": 1000}
    existing = build_chip_table(enriched.slice(0, n - 1), **kwargs)
    full_n = build_chip_table(enriched, **kwargs)

    advanced, recompute = advance_chip_table(existing, enriched)

    assert recompute == []
    assert advanced.height == 1
    np.testing.assert_allclose(advanced["distribution"][0], full_n["distribution"][0], atol=1e-12)


def test_advance_grid_escape_triggers_recompute():
    n = 30
    dates = _dates(n)
    close = np.full(n, 10.0)
    high = np.full(n, 10.1)
    low = np.full(n, 9.9)
    turnover = np.full(n, 5.0)
    enriched = _frame("600000.SH", close, high, low, turnover, dates)
    existing = build_chip_table(enriched, min_cum_turnover=1e9, max_lookback=1000)

    # 新 bar 大幅越出网格 → 应进入重算列表
    new_bar = _frame(
        "600000.SH", [15.0], [15.2], [14.8], [5.0], [dates[-1] + timedelta(days=1)],
    )
    _, recompute = advance_chip_table(existing, pl.concat([enriched, new_bar]))

    assert recompute == ["600000.SH"]


def test_advance_no_new_bars_keeps_row():
    n = 30
    dates = _dates(n)
    close = np.full(n, 10.0)
    high = np.full(n, 10.1)
    low = np.full(n, 9.9)
    turnover = np.full(n, 5.0)
    enriched = _frame("600000.SH", close, high, low, turnover, dates)
    existing = build_chip_table(enriched, min_cum_turnover=1e9, max_lookback=1000)

    advanced, recompute = advance_chip_table(existing, enriched)

    assert recompute == []
    assert advanced.height == 1
    assert advanced["date"][0] == dates[-1]


# --------------------------------------------------------------------------- #
# IO 编排
# --------------------------------------------------------------------------- #


def test_compute_full_then_incremental_roundtrip(tmp_path):
    data_dir = tmp_path
    dates = _dates(30)
    for d in dates:
        part = data_dir / "kline_daily_enriched" / f"date={d.isoformat()}"
        part.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({
            "symbol": ["600000.SH"],
            "date": [d],
            "high": [10.1],
            "low": [9.9],
            "close": [10.0],
            "turnover_rate": [5.0],
        }).write_parquet(part / "part.parquet")

    n = compute_chip_table_full(data_dir)
    assert n == 1
    out = data_dir / "chip_distribution" / "all.parquet"
    assert out.exists()
    chip = pl.read_parquet(out)
    assert chip.height == 1
    assert chip.schema["distribution"] == pl.List(pl.Float64)

    # 增量: 无新数据 → 仍 1 行
    n2 = compute_chip_table_incremental(data_dir)
    assert n2 == 1
