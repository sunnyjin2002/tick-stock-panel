"""筹码分布查询 API 测试。"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.chip import router


class _Repo:
    def __init__(self, data_dir: Path) -> None:
        self.store = SimpleNamespace(data_dir=data_dir)


def _write_chip_parquet(data_dir: Path) -> None:
    chip_dir = data_dir / "chip_distribution"
    chip_dir.mkdir(parents=True, exist_ok=True)
    grid = [10.0 + i * 0.1 for i in range(100)]
    dist = [0.01] * 100
    pl.DataFrame({
        "symbol": pl.Series(["600000.SH", "000001.SZ"], dtype=pl.Utf8),
        "date": pl.Series([date(2026, 9, 1), date(2026, 9, 1)], dtype=pl.Date),
        "price_grid": pl.Series([grid, grid], dtype=pl.List(pl.Float64)),
        "distribution": pl.Series([dist, dist], dtype=pl.List(pl.Float64)),
        "profit_ratio": pl.Series([0.6, 0.4], dtype=pl.Float64),
        "concentration_90": pl.Series([0.08, 0.12], dtype=pl.Float64),
        "avg_cost_deviation": pl.Series([0.05, -0.03], dtype=pl.Float64),
        "single_peak_ratio": pl.Series([0.7, 0.5], dtype=pl.Float64),
        "single_peak_width": pl.Series([0.1, 0.2], dtype=pl.Float64),
        "is_low_position": pl.Series([True, False], dtype=pl.Boolean),
        "avg_cost": pl.Series([10.5, 11.0], dtype=pl.Float64),
        "main_cost": pl.Series([10.4, 10.8], dtype=pl.Float64),
        "current_price": pl.Series([10.6, 10.9], dtype=pl.Float64),
    }).write_parquet(chip_dir / "all.parquet")


def _client(tmp_path: Path) -> TestClient:
    _write_chip_parquet(tmp_path)
    app = FastAPI()
    app.include_router(router)
    app.state.repo = _Repo(tmp_path)
    return TestClient(app)


def test_get_chip_returns_distribution(tmp_path):
    client = _client(tmp_path)
    resp = client.get("/api/chip/600000.SH")

    assert resp.status_code == 200
    data = resp.json()
    assert data["symbol"] == "600000.SH"
    assert len(data["price_grid"]) == 100
    assert len(data["distribution"]) == 100
    assert abs(sum(data["distribution"]) - 1.0) < 1e-9
    assert data["profit_ratio"] == pytest.approx(0.6)
    assert data["is_low_position"] is True
    assert data["current_price"] == pytest.approx(10.6)
    assert data["main_cost"] == pytest.approx(10.4)
    assert isinstance(data["date"], str)


def test_get_chip_404_unknown_symbol(tmp_path):
    client = _client(tmp_path)
    resp = client.get("/api/chip/999999.SZ")
    assert resp.status_code == 404


def test_get_chip_404_when_no_table(tmp_path):
    app = FastAPI()
    app.include_router(router)
    app.state.repo = _Repo(tmp_path)
    client = TestClient(app)

    resp = client.get("/api/chip/600000.SH")
    assert resp.status_code == 404


def test_list_factors_returns_all(tmp_path):
    client = _client(tmp_path)
    resp = client.get("/api/chip/factors")

    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2
    assert {i["symbol"] for i in data["items"]} == {"600000.SH", "000001.SZ"}
    # 批量接口不含分布数组（减小体积）
    assert "distribution" not in data["items"][0]
    assert "price_grid" not in data["items"][0]


def test_list_factors_filter_symbols(tmp_path):
    client = _client(tmp_path)

    resp = client.get("/api/chip/factors", params={"symbols": "600000.SH"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 1
    assert data["items"][0]["symbol"] == "600000.SH"


def test_list_factors_empty_table(tmp_path):
    app = FastAPI()
    app.include_router(router)
    app.state.repo = _Repo(tmp_path)
    client = TestClient(app)

    resp = client.get("/api/chip/factors")
    assert resp.status_code == 200
    assert resp.json() == {"count": 0, "items": []}
