"""筹码分布查询 API。

- ``GET /api/chip/factors``   批量筹码因子（不含分布数组，供筛选/列表）
- ``GET /api/chip/{symbol}``  单只股票的筹码分布 + 因子（供个股预览对话框画图）
"""
from __future__ import annotations

import polars as pl
from fastapi import APIRouter, HTTPException, Query, Request

from app.services import chip_pipeline

router = APIRouter(prefix="/api/chip", tags=["chip"])


@router.get("/factors")
def list_factors(
    request: Request,
    symbols: str | None = Query(None, description="逗号分隔的 symbol 列表; 为空返回全部"),
) -> dict:
    """批量筹码因子（不含分布数组，减小响应体积）。"""
    data_dir = request.app.state.repo.store.data_dir
    table = chip_pipeline.load_chip_table(data_dir)
    if symbols:
        sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
        if sym_list:
            table = table.filter(pl.col("symbol").is_in(sym_list))
    cols = [c for c in chip_pipeline.CHIP_FACTOR_COLS if c in table.columns]
    items = table.select(cols).to_dicts()
    for row in items:
        d = row.get("date")
        if d is not None:
            row["date"] = d.isoformat() if hasattr(d, "isoformat") else str(d)
    return {"count": len(items), "items": items}


@router.get("/{symbol}")
def get_chip(request: Request, symbol: str) -> dict:
    """单只股票的筹码分布与因子。symbol 不存在时返回 404。"""
    data_dir = request.app.state.repo.store.data_dir
    data = chip_pipeline.get_chip_symbol(symbol, data_dir)
    if data is None:
        raise HTTPException(status_code=404, detail=f"chip data not found: {symbol}")
    return data
