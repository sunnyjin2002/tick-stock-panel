"""筹码分布全市场计算与落盘流水线。

数据流: 读 enriched(前复权 OHLC + 换手率) → 逐只递推筹码分布 → 写
``data/chip_distribution/all.parquet``(每只一行, 含 price_grid + distribution + 因子)。

发布复用 ``EnrichedPublication`` 的 generation 机制(独立 asset_type="chip"),
读者通过 ``get_enriched_generation(data_dir, "chip")`` 感知落盘更新; 写入原子替换。

两种模式:
  - 全量 ``compute_chip_table_full``: 读全部 enriched, 全市场重算。用于首次同步、
    往前扩展历史、数据修正, 以及除权/网格越界等慢路径的兜底。
  - 增量 ``compute_chip_table_incremental``: 对除权受影响/新增/越界的 symbol 走
    全量, 其余 symbol 用 advance_chip_distribution 逐根新 K 线向前推进。
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl

from app.config import settings
from app.enriched_generation import EnrichedPublication, enriched_publication_incomplete
from app.indicators.chip import (
    DEFAULT_BINS,
    DEFAULT_DECAY,
    DEFAULT_LOW_WINDOW,
    DEFAULT_MAX_LOOKBACK,
    DEFAULT_MIN_CUM_TURNOVER,
    DEFAULT_PADDING,
    DEFAULT_SMOOTH_WIDTH,
    advance_chip_distribution,
    compute_chip_distribution,
    extract_factors,
)
from app.parquet import scan_enriched_parquet

logger = logging.getLogger(__name__)

CHIP_ASSET_TYPE = "chip"
CHIP_DIRNAME = "chip_distribution"
CHIP_FILENAME = "all.parquet"

# 计算筹码所需的 enriched 列（前复权价 + 百分数换手率）
_ENRICHED_COLS = ["symbol", "date", "high", "low", "close", "turnover_rate"]

_CHIP_SCHEMA: dict[str, pl.DataType] = {
    "symbol": pl.Utf8,
    "date": pl.Date,
    "price_grid": pl.List(pl.Float64),
    "distribution": pl.List(pl.Float64),
    "profit_ratio": pl.Float64,
    "concentration_90": pl.Float64,
    "avg_cost_deviation": pl.Float64,
    "single_peak_ratio": pl.Float64,
    "single_peak_width": pl.Float64,
    "is_low_position": pl.Boolean,
    "avg_cost": pl.Float64,
    "current_price": pl.Float64,
}

# 筛选/列表查询用到的因子列（不含分布数组，减小批量响应体积）
CHIP_FACTOR_COLS = [
    "symbol",
    "date",
    "profit_ratio",
    "concentration_90",
    "avg_cost_deviation",
    "single_peak_ratio",
    "single_peak_width",
    "is_low_position",
    "avg_cost",
    "current_price",
]


def _empty_chip_table() -> pl.DataFrame:
    return pl.DataFrame({name: pl.Series([], dtype=dtype) for name, dtype in _CHIP_SCHEMA.items()})


def build_chip_table(
    enriched: pl.DataFrame,
    *,
    decay: float = DEFAULT_DECAY,
    bins: int = DEFAULT_BINS,
    padding: float = DEFAULT_PADDING,
    min_cum_turnover: float = DEFAULT_MIN_CUM_TURNOVER,
    max_lookback: int = DEFAULT_MAX_LOOKBACK,
    low_window: int = DEFAULT_LOW_WINDOW,
    smooth_width: int = DEFAULT_SMOOTH_WIDTH,
) -> pl.DataFrame:
    """全量计算：长表 enriched → 宽表 chip（每 symbol 一行）。

    enriched 需含 symbol/date/high/low/close/turnover_rate（前复权 + 百分数换手）。
    """
    if enriched.is_empty():
        return _empty_chip_table()

    df = enriched.select(_ENRICHED_COLS).sort(["symbol", "date"])

    symbols: list[str] = []
    dates: list = []
    grids: list[list[float]] = []
    dists: list[list[float]] = []
    profit_ratio: list[float] = []
    concentration_90: list[float] = []
    avg_cost_deviation: list[float] = []
    single_peak_ratio: list[float] = []
    single_peak_width: list[float] = []
    is_low_position: list[bool] = []
    avg_cost: list[float] = []
    current_prices: list[float] = []

    for _key, group in df.group_by("symbol", maintain_order=True):
        symbol = group["symbol"][0]  # 取标量字符串，规避 polars>=1.3x 迭代 key 变为 tuple
        high = group["high"].to_numpy()
        low = group["low"].to_numpy()
        close = group["close"].to_numpy()
        turnover = group["turnover_rate"].to_numpy()

        dist = compute_chip_distribution(
            high, low, close, turnover,
            decay=decay, bins=bins, padding=padding,
            min_cum_turnover=min_cum_turnover, max_lookback=max_lookback,
        )
        factors = extract_factors(
            dist.price_grid, dist.chip_dist,
            current_price=float(close[-1]), hist_close=close,
            low_window=low_window, smooth_width=smooth_width,
        )

        symbols.append(symbol)
        dates.append(group["date"][-1])
        grids.append(dist.price_grid.tolist())
        dists.append(dist.chip_dist.tolist())
        profit_ratio.append(factors.profit_ratio)
        concentration_90.append(factors.concentration_90)
        avg_cost_deviation.append(factors.avg_cost_deviation)
        single_peak_ratio.append(factors.single_peak_ratio)
        single_peak_width.append(factors.single_peak_width)
        is_low_position.append(factors.is_low_position)
        avg_cost.append(factors.avg_cost)
        current_prices.append(float(close[-1]))

    return pl.DataFrame({
        "symbol": pl.Series(symbols, dtype=pl.Utf8),
        "date": pl.Series(dates, dtype=pl.Date),
        "price_grid": pl.Series(grids, dtype=pl.List(pl.Float64)),
        "distribution": pl.Series(dists, dtype=pl.List(pl.Float64)),
        "profit_ratio": pl.Series(profit_ratio, dtype=pl.Float64),
        "concentration_90": pl.Series(concentration_90, dtype=pl.Float64),
        "avg_cost_deviation": pl.Series(avg_cost_deviation, dtype=pl.Float64),
        "single_peak_ratio": pl.Series(single_peak_ratio, dtype=pl.Float64),
        "single_peak_width": pl.Series(single_peak_width, dtype=pl.Float64),
        "is_low_position": pl.Series(is_low_position, dtype=pl.Boolean),
        "avg_cost": pl.Series(avg_cost, dtype=pl.Float64),
        "current_price": pl.Series(current_prices, dtype=pl.Float64),
    })


def advance_chip_table(
    existing: pl.DataFrame,
    enriched: pl.DataFrame,
    *,
    decay: float = DEFAULT_DECAY,
    low_window: int = DEFAULT_LOW_WINDOW,
    smooth_width: int = DEFAULT_SMOOTH_WIDTH,
) -> tuple[pl.DataFrame, list[str]]:
    """把已有筹码表用 enriched 新 K 线向前推进。

    - existing：已落盘的 chip 表（每 symbol 一行）。
    - enriched：**全量** enriched 长表（新 bar 用于推进，全量 close 历史用于低位判断）。

    返回 (更新后的行, 需要全量重算的 symbol 列表)。enriched 中存在但 existing 中
    没有的 symbol（首次出现）不在本函数处理，由调用方走全量。推进过程中价格越出
    网格的 symbol 进入重算列表。
    """
    if existing.is_empty():
        return existing, []

    enriched_by_symbol: dict[str, pl.DataFrame] = {}
    for _key, group in enriched.group_by("symbol", maintain_order=True):
        enriched_by_symbol[group["symbol"][0]] = group.sort("date")

    rows: list[dict] = []
    recompute: list[str] = []

    for row in existing.iter_rows(named=True):
        symbol = row["symbol"]
        group = enriched_by_symbol.get(symbol)
        grid = np.asarray(row["price_grid"], dtype=np.float64)
        chip = np.asarray(row["distribution"], dtype=np.float64)
        last_date = row["date"]

        # 无该 symbol 数据，或无新 bar → 原样保留
        if group is None or group.is_empty():
            rows.append(row)
            continue
        new_bars = group.filter(pl.col("date") > last_date) if last_date is not None else group
        if new_bars.is_empty():
            rows.append(row)
            continue

        failed = False
        for h, lo, c, t in zip(
            new_bars["high"].to_numpy(),
            new_bars["low"].to_numpy(),
            new_bars["close"].to_numpy(),
            new_bars["turnover_rate"].to_numpy(),
            strict=True,
        ):
            out = advance_chip_distribution(grid, chip, h, lo, c, t, decay=decay)
            if out is None:
                failed = True
                break
            chip = out
        if failed:
            recompute.append(symbol)
            continue

        full_close = group["close"].to_numpy()
        factors = extract_factors(
            grid, chip,
            current_price=float(full_close[-1]), hist_close=full_close,
            low_window=low_window, smooth_width=smooth_width,
        )

        new_row = dict(row)
        new_row["date"] = new_bars["date"][-1]
        new_row["distribution"] = chip.tolist()
        new_row["profit_ratio"] = factors.profit_ratio
        new_row["concentration_90"] = factors.concentration_90
        new_row["avg_cost_deviation"] = factors.avg_cost_deviation
        new_row["single_peak_ratio"] = factors.single_peak_ratio
        new_row["single_peak_width"] = factors.single_peak_width
        new_row["is_low_position"] = factors.is_low_position
        new_row["avg_cost"] = factors.avg_cost
        new_row["current_price"] = float(full_close[-1])
        rows.append(new_row)

    if not rows:
        return _empty_chip_table(), recompute
    return pl.DataFrame(rows).cast(_CHIP_SCHEMA), recompute

# --------------------------------------------------------------------------- #
# IO 与编排
# --------------------------------------------------------------------------- #


def _read_enriched(data_dir: Path) -> pl.DataFrame:
    enriched_base = data_dir / "kline_daily_enriched"
    if not enriched_base.exists() or not any(enriched_base.rglob("*.parquet")):
        return pl.DataFrame({
            "symbol": pl.Series([], dtype=pl.Utf8),
            "date": pl.Series([], dtype=pl.Date),
            "high": pl.Series([], dtype=pl.Float64),
            "low": pl.Series([], dtype=pl.Float64),
            "close": pl.Series([], dtype=pl.Float64),
            "turnover_rate": pl.Series([], dtype=pl.Float64),
        })
    glob = (enriched_base / "**" / "*.parquet").as_posix()
    return (
        scan_enriched_parquet(glob)
        .select(_ENRICHED_COLS)
        .sort(["symbol", "date"])
        .collect(engine="streaming")
    )


def _read_chip_table(data_dir: Path) -> pl.DataFrame:
    path = data_dir / CHIP_DIRNAME / CHIP_FILENAME
    if not path.exists():
        return _empty_chip_table()
    try:
        return pl.read_parquet(path)
    except Exception as exc:
        logger.warning("chip 表读取失败,按空表处理: %s", exc)
        return _empty_chip_table()


def _write_chip_table(chip: pl.DataFrame, data_dir: Path) -> int:
    out = data_dir / CHIP_DIRNAME / CHIP_FILENAME
    publication = EnrichedPublication(data_dir, CHIP_ASSET_TYPE, recover=True)
    publication.begin()
    try:
        publication.write_parquet(chip, out)
        publication.commit()
    except Exception:
        publication.abandon()
        raise
    return chip.height


def compute_chip_table_full(
    data_dir: Path | None = None,
    *,
    decay: float = DEFAULT_DECAY,
    bins: int = DEFAULT_BINS,
    padding: float = DEFAULT_PADDING,
    min_cum_turnover: float = DEFAULT_MIN_CUM_TURNOVER,
    max_lookback: int = DEFAULT_MAX_LOOKBACK,
    low_window: int = DEFAULT_LOW_WINDOW,
    smooth_width: int = DEFAULT_SMOOTH_WIDTH,
) -> int:
    """读 enriched 全量重算全市场筹码，原子写 chip 表。返回写入行数。"""
    d = Path(data_dir or settings.data_dir)
    enriched = _read_enriched(d)
    if enriched.is_empty():
        logger.info("chip: 无 enriched 数据,跳过")
        return 0
    chip = build_chip_table(
        enriched, decay=decay, bins=bins, padding=padding,
        min_cum_turnover=min_cum_turnover, max_lookback=max_lookback,
        low_window=low_window, smooth_width=smooth_width,
    )
    return _write_chip_table(chip, d)


def compute_chip_table_incremental(
    data_dir: Path | None = None,
    affected_symbols: set[str] | None = None,
    *,
    decay: float = DEFAULT_DECAY,
    bins: int = DEFAULT_BINS,
    padding: float = DEFAULT_PADDING,
    min_cum_turnover: float = DEFAULT_MIN_CUM_TURNOVER,
    max_lookback: int = DEFAULT_MAX_LOOKBACK,
    low_window: int = DEFAULT_LOW_WINDOW,
    smooth_width: int = DEFAULT_SMOOTH_WIDTH,
) -> int:
    """增量更新全市场筹码：受影响/新增/越界 symbol 全量，其余逐根推进。返回行数。"""
    d = Path(data_dir or settings.data_dir)
    enriched = _read_enriched(d)
    if enriched.is_empty():
        logger.info("chip: 无 enriched 数据,跳过")
        return 0

    # 上次 chip 发布未完成 → 直接全量重建
    if enriched_publication_incomplete(d, CHIP_ASSET_TYPE):
        logger.warning("chip: 检测到未完成的 chip 发布,改为全量重建")
        return compute_chip_table_full(d, decay=decay, bins=bins, padding=padding,
                                       min_cum_turnover=min_cum_turnover,
                                       max_lookback=max_lookback,
                                       low_window=low_window, smooth_width=smooth_width)

    existing = _read_chip_table(d)
    if existing.is_empty():
        return compute_chip_table_full(d, decay=decay, bins=bins, padding=padding,
                                       min_cum_turnover=min_cum_turnover,
                                       max_lookback=max_lookback,
                                       low_window=low_window, smooth_width=smooth_width)

    affected = set(affected_symbols or ())
    existing_symbols = set(existing["symbol"].to_list())
    new_symbols = set(enriched["symbol"].unique().to_list()) - existing_symbols

    advanced, grid_fail = advance_chip_table(
        existing, enriched, decay=decay,
        low_window=low_window, smooth_width=smooth_width,
    )

    full_set = affected | new_symbols | set(grid_fail)

    full_chip = _empty_chip_table()
    if full_set:
        full_chip = build_chip_table(
            enriched.filter(pl.col("symbol").is_in(list(full_set))),
            decay=decay, bins=bins, padding=padding,
            min_cum_turnover=min_cum_turnover, max_lookback=max_lookback,
            low_window=low_window, smooth_width=smooth_width,
        )

    if advanced.is_empty():
        merged = full_chip
    else:
        advanced_kept = (
            advanced.filter(~pl.col("symbol").is_in(list(full_set)))
            if full_set else advanced
        )
        merged = (
            pl.concat([advanced_kept, full_chip], how="diagonal_relaxed")
            if not full_chip.is_empty() else advanced_kept
        )

    if merged.is_empty():
        return 0
    return _write_chip_table(merged, d)


# --------------------------------------------------------------------------- #
# 查询
# --------------------------------------------------------------------------- #


def load_chip_table(data_dir: Path | None = None) -> pl.DataFrame:
    """读取已落盘的筹码表；不存在或损坏时返回空表（schema 一致）。"""
    d = Path(data_dir or settings.data_dir)
    return _read_chip_table(d)


def get_chip_symbol(symbol: str, data_dir: Path | None = None) -> dict | None:
    """查询单只股票的筹码分布与因子；不存在返回 None。"""
    table = load_chip_table(data_dir)
    if table.is_empty():
        return None
    row = table.filter(pl.col("symbol") == symbol)
    if row.is_empty():
        return None
    data = row.to_dicts()[0]
    d = data.get("date")
    if d is not None:
        data["date"] = d.isoformat() if hasattr(d, "isoformat") else str(d)
    return data
