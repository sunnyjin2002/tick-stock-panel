"""筹码分布（CYQ / 移动成本分布）核心计算。

筹码分布是路径依赖的逐日递推（第 N 天的分布依赖第 N-1 天），无法用 Polars
表达式向量化，故采用「外层逐日 Python 循环 + 内层 NumPy 向量化」实现。

数据口径（与 enriched 表一致，见 CONTRIBUTING §3）：
- high / low / close 为**前复权价**；
- turnover_rate 为**百分数**（5.0 表示 5%），内部统一转为小数制；
- 递推只需换手率序列 + 前复权 H/L/C，**不需要流通股本**：所有输出因子均为
  比例，归一化基准可任意选取。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 默认参数
DEFAULT_BINS = 100
DEFAULT_DECAY = 1.0
DEFAULT_PADDING = 0.03          # 网格上下各留 3% 缓冲
DEFAULT_MIN_CUM_TURNOVER = 3.0  # 回溯窗口累计换手 ≥ 3 倍流通盘
DEFAULT_MAX_LOOKBACK = 2500     # 回溯窗口上限（交易日）
DEFAULT_LOW_WINDOW = 250        # is_low_position 的「近一年」窗口（交易日）
DEFAULT_SMOOTH_WIDTH = 3        # 单峰识别的平滑宽度

_GRID_MIN_FLOOR = 1e-6


@dataclass(frozen=True)
class ChipDistribution:
    """一只股票的筹码分布结果。"""

    price_grid: np.ndarray  # shape (bins,)，前复权价格刻度，升序
    chip_dist: np.ndarray   # shape (bins,)，归一化筹码分布，Σ=1


@dataclass(frozen=True)
class ChipFactors:
    """从筹码分布提取的数值因子。"""

    profit_ratio: float        # 获利盘比例 [0,1]
    concentration_90: float    # 90% 成本集中度（无量纲）
    avg_cost_deviation: float  # 平均成本偏离度（无量纲）
    single_peak_ratio: float   # 主峰筹码占比 [0,1]（简化版）
    single_peak_width: float   # 主峰价格跨度 / 现价（无量纲，简化版）
    is_low_position: bool      # 现价是否处于近一年低位
    avg_cost: float            # 平均成本（前复权价）
    main_cost: float           # 主力成本（筹码最密集主峰的加权均价，前复权价）


def compute_chip_distribution(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    turnover_rate: np.ndarray,
    *,
    decay: float = DEFAULT_DECAY,
    bins: int = DEFAULT_BINS,
    padding: float = DEFAULT_PADDING,
    min_cum_turnover: float = DEFAULT_MIN_CUM_TURNOVER,
    max_lookback: int = DEFAULT_MAX_LOOKBACK,
) -> ChipDistribution:
    """计算一只股票的筹码分布。

    参数为按日期升序排列的数组（可由 enriched 表 `.to_numpy()` 提取）：
    - high / low / close：前复权价；
    - turnover_rate：百分数换手率（5.0 = 5%）。

    内部流程：选定回溯窗口 → 构建固定价格网格 → 逐日递推 → 归一化。
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    # 换手率缺失（NaN）按 0 处理：当日既不衰减也不新增，fail-closed。
    turnover_rate = np.nan_to_num(
        np.asarray(turnover_rate, dtype=np.float64), nan=0.0,
    )

    if not (high.size == low.size == close.size == turnover_rate.size):
        raise ValueError("high/low/close/turnover_rate 长度必须一致")
    if high.size == 0:
        raise ValueError("输入数据为空")
    if bins < 2:
        raise ValueError("bins 必须 ≥ 2")

    start = _select_window_start(turnover_rate, min_cum_turnover, max_lookback)
    price_grid = _build_price_grid(low[start:], high[start:], bins, padding)
    chip = _recurse(
        high[start:], low[start:], close[start:], turnover_rate[start:],
        decay=decay, price_grid=price_grid,
    )
    return ChipDistribution(price_grid=price_grid, chip_dist=chip)


def compute_chip_history(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    turnover_rate: np.ndarray,
    dates: list,
    *,
    decay: float = DEFAULT_DECAY,
    bins: int = DEFAULT_BINS,
    padding: float = DEFAULT_PADDING,
    min_cum_turnover: float = DEFAULT_MIN_CUM_TURNOVER,
    max_lookback: int = DEFAULT_MAX_LOOKBACK,
    smooth_width: int = DEFAULT_SMOOTH_WIDTH,
) -> tuple[np.ndarray, list[dict]]:
    """返回 (price_grid, 每日筹码分布序列)，用于「悬浮 K 线看历史筹码峰」。

    - high / low / close：前复权价（按日期升序）；
    - turnover_rate：百分数换手率；
    - dates：与上述数组对齐的日期序列（date 或字符串）。

    序列每项为 {"date": ISO 日期字符串, "distribution": 归一化总筹码分布列表,
    "daily_new": 当日新增筹码分布列表}。回溯窗口起点之前的日期，两个分布均为全零
    （筹码尚未开始累积）。所有天共享同一个 price_grid。
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    turnover_rate = np.nan_to_num(np.asarray(turnover_rate, dtype=np.float64), nan=0.0)

    if not (high.size == low.size == close.size == turnover_rate.size):
        raise ValueError("high/low/close/turnover_rate 长度必须一致")
    if high.size == 0:
        raise ValueError("输入数据为空")
    if bins < 2:
        raise ValueError("bins 必须 ≥ 2")

    start = _select_window_start(turnover_rate, min_cum_turnover, max_lookback)
    price_grid = _build_price_grid(low[start:], high[start:], bins, padding)

    fraction = turnover_rate / 100.0
    chip = np.zeros(bins, dtype=np.float64)
    rows: list[dict] = []
    for i in range(high.size):
        if i < start:
            # 回溯窗口起点之前：无筹码累积
            zero = np.zeros(bins, dtype=np.float64)
            rows.append({
                "date": _date_to_str(dates[i]),
                "distribution": zero.tolist(),
                "daily_new": zero.tolist(),
                "avg_cost": float(close[i]),
                "main_cost": float(close[i]),
            })
            continue
        t_decay = min(max(float(fraction[i]) * decay, 0.0), 1.0)
        if t_decay <= 0.0:
            daily_new = np.zeros(bins, dtype=np.float64)
        else:
            daily_new = _triangle_new_chip(price_grid, high[i], low[i], close[i], t_decay)
            chip = chip * (1.0 - t_decay) + daily_new
            total = float(chip.sum())
            if total > 0.0:
                chip = chip / total
        chip_total = float(chip.sum())
        _avg_cost = float(np.average(price_grid, weights=chip)) if chip_total > 0.0 else float(close[i])
        _, _, _main_cost = _single_peak_measures(price_grid, chip, close[i], smooth_width)
        rows.append({
            "date": _date_to_str(dates[i]),
            "distribution": chip.tolist(),
            "daily_new": daily_new.tolist(),
            "avg_cost": _avg_cost,
            "main_cost": _main_cost,
        })
    return price_grid, rows


def _date_to_str(value) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def extract_factors(
    price_grid: np.ndarray,
    chip_dist: np.ndarray,
    current_price: float,
    hist_close: np.ndarray,
    *,
    low_window: int = DEFAULT_LOW_WINDOW,
    smooth_width: int = DEFAULT_SMOOTH_WIDTH,
) -> ChipFactors:
    """从筹码分布中提取因子。

    - price_grid / chip_dist：来自 compute_chip_distribution；
    - current_price：现价（前复权）；
    - hist_close：按日期升序的前复权收盘价序列（用于低位判断）。
    """
    grid = np.asarray(price_grid, dtype=np.float64)
    dist = np.asarray(chip_dist, dtype=np.float64)
    if grid.size == 0 or grid.size != dist.size:
        raise ValueError("price_grid 与 chip_dist 长度必须一致且非空")

    total = float(dist.sum())
    if total <= 0.0:
        return ChipFactors(0.0, 0.0, 0.0, 0.0, 0.0, False, float(current_price), float(current_price))

    profit_ratio = float(dist[grid <= current_price].sum() / total)

    cum = np.cumsum(dist)
    p5 = grid[min(int(np.searchsorted(cum, 0.05 * total)), grid.size - 1)]
    p95 = grid[min(int(np.searchsorted(cum, 0.95 * total)), grid.size - 1)]
    concentration_90 = float((p95 - p5) / (p95 + p5)) if (p95 + p5) > 0.0 else 0.0

    avg_cost = float(np.average(grid, weights=dist))
    avg_cost_deviation = (
        float((current_price - avg_cost) / avg_cost) if avg_cost > 0.0 else 0.0
    )

    single_peak_ratio, single_peak_width, main_cost = _single_peak_measures(
        grid, dist, current_price, smooth_width,
    )

    hist = np.asarray(hist_close, dtype=np.float64)
    window = hist[-low_window:] if hist.size > low_window else hist
    q30 = float(np.quantile(window, 0.30)) if window.size > 0 else float(current_price)
    is_low_position = bool(current_price < q30)

    return ChipFactors(
        profit_ratio=profit_ratio,
        concentration_90=concentration_90,
        avg_cost_deviation=avg_cost_deviation,
        single_peak_ratio=single_peak_ratio,
        single_peak_width=single_peak_width,
        is_low_position=is_low_position,
        avg_cost=avg_cost,
        main_cost=main_cost,
    )


def advance_chip_distribution(
    price_grid: np.ndarray,
    chip_dist: np.ndarray,
    high: float,
    low: float,
    close: float,
    turnover_rate: float,
    *,
    decay: float = DEFAULT_DECAY,
) -> np.ndarray | None:
    """用一根新 K 线把筹码分布向前推进一天（增量更新的核心数学）。

    - price_grid / chip_dist：上一交易日的网格与分布；
    - high / low / close：新 K 线前复权价；
    - turnover_rate：新 K 线换手率（百分数，5.0 = 5%）。

    返回 None 表示新 K 线价格越出 price_grid 范围（新筹码会丢失），调用方
    应对该股 re-base（重新构建网格后全量重算）。
    """
    grid = np.asarray(price_grid, dtype=np.float64)
    chip = np.asarray(chip_dist, dtype=np.float64)
    if grid.size == 0 or chip.size != grid.size:
        raise ValueError("price_grid 与 chip_dist 长度必须一致且非空")

    h, lo, c = float(high), float(low), float(close)
    grid_min, grid_max = float(grid[0]), float(grid[-1])
    # 网格自带 padding，因此只有触及/越出 grid[0]/grid[-1] 才丢筹码。
    if h >= grid_max or lo <= grid_min:
        return None

    t = float(turnover_rate)
    if not np.isfinite(t):
        t = 0.0
    t_decay = min(max(t / 100.0 * decay, 0.0), 1.0)
    if t_decay <= 0.0:
        return chip.copy()
    chip = chip * (1.0 - t_decay) + _triangle_new_chip(grid, h, lo, c, t_decay)
    total = float(chip.sum())
    if total > 0.0:
        chip = chip / total
    return chip


# --------------------------------------------------------------------------- #
# 内部实现
# --------------------------------------------------------------------------- #


def _select_window_start(
    turnover_rate: np.ndarray,
    min_cum_turnover: float,
    max_lookback: int,
) -> int:
    """返回回溯窗口的起始下标（含），使累计换手 ≥ min_cum_turnover 或达到上限。"""
    n = turnover_rate.size
    fraction = turnover_rate / 100.0
    cum = 0.0
    take = 0
    for i in range(n - 1, -1, -1):
        cum += fraction[i]
        take += 1
        if cum >= min_cum_turnover or take >= max_lookback:
            break
    return n - take


def _build_price_grid(low: np.ndarray, high: np.ndarray, bins: int, padding: float) -> np.ndarray:
    """基于窗口 min(low)/max(high) 构建固定线性网格，上下各留 padding 缓冲。"""
    lo = float(np.min(low))
    hi = float(np.max(high))
    pad = max(float(padding), 0.0)
    grid_min = max(lo * (1.0 - pad), _GRID_MIN_FLOOR)
    grid_max = hi * (1.0 + pad)
    if grid_max <= grid_min:
        grid_max = grid_min + _GRID_MIN_FLOOR
    return np.linspace(grid_min, grid_max, bins, dtype=np.float64)


def _recurse(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    turnover_rate: np.ndarray,
    *,
    decay: float,
    price_grid: np.ndarray,
) -> np.ndarray:
    """逐日递推筹码分布，返回归一化到 Σ=1 的分布。

    每步归一化（与 advance_chip_distribution 一致）：每天 t_decay 比例换手、
    总量恒为 1，保证全量与增量两种路径结果一致。
    """
    bins = price_grid.size
    chip = np.zeros(bins, dtype=np.float64)
    fraction = turnover_rate / 100.0
    for h, lo, c, t in zip(high, low, close, fraction, strict=True):
        t_decay = min(max(float(t) * decay, 0.0), 1.0)
        if t_decay <= 0.0:
            continue
        chip = chip * (1.0 - t_decay) + _triangle_new_chip(price_grid, h, lo, c, t_decay)
        total = float(chip.sum())
        if total > 0.0:
            chip = chip / total
    return chip


def _triangle_new_chip(
    price_grid: np.ndarray,
    high: float,
    low: float,
    close: float,
    mass: float,
) -> np.ndarray:
    """当日新筹码的三角形分布（以 close 为顶点），总质量 = mass。"""
    bins = price_grid.size
    if high <= low:
        # 一字板等无波动日：全部新筹码落在最接近 close 的 bin。
        idx = int(np.argmin(np.abs(price_grid - close)))
        out = np.zeros(bins, dtype=np.float64)
        out[idx] = mass
        return out
    span = high - low
    weights = 1.0 - np.abs(price_grid - close) / span
    np.maximum(weights, 0.0, out=weights)
    weights = np.where((price_grid >= low) & (price_grid <= high), weights, 0.0)
    total = float(weights.sum())
    if total <= 0.0:
        return np.zeros(bins, dtype=np.float64)
    return weights / total * mass


def _smooth(x: np.ndarray, width: int) -> np.ndarray:
    """边缘填充的滑动平均（宽度自动取奇）。"""
    x = np.asarray(x, dtype=np.float64)
    if x.size < 3 or width < 3:
        return x.copy()
    if width % 2 == 0:
        width += 1
    k = width // 2
    kernel = np.ones(width, dtype=np.float64) / width
    padded = np.pad(x, k, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _single_peak_measures(
    price_grid: np.ndarray,
    chip_dist: np.ndarray,
    current_price: float,
    smooth_width: int,
) -> tuple[float, float, float]:
    """简化单峰度量：平滑后取全局最大峰，向两侧走到局部极小得到 basin。

    返回 (single_peak_ratio, single_peak_width, main_cost)，其中 main_cost
    为主峰 basin 内筹码的加权均价（主力成本/顶格线）。
    """
    smoothed = _smooth(chip_dist, smooth_width)
    n = smoothed.size
    total = float(chip_dist.sum())
    if n == 0 or total <= 0.0:
        return 0.0, 0.0, float(current_price)

    p = int(np.argmax(smoothed))
    left = p
    while left > 0 and smoothed[left - 1] < smoothed[left]:
        left -= 1
    right = p
    while right < n - 1 and smoothed[right + 1] < smoothed[right]:
        right += 1

    basin_dist = chip_dist[left : right + 1]
    basin_grid = price_grid[left : right + 1]
    basin_mass = float(basin_dist.sum())
    peak_ratio = basin_mass / total
    width_price = float(price_grid[right] - price_grid[left])
    peak_width = width_price / current_price if current_price > 0.0 else 0.0
    main_cost = (
        float(np.average(basin_grid, weights=basin_dist))
        if basin_mass > 0.0 else float(current_price)
    )
    return peak_ratio, peak_width, main_cost
