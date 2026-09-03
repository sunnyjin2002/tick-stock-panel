"""筹码分布（chip.py）核心算法单元测试。"""
from __future__ import annotations

import numpy as np
import pytest

from app.indicators.chip import (
    DEFAULT_BINS,
    _build_price_grid,
    _select_window_start,
    _triangle_new_chip,
    advance_chip_distribution,
    compute_chip_distribution,
    extract_factors,
)

# --------------------------------------------------------------------------- #
# 三角形分布
# --------------------------------------------------------------------------- #


def test_triangle_new_chip_peaks_at_close_and_sums_to_mass():
    grid = np.linspace(9.0, 11.0, 21)  # 步长 0.1，grid[10] == 10.0
    out = _triangle_new_chip(grid, high=10.5, low=9.5, close=10.0, mass=0.4)

    assert out.sum() == pytest.approx(0.4)
    assert int(out.argmax()) == 10
    # 顶点处最高，向两侧单调下降
    assert out[9] < out[10]
    assert out[11] < out[10]
    assert out[8] < out[9]


def test_triangle_new_chip_one_word_board_places_at_close():
    grid = np.linspace(9.0, 11.0, 21)
    out = _triangle_new_chip(grid, high=10.0, low=10.0, close=10.0, mass=0.5)

    assert out.sum() == pytest.approx(0.5)
    assert int(out.argmax()) == 10
    assert out[10] == pytest.approx(0.5)
    assert out[9] == 0.0


# --------------------------------------------------------------------------- #
# 窗口选择与网格
# --------------------------------------------------------------------------- #


def test_select_window_by_cum_turnover():
    turnover = np.array([5.0, 5.0, 5.0, 5.0, 5.0])  # 每日 5%
    # 累计换手 ≥ 0.12 需要 3 天
    assert _select_window_start(turnover, min_cum_turnover=0.12, max_lookback=2500) == 2


def test_select_window_caps_at_max_lookback():
    turnover = np.array([1.0] * 10)  # 每日 1%，累计远达不到 3.0
    assert _select_window_start(turnover, min_cum_turnover=3.0, max_lookback=3) == 7


def test_build_price_grid_includes_padding():
    grid = _build_price_grid(np.array([10.0]), np.array([12.0]), bins=100, padding=0.05)

    assert grid.size == 100
    assert grid[0] == pytest.approx(9.5)
    assert grid[-1] == pytest.approx(12.6)
    assert np.all(np.diff(grid) > 0)


# --------------------------------------------------------------------------- #
# 因子提取
# --------------------------------------------------------------------------- #


def test_extract_factors_on_known_distribution():
    grid = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0])
    dist = np.array([0.05, 0.05, 0.4, 0.4, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0])

    factors = extract_factors(grid, dist, current_price=13.0, hist_close=np.array([10.0, 20.0]))

    assert factors.profit_ratio == pytest.approx(0.90)  # 10,11,12,13 共 0.9
    # p5 = 10, p95 = 14 → (14-10)/(14+10) = 1/6
    assert factors.concentration_90 == pytest.approx(4.0 / 24.0)
    assert factors.avg_cost == pytest.approx(12.5)
    assert factors.avg_cost_deviation == pytest.approx((13.0 - 12.5) / 12.5)
    # 现价 13，近一年窗口 [10,20] 的 30% 分位 = 13，13 < 13 为 False
    assert factors.is_low_position is False
    assert 0.0 <= factors.single_peak_ratio <= 1.0
    assert factors.single_peak_width >= 0.0


def test_is_low_position_threshold():
    grid = np.array([0.0, 1.0])
    dist = np.array([0.5, 0.5])
    hist = np.array([10.0, 20.0])  # 30% 分位 = 13.0

    low = extract_factors(grid, dist, current_price=12.0, hist_close=hist)
    high = extract_factors(grid, dist, current_price=14.0, hist_close=hist)

    assert low.is_low_position is True
    assert high.is_low_position is False


# --------------------------------------------------------------------------- #
# 端到端
# --------------------------------------------------------------------------- #


def test_compute_chip_distribution_sums_to_one():
    n = 300
    close = np.linspace(10.0, 12.0, n)
    high = close * 1.02
    low = close * 0.98
    turnover = np.full(n, 20.0)  # 20%

    result = compute_chip_distribution(high, low, close, turnover)

    assert result.price_grid.size == DEFAULT_BINS
    assert result.chip_dist.size == DEFAULT_BINS
    assert result.chip_dist.sum() == pytest.approx(1.0)
    assert np.all(result.chip_dist >= 0.0)
    assert np.all(np.diff(result.price_grid) > 0)


def test_constant_price_concentrates_near_price():
    n = 300
    close = np.full(n, 10.0)
    high = np.full(n, 10.0)
    low = np.full(n, 10.0)
    turnover = np.full(n, 30.0)

    result = compute_chip_distribution(high, low, close, turnover)
    factors = extract_factors(result.price_grid, result.chip_dist, 10.0, close)

    assert factors.avg_cost == pytest.approx(10.0, abs=0.1)
    assert factors.concentration_90 < 0.05


def test_decay_clamped_never_negative():
    # 换手 100% + decay=2.0 会让 t_decay 超 1，必须 clamp 到 1，否则出现负分布
    high = np.array([10.5, 10.5])
    low = np.array([9.5, 9.5])
    close = np.array([10.0, 10.0])
    turnover = np.array([100.0, 100.0])

    result = compute_chip_distribution(high, low, close, turnover, decay=2.0)

    assert np.all(result.chip_dist >= 0.0)
    assert result.chip_dist.sum() == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 边界与失败路径
# --------------------------------------------------------------------------- #


def test_turnover_nan_treated_as_zero():
    high = np.full(3, 10.0)
    low = np.full(3, 10.0)
    close = np.full(3, 10.0)
    turnover = np.array([np.nan, np.nan, 30.0])

    result = compute_chip_distribution(high, low, close, turnover)

    assert np.all(np.isfinite(result.chip_dist))
    assert result.chip_dist.sum() == pytest.approx(1.0)


def test_all_nan_turnover_returns_zero_distribution():
    high = np.full(3, 10.0)
    low = np.full(3, 10.0)
    close = np.full(3, 10.0)
    turnover = np.full(3, np.nan)

    result = compute_chip_distribution(high, low, close, turnover)

    assert result.chip_dist.sum() == 0.0


def test_empty_input_raises():
    with pytest.raises(ValueError):
        compute_chip_distribution(
            np.array([]), np.array([]), np.array([]), np.array([]),
        )


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        compute_chip_distribution(
            np.array([10.0, 10.0]),
            np.array([9.0, 9.0]),
            np.array([10.0, 10.0]),
            np.array([5.0]),  # 长度不一致
        )


def test_bins_too_small_raises():
    with pytest.raises(ValueError):
        compute_chip_distribution(
            np.array([10.0]), np.array([9.0]), np.array([10.0]), np.array([5.0]),
            bins=1,
        )


# --------------------------------------------------------------------------- #
# 增量推进 advance_chip_distribution
# --------------------------------------------------------------------------- #


def test_advance_preserves_unit_mass():
    grid = np.linspace(9.0, 11.0, 21)
    chip = np.full(grid.size, 1.0 / grid.size)  # 均匀分布，Σ=1

    out = advance_chip_distribution(
        grid, chip, high=10.5, low=9.5, close=10.0, turnover_rate=20.0, decay=1.0,
    )

    assert out is not None
    assert out.sum() == pytest.approx(1.0)
    assert np.all(out >= 0.0)


def test_advance_returns_none_when_high_exceeds_grid():
    grid = np.linspace(9.0, 11.0, 21)  # grid_max = 11.0
    chip = np.full(grid.size, 1.0 / grid.size)

    out = advance_chip_distribution(
        grid, chip, high=11.0, low=10.0, close=10.5, turnover_rate=20.0,
    )

    assert out is None


def test_advance_returns_none_when_low_below_grid():
    grid = np.linspace(9.0, 11.0, 21)  # grid_min = 9.0
    chip = np.full(grid.size, 1.0 / grid.size)

    out = advance_chip_distribution(
        grid, chip, high=10.0, low=9.0, close=9.5, turnover_rate=20.0,
    )

    assert out is None


def test_advance_zero_turnover_returns_copy_unchanged():
    grid = np.linspace(9.0, 11.0, 21)
    chip = np.full(grid.size, 1.0 / grid.size)

    out = advance_chip_distribution(
        grid, chip, high=10.5, low=9.5, close=10.0, turnover_rate=0.0,
    )

    assert out is not None
    np.testing.assert_allclose(out, chip)


def test_advance_matches_full_recompute_exactly():
    # 用固定窗口(min_cum_turnover 永不达标)使窗口=全部历史, 网格不变,
    # 从而 advance 与全量重算在数学上严格一致。
    n = 100
    high = np.full(n, 10.1)
    low = np.full(n, 9.9)
    close = np.full(n, 10.0)
    turnover = np.full(n, 5.0)
    kwargs = {"min_cum_turnover": 1e9, "max_lookback": 1000}

    full99 = compute_chip_distribution(high[:99], low[:99], close[:99], turnover[:99], **kwargs)
    full100 = compute_chip_distribution(high, low, close, turnover, **kwargs)

    advanced = advance_chip_distribution(
        full99.price_grid, full99.chip_dist,
        high[99], low[99], close[99], turnover[99],
    )

    assert advanced is not None
    np.testing.assert_allclose(full99.price_grid, full100.price_grid)
    np.testing.assert_allclose(advanced, full100.chip_dist, atol=1e-12)
