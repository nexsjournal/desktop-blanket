"""tests/test_bumps.py — BumpField 契约验收（contract.md §3A + interfaces.py BumpField docstring）。

高度场公式（contracts/interfaces.py + 开发文档 §7.3）::

    A_i  = base_height_pt * (1 + min(log2(1 + size_bytes / 2**20), 2.0) * size_gain)
    c_i  = (x_pt + dx, y_pt + dy)
    z(p) = min(max_height_pt, Σ_i A_i * exp(-|p - c_i|² / (2 * sigma_pt²)))

默认参数：sigma=45、base=10、gain=0.6、max=28。由此推导：
  size=0     → log2(1)=0     → A=10.0
  size=1 MiB → log2(2)=1     → A=10*(1+0.6)=16.0
  size=10MiB → log2(11)>2    → A=10*(1+2*0.6)=22.0（饱和档）
  距峰值 σ 处衰减系数 = exp(-σ²/2σ²) = exp(-0.5)
被测实现 bumps.py 只读；纯逻辑、无 I/O。
"""
from __future__ import annotations

import numpy as np
import pytest

m = pytest.importorskip("bumps")
m_ci = pytest.importorskip("contracts.interfaces")

BumpField = m.BumpField
IconInfo = m_ci.IconInfo

W, H = 1440.0, 900.0
SIGMA = 45.0  # 契约默认


def field_with_one_icon(size_bytes: int, x: float = 700.0, y: float = 400.0,
                        calibration=(0.0, 0.0)) -> BumpField:
    return BumpField(
        [IconInfo(name="f.bin", x_pt=x, y_pt=y, size_bytes=size_bytes)],
        W, H, calibration=calibration,
    )


# ---------------------------------------------------------------------------
# 基本语义
# ---------------------------------------------------------------------------
def test_empty_icons_zero_field():
    """契约：icons 为空 → 恒为 0，且保持输入形状。"""
    field = BumpField([], W, H)
    xs = np.array([[0.0, 100.0], [500.0, 1439.0]])
    ys = np.array([[0.0, 800.0], [400.0, 899.0]])
    out = field.heights(xs, ys)
    assert out.shape == xs.shape
    assert np.all(out == 0.0)


@pytest.mark.parametrize(
    "size_bytes, expected_a",
    [
        (0, 10.0),                 # log2(1)=0 → A=10*(1+0)=10
        (1 * 2**20, 16.0),         # log2(2)=1 → A=10*(1+1*0.6)=16
        (10 * 2**20, 22.0),        # log2(11)=3.46→min 2 → A=10*(1+2*0.6)=22（饱和档）
        (int(0.25 * 2**20), 10.0 * (1.0 + np.log2(1.25) * 0.6)),  # 非饱和小数档：log2(1.25)≈0.3219
    ],
)
def test_single_icon_amplitude_matches_contract_formula(size_bytes, expected_a):
    """单图标中心高度 = A（高斯在中心处系数为 1），rtol=1e-3。"""
    field = field_with_one_icon(size_bytes)
    z = field.heights(np.array([700.0]), np.array([400.0]))
    assert z.shape == (1,)
    assert float(z[0]) == pytest.approx(expected_a, rel=1e-3)
    assert float(z[0]) >= 0.0


def test_gaussian_decay_at_one_sigma():
    """高斯衰减：距峰值 σ(=45pt) 处 z = A·exp(-0.5)，rtol=1e-3。"""
    field = field_with_one_icon(1 * 2**20)  # A=16
    z0 = float(field.heights(np.array([700.0]), np.array([400.0]))[0])
    z1 = float(field.heights(np.array([700.0 + SIGMA]), np.array([400.0]))[0])
    assert z0 == pytest.approx(16.0, rel=1e-3)
    assert z1 == pytest.approx(16.0 * np.exp(-0.5), rel=1e-3)
    # 远处单调衰减：3σ 处应远小于 σ 处
    z3 = float(field.heights(np.array([700.0 + 3 * SIGMA]), np.array([400.0]))[0])
    assert z3 < z1 * 0.05


def test_calibration_shifts_peak_position():
    """标定偏移在类内应用：c = finder + (dx,dy)。

    图标 finder(100,200)、calibration=(10,20) → 峰值在屏幕 (110,220)；
    (100,200) 处距峰值 √(10²+20²)=√500 → z = A·exp(-500/(2·45²))。
    """
    field = BumpField(
        [IconInfo(name="a.txt", x_pt=100.0, y_pt=200.0, size_bytes=0)],
        W, H, calibration=(10.0, 20.0),
    )
    z_peak = float(field.heights(np.array([110.0]), np.array([220.0]))[0])
    z_orig = float(field.heights(np.array([100.0]), np.array([200.0]))[0])
    assert z_peak == pytest.approx(10.0, rel=1e-3)                      # A(size=0)=10
    assert z_orig == pytest.approx(10.0 * np.exp(-500.0 / (2 * SIGMA**2)), rel=1e-3)
    assert z_peak > z_orig


def test_offscreen_icon_clipped():
    """契约：width/height 用于裁剪屏外图标 —— 标定后中心落在主屏外 → 无隆起。"""
    field = BumpField(
        [IconInfo(name="off.txt", x_pt=100.0, y_pt=100.0, size_bytes=0)],
        W, H, calibration=(-200.0, 0.0),  # 中心被移到 x=-100 < 0
    )
    xs = np.array([0.0, 100.0, 720.0, 1439.0])
    out = field.heights(xs, np.zeros_like(xs))
    assert np.all(out == 0.0)


# ---------------------------------------------------------------------------
# 堆叠叠加
# ---------------------------------------------------------------------------
def test_two_icons_same_spot_additive_but_capped():
    """同格叠加：两枚 size=0 图标 → 2·10=20 > 单枚 10，且 ≤ max_height_pt(28)。"""
    icons = [
        IconInfo(name="a.txt", x_pt=300.0, y_pt=300.0, size_bytes=0),
        IconInfo(name="b.txt", x_pt=300.0, y_pt=300.0, size_bytes=0),
    ]
    z = float(BumpField(icons, W, H).heights(np.array([300.0]), np.array([300.0]))[0])
    z_single = float(field_with_one_icon(0, 300.0, 300.0)
                     .heights(np.array([300.0]), np.array([300.0]))[0])
    assert z == pytest.approx(20.0, rel=1e-3)
    assert z > z_single
    assert z <= 28.0


def test_stack_truncated_at_max_height():
    """叠加截断：4 枚 1MiB 图标同点 → 裸和 4·16=64 → 截到 max_height_pt=28。"""
    icons = [
        IconInfo(name=f"s{i}.txt", x_pt=500.0, y_pt=500.0, size_bytes=1 * 2**20)
        for i in range(4)
    ]
    z = float(BumpField(icons, W, H).heights(np.array([500.0]), np.array([500.0]))[0])
    assert z == pytest.approx(28.0, rel=1e-3)


# ---------------------------------------------------------------------------
# 输出性质
# ---------------------------------------------------------------------------
def test_heights_preserve_input_arrays_and_shape():
    """契约：不修改输入数组；输出与输入同形状（任意形状，含 2D）。"""
    field = field_with_one_icon(1 * 2**20)
    xs = np.array([[0.0, 700.0], [800.0, 1200.0]])
    ys = np.array([[0.0, 400.0], [300.0, 800.0]])
    xs0, ys0 = xs.copy(), ys.copy()
    out = field.heights(xs, ys)
    assert out.shape == xs.shape
    assert np.array_equal(xs, xs0)
    assert np.array_equal(ys, ys0)
    assert np.all(out >= 0.0)
    # 中心点（数组里 (700,400)）应是全场最大
    assert float(out[0, 1]) == pytest.approx(16.0, rel=1e-3)
    assert float(out.max()) == float(out[0, 1])


def test_heights_deterministic_on_repeat_calls():
    """契约：同一输入重复调用结果确定（纯函数、无 I/O、无时钟）。"""
    field = field_with_one_icon(10 * 2**20)
    xs = np.linspace(0.0, W, 33)
    ys = np.linspace(0.0, H, 33)
    a = field.heights(xs, ys)
    b = field.heights(xs, ys)
    assert np.array_equal(a, b)
