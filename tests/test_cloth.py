"""tests/test_cloth.py — ClothSim 契约验收（contract.md §3A + interfaces.py ClothSim docstring）。

断言数值均从契约/开发文档 §7.2 推导，来源在用例注释中注明。
被测实现 cloth.py 只读；本文件无 AppKit、无 I/O、无真实时钟依赖。
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

m = pytest.importorskip("cloth")
m_ci = pytest.importorskip("contracts.interfaces")

ClothSim = m.ClothSim
BumpField = pytest.importorskip("bumps").BumpField
IconInfo = m_ci.IconInfo

# 契约默认网格 90×56 → N = 5040（interfaces.py vertices() docstring）
W, H = 1440.0, 900.0
COLS, ROWS = 90, 56
N = COLS * ROWS  # 5040
SX = W / (COLS - 1)
SY = H / (ROWS - 1)
DT = 1.0 / 30.0


def make_sim(**kw) -> ClothSim:
    return ClothSim(W, H, **kw)


# ---------------------------------------------------------------------------
# vertices()
# ---------------------------------------------------------------------------
def test_vertices_shape_dtype_layout():
    """§3A：vertices 形状 (5040,3)、float32、C-contiguous、初始平铺 z 全 0。"""
    sim = make_sim()
    v = sim.vertices()
    assert v.shape == (N, 3)
    assert v.dtype == np.float32
    assert v.flags.c_contiguous
    # 初始静止平铺：z（高度）全 0，且 ≥0（契约 z 向上）
    assert np.all(v[:, 2] == 0.0)
    assert np.all(v[:, 2] >= 0.0)


def test_vertices_initial_tiling_covers_screen():
    """§3A：初始 xy 覆盖 (0,0)-(w,h)；行主序 index = row*cols + col。

    间距 = W/(cols-1)、H/(rows-1)（类 docstring）→
    v[1] = (SX, 0)、v[COLS] = (0, SY)、v[-1] = (W, H)。
    """
    sim = make_sim()
    v = sim.vertices()
    assert v[0, 0] == pytest.approx(0.0, abs=1e-3)
    assert v[0, 1] == pytest.approx(0.0, abs=1e-3)
    assert v[1, 0] == pytest.approx(SX, abs=1e-3)      # 第二个粒子 = row0,col1
    assert v[1, 1] == pytest.approx(0.0, abs=1e-3)
    assert v[COLS, 0] == pytest.approx(0.0, abs=1e-3)  # 第 91 个 = row1,col0
    assert v[COLS, 1] == pytest.approx(SY, abs=1e-3)
    assert v[:, 0].min() == pytest.approx(0.0, abs=1e-2)
    assert v[:, 0].max() == pytest.approx(W, abs=1e-2)
    assert v[:, 1].min() == pytest.approx(0.0, abs=1e-2)
    assert v[:, 1].max() == pytest.approx(H, abs=1e-2)


# ---------------------------------------------------------------------------
# indices()
# ---------------------------------------------------------------------------
def test_indices_properties_and_reused_buffer():
    """§3A：dtype uint32、长度可整除 3、两次调用同一 buffer 对象、
    首格环绕顺序 [0, 1, cols+1, 0, cols+1, cols]（interfaces.py indices docstring）。
    """
    sim = make_sim()
    idx = sim.indices()
    assert idx.dtype == np.uint32
    assert idx.ndim == 1
    assert idx.size % 3 == 0
    # 每格 2 个三角形 × 3 顶点，共 (rows-1)*(cols-1) 格
    assert idx.size == (ROWS - 1) * (COLS - 1) * 6
    # 首格 (row=0, col=0)：a=0 → [0, 1, 0+cols+1, 0, 0+cols+1, 0+cols]
    assert list(idx[:6]) == [0, 1, COLS + 1, 0, COLS + 1, COLS]
    # 同一 buffer：第二次调用返回同一数组对象（勿逐帧重建）
    assert sim.indices() is idx
    # 索引值都在粒子范围内
    assert int(idx.min()) >= 0 and int(idx.max()) <= N - 1


# ---------------------------------------------------------------------------
# uv()
# ---------------------------------------------------------------------------
def test_uv_grid_and_corners():
    """§3A：uv 形状 (5040,2)、范围 [0,1]、四角映射 u=col/(cols-1)、v=row/(rows-1)。"""
    sim = make_sim()
    uv = sim.uv()
    assert uv.shape == (N, 2)
    assert float(uv.min()) >= 0.0
    assert float(uv.max()) <= 1.0
    assert uv[0] == pytest.approx((0.0, 0.0), abs=1e-6)                    # 左上
    assert uv[COLS - 1] == pytest.approx((1.0, 0.0), abs=1e-6)             # 右上
    assert uv[(ROWS - 1) * COLS] == pytest.approx((0.0, 1.0), abs=1e-6)    # 左下
    assert uv[-1] == pytest.approx((1.0, 1.0), abs=1e-6)                   # 右下


# ---------------------------------------------------------------------------
# grab / drag_to / release / 休眠
# ---------------------------------------------------------------------------
def test_grab_hit_center_and_miss_far_state_unchanged():
    """§3A：grab 中心→True；grab 远方→False 且状态完全不变（顶点与动能不变）。"""
    hit = make_sim()
    assert hit.grab(W / 2, H / 2) is True

    miss = make_sim()
    before = miss.vertices().copy()
    ke_before = miss.kinetic_energy()
    # 屏外远点：最近粒子 (0,0) 距 (-500,-500) ≈ 707pt > 默认抓取半径 40pt
    assert miss.grab(-500.0, -500.0) is False
    assert miss.grab(W + 500.0, H + 500.0) is False
    assert np.array_equal(before, miss.vertices())
    assert miss.kinetic_energy() == ke_before


def test_regrab_updates_grab_point():
    """已抓取时再次 grab 视为重抓（更新抓点，docstring 行为）。"""
    sim = make_sim()
    assert sim.grab(W / 2, H / 2) is True
    assert sim.grab(200.0, 200.0) is True  # 平铺布上该点必有粒子在 40pt 内
    sim.drag_to(150.0, 150.0)
    for _ in range(3):
        sim.step(DT)
    assert sim.nearest_distance(150.0, 150.0) < 60.0


def test_grab_then_drag_to_follows_target():
    """§3A：grab(中心)→drag_to(200,200)→step×5 后最近粒子距目标 <60pt。

    抓点粒子在约束迭代期间被持续钉在目标点（实现语义），5 帧后应已到位。
    """
    sim = make_sim()
    assert sim.grab(W / 2, H / 2) is True
    sim.drag_to(200.0, 200.0)
    for _ in range(5):
        sim.step(DT)
    assert sim.nearest_distance(200.0, 200.0) < 60.0


def test_release_converges_to_sleep_then_wake():
    """§3A：release 后能量收敛 → is_asleep()（实现者报告 ≈87 步，预算给足 400 步）；
    wake() 重新拉起后应能再次收敛。
    """
    sim = make_sim()
    assert sim.grab(W / 2, H / 2) is True
    sim.drag_to(250.0, 250.0)
    for _ in range(5):
        sim.step(DT)
    sim.release(fling=False)
    assert not sim.is_asleep()

    asleep_at = None
    for i in range(400):
        sim.step(DT)
        if sim.is_asleep():
            asleep_at = i + 1
            break
    print(f"INFO cloth: asleep after {asleep_at} steps")
    assert asleep_at is not None, "400 步内未休眠"
    # is_asleep 定义：kinetic_energy() < ε 且未抓取（契约）→ 收敛值应有限且为正小量
    ke = sim.kinetic_energy()
    assert math.isfinite(ke) and ke >= 0.0

    sim.wake()
    assert not sim.is_asleep()
    for _ in range(10):
        sim.step(DT)
    assert sim.is_asleep()  # 无新能量注入，应迅速再次收敛


# ---------------------------------------------------------------------------
# BumpField 注入
# ---------------------------------------------------------------------------
def test_bump_field_raises_center_then_clears():
    """§3A：注入中心 10MB 图标 → 中心附近顶点 z>3；set_bumps(None) 后回落。

    幅度按契约公式：A = 10*(1 + min(log2(1+10), 2)*0.6) = 10*1.2 = 22pt，
    中心顶点目标 z ≈ 22（顶点距图标中心 < 1 格 ≈ 16pt，高斯衰减 < 5%），
    弱弹簧追踪 90 帧（3s）后必然 >3pt；撤场后波动方程阻尼衰减 300 帧回落 <3pt。
    """
    icon = IconInfo(name="big.bin", x_pt=W / 2, y_pt=H / 2, size_bytes=10 * 2**20)
    field = BumpField([icon], W, H)  # calibration (0,0) → 峰值正好在 (W/2, H/2)
    sim = make_sim(bumps=field)
    for _ in range(90):
        sim.step(DT)
    v = sim.vertices()
    assert v[:, 2].max() > 3.0
    d2 = (v[:, 0] - W / 2) ** 2 + (v[:, 1] - H / 2) ** 2
    assert float(v[int(np.argmin(d2)), 2]) > 3.0  # 图标正上方顶点抬升

    sim.set_bumps(None)  # 降级/无隆起模式
    sim.wake()
    for _ in range(300):
        sim.step(DT)
    v2 = sim.vertices()
    assert v2[:, 2].max() < 3.0
    assert np.isfinite(v2).all()


# ---------------------------------------------------------------------------
# nearest_distance
# ---------------------------------------------------------------------------
def test_nearest_distance_thresholds():
    """§3A：平铺时中心 <40pt（网格间距 ≈16pt，最近粒子 ≈11pt）；远点 >40pt。

    供 hitTest 的布尔语义：≤40 收鼠标、否则穿透（契约 nearest_distance docstring）。
    """
    sim = make_sim()
    assert sim.nearest_distance(W / 2, H / 2) < 40.0
    assert sim.nearest_distance(-1000.0, -1000.0) > 40.0  # 最近粒子 (0,0)，距 ≈1414pt


# ---------------------------------------------------------------------------
# throw_in
# ---------------------------------------------------------------------------
def test_throw_in_invalid_corner_raises_valueerror():
    """§3A/§5：非法 from_corner 抛 ValueError。"""
    sim = make_sim()
    for bad in ("middle", "", "TOP_LEFT", "top-center"):
        with pytest.raises(ValueError):
            sim.throw_in((W / 2, H / 2), bad)


@pytest.mark.parametrize("corner", ["top_left", "top_right", "bottom_left", "bottom_right"])
def test_throw_in_valid_corner_leaves_original_spot(corner):
    """§3A：合法 corner → 整布移到屏外角落再飞向落点；step 后布料离开原位。

    抛掷锚点在屏外（如 top_left → (-0.5W-60, -0.5H-60)，实现语义），
    故 throw_in+3 步后布料中心与初始中心 (W/2,H/2) 距离必然 >300pt。
    """
    sim = make_sim()
    sim.throw_in((W / 2, H / 2), corner)
    for _ in range(3):
        sim.step(DT)
    v = sim.vertices()
    cx, cy = float(v[:, 0].mean()), float(v[:, 1].mean())
    assert math.hypot(cx - W / 2, cy - H / 2) > 300.0
    assert np.isfinite(v).all()


# ---------------------------------------------------------------------------
# 稳健性
# ---------------------------------------------------------------------------
def test_step_extreme_dt_clamped_no_nan():
    """契约 step docstring：dt 内部 clamp 到 [1/240, 0.05]。dt=0 / 1.0 / 负值不崩、无 NaN。"""
    sim = make_sim()
    sim.step(0.0)
    sim.step(1.0)
    sim.step(-0.5)
    v = sim.vertices()
    assert np.isfinite(v).all()
    assert math.isfinite(sim.kinetic_energy())


def test_500_consecutive_steps_no_nan_inf_and_settles():
    """稳健性：抓取甩动 → fling 释放 → 连续 500 步无 NaN/Inf、z≥0、最终可休眠。"""
    sim = make_sim()
    assert sim.grab(W / 2, H / 2) is True
    for i in range(6):  # 快速拖动制造 fling 历史（速度远超 600pt/s 阈值语义）
        sim.drag_to(100.0 + i * 180.0, 700.0 - i * 90.0)
        sim.step(DT)
    sim.release(fling=True)
    for _ in range(500):
        sim.step(DT)
    v = sim.vertices()
    assert np.isfinite(v).all()
    assert np.all(v[:, 2] >= 0.0)  # 契约：z ≥ 0（向上）
    ke = sim.kinetic_energy()
    assert math.isfinite(ke) and ke >= 0.0
    assert sim.is_asleep()


def test_perf_smoke_100_steps():
    """性能冒烟：持续抓取拖动下 100 步耗时打印（不设硬阈值，仅供人工核对）。"""
    sim = make_sim()
    assert sim.grab(W / 2, H / 2) is True
    t0 = time.perf_counter()
    for i in range(100):
        # 抓取中持续移动目标（幅度保证每帧速度 >40pt/s，走全量物理路径）
        sim.drag_to(W / 2 + 80.0 * math.sin(i / 7.0), H / 2 + 80.0 * math.cos(i / 5.0))
        sim.step(DT)
    elapsed = time.perf_counter() - t0
    print(f"PERF ClothSim: 100 steps in {elapsed * 1000:.1f} ms "
          f"(avg {elapsed * 10:.3f} ms/step)")
    assert np.isfinite(sim.vertices()).all()


# ---------------------------------------------------------------------------
# 入睡时限（回归锚点：有隆起时曾需 15–24s，目标 ≤8s；无隆起 ≤6s）
# 公开路径：throw_in 到中心 → 30fps step 直到 is_asleep()；不碰私有属性。
# ---------------------------------------------------------------------------
SLEEP_MAX_STEPS = 30 * 20  # 上界保护：20s（超出即明确失败，避免死循环）


def steps_until_asleep(sim: ClothSim, target=(600.0, 450.0)) -> int:
    """throw_in 到 target 后按 30fps 步进直到 is_asleep()；返回步数。"""
    sim.throw_in(target)
    steps = 0
    for _ in range(SLEEP_MAX_STEPS):
        sim.step(DT)
        steps += 1
        if sim.is_asleep():
            break
    return steps


def test_sleep_within_6s_without_bumps():
    """无隆起：throw_in 落定后 ≤6.0s（180 帧）入睡；上界保护 20s。"""
    sim = ClothSim(1200.0, 900.0)
    steps = steps_until_asleep(sim)
    seconds = steps / 30.0
    print(f"INFO cloth: 无隆起入睡 {steps} 步 = {seconds:.2f}s")
    assert sim.is_asleep(), f"{SLEEP_MAX_STEPS} 步（20s 上界）内未入睡"
    assert seconds <= 20.0, f"上界保护失败：入睡耗时 {seconds:.2f}s > 20s"
    assert seconds <= 6.0, f"无隆起入睡耗时 {seconds:.2f}s > 6.0s"


def test_sleep_within_8s_with_bumps():
    """带 3 个图标隆起（1MB~100MB、毯中部）：throw_in 落定后 ≤8.0s 入睡。

    （现状回归项：隆起弱弹簧让 z 场长期微振，曾需约 15–24s。）
    """
    icons = [
        IconInfo(name="small.txt", x_pt=480.0, y_pt=380.0, size_bytes=1 * 2**20),
        IconInfo(name="mid.bin", x_pt=600.0, y_pt=460.0, size_bytes=20 * 2**20),
        IconInfo(name="big.mov", x_pt=720.0, y_pt=540.0, size_bytes=100 * 2**20),
    ]
    field = BumpField(icons, 1200.0, 900.0)
    sim = ClothSim(1200.0, 900.0, bumps=field)
    steps = steps_until_asleep(sim)
    seconds = steps / 30.0
    print(f"INFO cloth: 带隆起入睡 {steps} 步 = {seconds:.2f}s")
    assert sim.is_asleep(), f"{SLEEP_MAX_STEPS} 步（20s 上界）内未入睡"
    assert seconds <= 20.0, f"上界保护失败：入睡耗时 {seconds:.2f}s > 20s"
    assert seconds <= 8.0, f"带隆起入睡耗时 {seconds:.2f}s > 8.0s"


# ---------------------------------------------------------------------------
# v4：真 3D 翻折（§12.16）—— 翻底三角 = 几何环绕方向朝下（渲染用背面贴图）
# ---------------------------------------------------------------------------
def _flipped_tris(sim: ClothSim) -> int:
    pos = sim._pos
    tri = np.asarray(sim.indices()).reshape(-1, 3).astype(np.int64)
    e1 = pos[tri[:, 1]] - pos[tri[:, 0]]
    e2 = pos[tri[:, 2]] - pos[tri[:, 0]]
    nz = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    return int((nz < -1e-6).sum())


def _drag_across(sim: ClothSim, w: float, h: float) -> None:
    """抓左上角附近 → 平滑横过布身（50 帧）→ 松手 → 落定（或入睡）。"""
    gx0, gy0 = 0.12 * w, 0.14 * h
    assert sim.grab(gx0, gy0, 40.0) is True
    tx, ty = 0.88 * w, 0.72 * h
    n = 50
    for i in range(1, n + 1):
        t = i / n
        tt = t * t * (3.0 - 2.0 * t)
        sim.drag_to(gx0 + (tx - gx0) * tt, gy0 + (ty - gy0) * tt)
        sim.step(DT)
    sim.release(fling=False)
    for _ in range(200):
        sim.step(DT)
        if sim.is_asleep():
            break


def test_fold_over_appears_and_persists():
    """v4 核心：拖一角横过布身 → 出现真翻折（几何翻底），落定后仍保持（摩擦锁住）。"""
    sim = make_sim()
    _drag_across(sim, W, H)
    flipped = _flipped_tris(sim)
    assert flipped >= 100, f"翻底三角过少（{flipped}）：没有形成真翻折"
    sim.wake()
    for _ in range(90):          # 再静置 3s
        sim.step(DT)
    assert _flipped_tris(sim) >= 50, "落定后翻折消散了"


def test_no_interpenetration_after_fold():
    """v4：翻折后不应有非邻接粒子互穿（自碰撞保证两层间距）。"""
    sim = make_sim()
    _drag_across(sim, W, H)
    p = sim._pos[::2]
    idx = np.arange(0, N, 2)
    rr = idx // COLS
    cc = idx % COLS
    d2 = ((p[:, None, 0] - p[None, :, 0]) ** 2
          + (p[:, None, 1] - p[None, :, 1]) ** 2
          + (p[:, None, 2] - p[None, :, 2]) ** 2)
    near = (np.abs(rr[:, None] - rr[None, :]) <= 1) & (np.abs(cc[:, None] - cc[None, :]) <= 1)
    d2 = d2[~near]
    dmin = float(np.sqrt(d2.min()))
    assert dmin > 0.3, f"存在互穿：非邻接粒子最小间距 {dmin:.2f}pt"


def test_flatten_folds_lays_flat():
    """菜单「摊平」：flatten_folds() 后应完全落平、翻底归零。"""
    sim = make_sim()
    _drag_across(sim, W, H)
    assert _flipped_tris(sim) >= 100
    sim.flatten_folds()
    for _ in range(60):
        sim.step(DT)
    assert float(sim.vertices()[:, 2].max()) < 3.0, "摊平后仍有高度"
    assert _flipped_tris(sim) < 20, "摊平后仍有翻底"
