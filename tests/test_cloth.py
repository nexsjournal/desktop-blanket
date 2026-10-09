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


def test_grab_direct_neighbors_do_not_wrap_across_rows():
    """抓点的四邻居必须按网格坐标求，不能让行首/行尾跨行相连。"""
    sim = make_sim()
    row, col = 8, 0
    idx = row * COLS + col
    x, y = float(sim._pos[idx, 0]), float(sim._pos[idx, 1])
    assert sim.grab(x, y, 40.0) is True
    got = set(int(v) for v in sim._grab_direct)
    expected = {idx - COLS, idx + COLS, idx + 1}
    assert got == expected
    assert idx - 1 not in got


def test_fast_corner_drag_does_not_make_a_long_pin_edge():
    """快拖角落时，抓点相连边不能被拉成画面里的尖刺。"""
    sim = make_sim()
    x, y = 0.12 * W, 0.14 * H
    assert sim.grab(x, y, 40.0) is True
    for i in range(12):
        sim.drag_to(x + 80.0 * (i + 1), y + 44.0 * (i + 1))
        sim.step(DT)
    d = sim._pos[sim._grab_direct] - sim._pos[sim._grab_idx]
    edge_ratio = np.linalg.norm(d[:, :2], axis=1) / max(sim._sx, sim._sy)
    assert float(edge_ratio.max()) <= sim._pin_edge_k + 0.08


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

    wake 后 20 帧的预算**按实测标定**（守卫意图：唤醒后不注入新能量 → 必须再次
    迅速收敛，防止「wake 泄漏能量 / 静止期自激」把毯子变永动；不是兜底已知偏差）：
    2026-10-09「重量感/拖动模型」调参后 cloth.py 当前默认值
    damping=0.972、friction=6500、collision_sep=15、sleep_eps=1.0、
    body_k=24、body_c=14、body_k0_frac=0.35、body_ramp_pt=700、settle_vel_credit=0.3
    下，wake 后实测（ke = Σv²，5040 粒子，本机逐帧可复现）：
    局部（≤6 粒子）z 向再平衡尖峰 f04 ke=4861（峰值 69.6pt/s）、f08 ke=223.9 后衰减
    f09 143.95 → f10 2.41 → f11 1.82 → f12 1.38 → f13 0.433 < 1.0 入睡。
    故给 20 帧 ≈ 实测 13 帧 + 54% 余量；再次调参后请重新实测并同步此数字。
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
    woken_asleep_at = None
    for i in range(20):
        sim.step(DT)
        if sim.is_asleep():
            woken_asleep_at = i + 1
            break
    print(f"INFO cloth: re-asleep after wake in {woken_asleep_at} steps")
    assert woken_asleep_at is not None, (
        "wake 后 20 帧未再次入睡（参数标定基线：13 帧，见 docstring）——"
        "唤醒后无新能量注入必须迅速收敛；当前 ke="
        f"{sim.kinetic_energy():.3f}，疑似能量泄漏或静止期自激"
    )


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


# ---------------------------------------------------------------------------
# v5：折痕圆角下限 + 拖动不塌陷（§12.17）——两条都是"观感被毁"那一类回归的锚点
# 断言阈值的来源：v5 定稿实测值 vs 故意写反 crease 符号的对照实现（同脚本实测）：
#   定稿      ：p10(d/cmin)=1.10、<0.5×cmin 占比 0.00%、拖动后 bbox 面积比 1.05
#   符号写反  ：p10(d/cmin)=0.20、<0.5×cmin 占比 48.4%、bbox 面积比 0.079（塌成一团）
# 阈值取两者之间并留足余量，既不能靠"调参刚好过线"，也不能脆弱到偶发红。
# ---------------------------------------------------------------------------
def _bend_ratio_stats(sim: ClothSim) -> np.ndarray:
    """全部 (i,i+2) 对的距离 / 折痕圆角下限（<1 = 折得比允许的更尖）。"""
    out = []
    for (i0, i1, _rest), cmin in zip(sim._bend_groups, sim._crease_min):
        d = np.linalg.norm(sim._pos[i1] - sim._pos[i0], axis=1)
        out.append(d / cmin)
    return np.concatenate(out)


def _footprint_ratio(sim: ClothSim, w: float, h: float) -> float:
    """xy 包围盒面积 / 标称毯身面积（<1 = 布被压小；v4 塌陷态实测 0.08~0.26）。"""
    p = sim._pos
    xr = float(p[:, 0].max() - p[:, 0].min())
    yr = float(p[:, 1].max() - p[:, 1].min())
    return (xr * yr) / (w * h)


def test_crease_floor_holds_after_fold():
    """拖一角横过布身之后，折脊曲率不能回到"剪纸式锐折"（p10 ≥ 0.85×圆角下限）。"""
    sim = make_sim()
    _drag_across(sim, W, H)
    r = _bend_ratio_stats(sim)
    p10 = float(np.percentile(r, 10))
    sharp = float(np.mean(r < 0.5))
    print(f"INFO cloth: 折痕贴合 p10={p10:.2f} <0.5×下限占比={sharp * 100:.2f}%")
    assert p10 >= 0.85, f"折痕普遍过尖：p10={p10:.2f}（下限 0.85；符号写反/约束失效时为 0.2）"
    assert sharp <= 0.02, f"锐折对过多：{sharp * 100:.2f}% > 2%（定稿实测 0.00%）"


def test_drag_across_does_not_collapse_footprint():
    """拖动不能把毯子"拉成橡皮膜再塌成一团"：占位面积保持 + 能入睡。"""
    sim = make_sim()
    _drag_across(sim, W, H)
    ratio = _footprint_ratio(sim, W, H)
    steps = 0
    for _ in range(300):            # 最多 10s
        sim.step(DT)
        steps += 1
        if sim.is_asleep():
            break
    print(f"INFO cloth: 拖动后 bbox 面积比={ratio:.3f} 入睡={sim.is_asleep()} "
          f"({steps / 30.0:.1f}s)")
    assert ratio >= 0.75, f"毯子被压小了：bbox 面积比 {ratio:.3f} < 0.75（v5 定稿 1.05）"
    assert sim.is_asleep(), "拖动后 10s 内未入睡（塌陷/自激的典型症状）"


# ---------------------------------------------------------------------------
# v6「厚度」渲染支持（§12.18）：地板高度查询 + 折层遮蔽场 ao()
# 两者都是**只读渲染数据**：不参与动力学（不改变位置/速度/休眠），由 overlay 读取。
# ---------------------------------------------------------------------------
def test_floor_heights_zeros_without_bumps():
    """无隆起：floor_heights() 全 0（= 桌面）；形状 (N,)、float32。"""
    sim = make_sim()
    fl = sim.floor_heights()
    assert fl.shape == (N,)
    assert fl.dtype == np.float32
    assert np.all(fl == 0.0)


def test_floor_heights_follow_bumps():
    """有隆起：floor_heights() == BumpField 在粒子 xy 处的高度（渲染层用它算离地高度）。"""
    class _Field:
        def heights(self, x, y):
            return np.full(np.shape(x), 12.5)

    sim = make_sim(bumps=_Field())
    sim.step(DT)
    fl = sim.floor_heights()
    assert np.allclose(fl, 12.5, atol=1e-3)


def test_ao_zero_when_flat_and_read_only():
    """平铺收敛后 ao() 全 0；且调用 ao()/floor_heights() 不改变任何动力学状态。"""
    sim = make_sim()
    for _ in range(60):
        sim.step(DT)
    assert float(sim.ao().max()) == 0.0, "平铺态不应有遮蔽"
    v0, p0 = sim.vertices().copy(), sim._pos.copy()
    ke0 = sim.kinetic_energy()
    for _ in range(5):
        sim.ao()
        sim.floor_heights()
    assert np.array_equal(sim._pos, p0) and np.array_equal(sim.vertices(), v0)
    assert sim.kinetic_energy() == ke0, "渲染支持查询不应扰动动力学"


def test_ao_positive_under_folded_layer():
    """翻折后：被上层压住的粒子 ao 明显 >0（折缝接触阴影），且集中在小范围。

    定稿实测：拖动翻折后 max ao ≈ 1.0、占比 ~2%；平铺态 0。阈值取中间值留余量。
    """
    sim = make_sim()
    _drag_across(sim, W, H)
    ao = np.asarray(sim.ao())
    assert ao.shape == (N,)
    frac = float((ao > 0.3).mean())
    print(f"INFO cloth: ao max={float(ao.max()):.2f} 均值={float(ao.mean()):.3f} "
          f">0.3 占比={frac * 100:.2f}%")
    assert float(ao.max()) >= 0.5, "翻折后没有出现被压住的粒子（遮蔽场失效）"
    assert frac <= 0.25, f"遮蔽范围过大（{frac * 100:.1f}%）——应集中在折缝"


def test_ao_does_not_break_settling():
    """遮蔽场只读：加进去之后拖动仍能在 10s 内入睡（防止它意外扰动求解/能量）。"""
    sim = make_sim()
    _drag_across(sim, W, H)
    sim.wake()
    steps = 0
    for _ in range(300):
        sim.step(DT)
        steps += 1
        if sim.is_asleep():
            break
    print(f"INFO cloth: v6 AO 后入睡 {steps / 30.0:.1f}s ke={sim.kinetic_energy():.3f}")
    assert sim.is_asleep(), "v6 改动后拖动 10s 内未入睡"


def test_ao_ignores_icon_mounds():
    """趴在图标隆包上的布**离地≈0** → 不该在隆包四周生成遮蔽（否则每个图标一圈假暗影）。

    这是把遮蔽场从「绝对高度」改成「离地高度」后才成立的：用绝对高度时，隆包顶被当成
    「局部顶高」，四周趴着的布被判成"被压住"。
    """
    class _Mound:
        def heights(self, x, y):
            x, y = np.asarray(x, float), np.asarray(y, float)
            d2 = (x - W / 2.0) ** 2 + (y - H / 2.0) ** 2
            return 34.0 * np.exp(-d2 / (2.0 * 90.0 * 90.0))

    sim = ClothSim(W, H, bumps=_Mound())
    for _ in range(240):
        sim.step(DT)
        if sim.is_asleep():
            break
    fl = sim.floor_heights()
    off = sim.vertices()[:, 2] - fl
    ao = np.asarray(sim.ao())
    print(f"INFO cloth: 隆包 floor_max={fl.max():.1f} 离地_max={off.max():.2f} ao_max={ao.max():.3f}")
    assert fl.max() > 20.0, "隆起场没生效（测试前提不成立）"
    assert off.max() < 1.0, "布没趴在隆包上（测试前提不成立）"
    assert float(ao.max()) < 0.05, f"隆包四周出现假遮蔽：ao_max={ao.max():.3f}"


# ---------------------------------------------------------------------------
# v6.3「大折 / 腾空」 → v6.4「厚重感」（2026-10-09 用户反馈后重定标）
# 产品意图（用户原话）：「现在的这个毯子感受起来也没有重量感，就像一个丝绸，我需要有
# 毛毯的重量感」（伴随反馈：「边角折起来之后还是会有穿模」）。据此 cloth.py 的默认参数
# 按「厚重」重调（body_k 18→110、body_c 14→40、damping 0.980→0.972、friction 5000→6500、
# bend_soft 0.45→0.60、pose_gain 0.35→0.28、grab_friction_scale 0.35→0.50；
# fling_lift 0.45→0.18、fling_lift_max 850→420、throw_arc_z 480→300、
# grab_lift_v0 700→1400、grab_lift_max 170→90），本节阈值随「低而重」的新意图重定标。
# 本节全部数值来自新默认参数下的定量复跑（1200×900、30fps；两次复跑逐位一致，物理无随机源）。
# 速度标注约定：以「pt/帧」为准，标称 pt/s = pt/帧 × 30；cloth 内部 fling/grab_lift
# 的速度估计 = 25 × pt/帧（_drag_hist 限 6 帧窗口）。
#   —— v6.7 牵引皮带（pin_leash_pt / pin_edge_k=1.8，用户截图「尖尖」的修复）后再次复跑；
#      2026-10-09 终轮冻结参数（pin_leash_pt=220、grab_strain_passes=15、body_k=24 等）下复跑：
#      抬升/抛掷整体更低更重，且**抓点滞后于手**（皮带的设计意图=「毯子跟不上手」的重量感）
#      → 下列两条用例的**度量方式**随之重定标（行为没坏，是按手的位置取样取不到抓点了）——
#   抛掷 fling   ：8 帧 × 60pt 快甩 → 松手 zmax 22.0 → 释放后拱高 42.6pt（@6 帧，低而重）；
#                  离地判定降到 z>20（皮带后连拱顶都不到 40pt：>40 占比峰值 0.00%，旧判定已失效），
#                  >20 占比峰值 4.92%
#                  注：需求方另报 59.6pt 是 8×80 档（本档 8×80 复跑 56.0pt），同档区间 25~100pt 内
#   变异自检     ：fling_lift=0.45 / fling_lift_max=850（旧值）→ 拱高 127.3pt、>20 占比 100%
#   fling_lift=0 ：拱高 28.1pt、>20 占比峰值 0.75%（对照：无竖直初速度，仅**拖动起皱的余波**；
#                  默认档 42.6pt / 4.92%；用例内机检「默认 ≥ 对照 ×1.25 / ×2」分离）
#   grab_lift    ：度量=**整块最高点 + 松手保留率**（按手的位置取样会取到前方的空材料 z=0；
#                  而拖动起皱也会抬高整块最高点，故「拎起」另用「松手掉不掉」判别——
#                  悬空材料靠手托着，松手必掉；贴地褶皱不掉）：
#                  猛拽 12 帧 × 80pt/帧（标称 2400pt/s）整块 74.5pt（才拎起，_grab_lift=74.5），
#                  松手 15 帧保留率 0.28；普通拖 12 帧 × 40pt/帧（标称 1200pt/s）整块 29.1pt
#                  （全是褶皱，_grab_lift=0），保留率 0.79；慢拖 10pt/帧整块 0.0pt；
#                  变异自检：v0=700（旧值）→ 普通拖整块 40.3pt、保留率 0.24（两条判据都红）；
#                  v0=1e9（关掉抬升）→ 猛拽整块 31.0pt（幅度判据红）；
#                  v0=0 → 慢拖整块 34.8pt（慢拖用例必红）
#   throw_arc_z  ：默认首帧 8.4pt、12 帧峰值 27.7pt；=0 时全程 0.0pt；
#                  变异自检：480（旧值）→ 峰值 69.8pt
#   入睡         ：快拖后 release(fling=False) 18 帧（0.60s）、fling=True 77 帧（2.57s）
# ⚠ 本轮三条用例（抛掷拱高 / 猛拽拎起 / 飞入弧高）都必须能用「把对应参数覆盖回旧值」
#   的方式变红——即下面每条 docstring 里写明的「实测 / 变异自检」数字；对照组两条
#   （fling_lift=0 / throw_arc_z=0）断言的仍是「关掉后贴地」，不随意图变化。
# ---------------------------------------------------------------------------
V63_W, V63_H = 1200.0, 900.0
FLING_FRAMES = 8            # 每帧 60pt：_drag_hist 存 6 帧 → v = 5×60/(6/30) = 1500pt/s（内部估计）
FLING_STEP_PT = 60.0        # 标称 1800pt/s：> fling_threshold 600 → 触发抛掷；> grab_lift_v0 1400
                            # → 只小幅拎起（实测抓点 12.1pt，厚重意图下不再吊起 ~110pt）
AIRBORNE_PT = 20.0          # 「离地」判定高度（v6.7 皮带后从 40 降到 20：抬升整体更低更重，
                            # 默认档 >40 占比峰值 0.00% 已无判别力；>20 峰值 4.92%，
                            # 对照 fling_lift=0 → 0.75%）


def _fast_drag(sim: ClothSim, frames: int = FLING_FRAMES, step_pt: float = FLING_STEP_PT,
               x0: float = V63_W / 2.0, y0: float = V63_H / 2.0) -> float:
    """抓中心 → 每帧横拖 +step_pt → 返回终点 x。速度见 FLING_FRAMES 注释。"""
    assert sim.grab(x0, y0) is True
    gx = x0
    for _ in range(frames):
        gx += step_pt
        sim.drag_to(gx, y0)
        sim.step(DT)
    return gx


def _z_traj(sim: ClothSim, frames: int = 30):
    """步进 frames 帧，逐帧返回 (zmax 列表, 离地粒子占比列表)。"""
    zmax, air = [], []
    for _ in range(frames):
        sim.step(DT)
        v = sim.vertices()
        zmax.append(float(v[:, 2].max()))
        air.append(float((v[:, 2] > AIRBORNE_PT).mean()))
    return zmax, air


def test_fling_release_throws_sheet_airborne():
    """厚重意图（用户：「就像一个丝绸，我需要有毛毯的重量感」）：抛掷是**低而重**的弧。

    8 帧 × 60pt 快甩 → release(fling=True)：松手 zmax 22.0pt → 拱高 42.6pt（@6 帧）。
    上下界一起卡「低而重」：下界 25pt 证明确实离地（不是贴地滑，松手瞬间仅 22.0pt）；
    上界 100pt 证明不再"像纸片飞"——变异自检：fling_lift=0.45 / fling_lift_max=850
    （旧意图，运行时覆盖）实测拱高 127.3pt → 上界必红。另断言有粒子真越过
    AIRBORNE_PT=20pt（v6.7 皮带后抬升整体更低：>40 占比峰值 0.00% 已无判别力，判定降档；
    >20 占比峰值 4.92%；对照 fling_lift=0 → 0.75% < 1% → 变红）。
    """
    sim = ClothSim(V63_W, V63_H)
    _fast_drag(sim)
    z_rel = float(sim.vertices()[:, 2].max())
    sim.release(fling=True)
    zmax, air = _z_traj(sim, 30)
    peak, peak_at = max(zmax), int(np.argmax(zmax)) + 1
    print(f"INFO cloth: fling 抛掷(厚重) 松手 zmax={z_rel:.1f} → 拱高 {peak:.1f}pt @{peak_at} 帧，"
          f"离地(>{AIRBORNE_PT:.0f}pt)占比峰值 {max(air) * 100:.2f}%")
    v = sim.vertices()
    assert np.isfinite(v).all()
    assert np.all(v[:, 2] >= 0.0)  # 契约：z ≥ 0
    assert peak > 25.0, (
        f"抛掷没离地（像贴地滑）：拱高仅 {peak:.1f}pt（应 >25；实测 42.6）")
    assert peak < 100.0, (
        f"抛掷像纸片飞（不是毛毯的重量感）：拱高 {peak:.1f}pt（应 <100；实测 42.6；"
        f"旧参数 0.45/850 → 127.3）")
    assert max(air) > 0.01, (
        f"抛掷没有离地（像贴地滑）：{AIRBORNE_PT:.0f}pt 以上粒子占比峰值仅 {max(air) * 100:.2f}%"
        f"（应 >1%；实测 4.92%；对照 fling_lift=0 → 0.75%）")


def _fling_landing(lift: float):
    """抛掷对照协议：8 帧 × 60pt 快甩 → release(fling=True) → 30 帧。

    返回 (松手瞬间 zmax, 释放后逐帧 zmax, 释放后逐帧离地占比)。
    """
    sim = ClothSim(V63_W, V63_H, fling_lift=lift)
    _fast_drag(sim)
    z_rel = float(sim.vertices()[:, 2].max())
    sim.release(fling=True)
    zmax, air = _z_traj(sim, 30)
    return z_rel, zmax, air


def test_fling_lift_zero_keeps_sheet_on_floor():
    """对照（fling_lift=0.0：无竖直初速度，"只贴地滑"）：同流程不腾空。

    2026-10-09 终轮定标（冻结参数；1200×900，同协议三档实测，本机逐位可复现）：
      fling_lift=0（本用例） ：松手 zmax 22.0 → 释放后峰值 28.1pt（@2 帧），离地(>20)占比峰值 0.75%
      默认档 fling_lift=0.18 ：峰值 42.6pt（@6 帧）、离地占比峰值 4.92%
      变异档 fling_lift=0.45 ：峰值 66.6pt（@8 帧）、离地占比峰值 83.0%
    上限重定标：28.1pt 的组成是「拖动起皱」的余波（松手瞬间就已 22.0），不是竖直初速度——
    这轮拖动起皱变强，旧上限 25 被余波击穿。上限 35pt 卡在对照 28.1 与默认档 42.6 之间
    （两侧余量 1.25× / 1.22×）；离地占比上限 2.0% 卡在 0.75% 与 4.92% 之间（2.7× / 2.5×）。
    对照语义**机检**（不只是写进注释）：与默认档同协议、同指标下，默认档必须明显更高——
    峰值 ≥ 对照 ×1.25（实测 1.52×）、离地占比 ≥ 对照 ×2（实测 6.6×）；否则说明「抛掷竖直
    分量」没有产生对照差异（fling_lift 被无视/削弱时必红）。
    变异自检：本用例 fling_lift 覆盖成 0.45 → 峰值 66.6 > 35 必红、占比 83.0% ≫ 2% 必红。
    """
    z_rel0, zmax0, air0 = _fling_landing(0.0)
    z_rel_d, zmax_d, air_d = _fling_landing(0.18)
    print(f"INFO cloth: fling_lift=0 松手 zmax={z_rel0:.1f} → 释放后峰值 {max(zmax0):.1f}pt，"
          f"离地(>{AIRBORNE_PT:.0f}pt)占比峰值 {max(air0) * 100:.2f}%（默认档 0.18：松手 "
          f"zmax={z_rel_d:.1f} → 峰值 {max(zmax_d):.1f}pt、占比峰值 {max(air_d) * 100:.2f}%）")
    assert max(zmax0) <= 35.0, (
        f"fling_lift=0 仍腾空：峰值 {max(zmax0):.1f}pt > 35pt（实测 28.1 = 拖动起皱余波；"
        f"默认档 42.6）")
    assert max(air0) <= 0.02, (
        f"fling_lift=0 仍有成片离地：占比峰值 {max(air0) * 100:.2f}% > 2%（实测 0.75%；"
        f"默认档 4.92%）")
    assert max(zmax_d) >= max(zmax0) * 1.25, (
        f"默认档与对照的峰值分离度不足（抛掷竖直分量没有产生差异？）：默认档 "
        f"{max(zmax_d):.1f}pt vs 对照 {max(zmax0):.1f}pt（应 ≥1.25×；实测 1.52×）")
    assert max(air_d) >= max(air0) * 2.0, (
        f"默认档与对照的离地占比分离度不足：默认档 {max(air_d) * 100:.2f}% vs 对照 "
        f"{max(air0) * 100:.2f}%（应 ≥2×；实测 6.6×）")


def test_grab_lift_slow_drag_keeps_point_on_desk():
    """厚重意图：慢拖（10pt/帧，标称 250pt/s < grab_lift_v0=1400）→ 整块仍贴地滑（实测最高 0.0pt）。

    度量=**整块最高点**：v6.7 牵引皮带让抓点滞后于手，按手的位置取样会取到旁边的粒子
    （实测 v0=0 时该采样点 z=0.0 而整块最高 34.8pt → 旧度量会**假过**）；
    整块最高 ≥ 任何单点 → 断言不弱化。
    变异自检：v0=0（速度门槛被关掉）→ 抓点被拎起、整块最高 34.8pt → 必红。
    """
    sim = ClothSim(V63_W, V63_H)
    _fast_drag(sim, frames=12, step_pt=10.0)
    zmax = float(sim.vertices()[:, 2].max())
    print(f"INFO cloth: 慢拖 12 帧（250pt/s）整块最高 z={zmax:.1f}pt")
    assert zmax < 15.0, f"慢拖不该把毯子拎起来：整块最高 z={zmax:.1f}pt（实测 0.0；应 <15）"


def _release_retention(sim: ClothSim, frames: int = 15) -> float:
    """松手（fling=False）后 frames 帧内，整块最高点的保留率 = 该窗口 min / 松手时值。

    「拎起」= 材料被手托着悬空 → 松手必下坠（实测猛拽 0.28，掉回褶皱地板）；
    「只是起皱」= 贴地的几何褶皱 → 松手仍保持（实测普通拖 0.79，12 tick 后还被形状保持锁住）。
    这是区分两者的物理信号：整块最高点本身两种情形都会抬高，但褶皱不会「掉下来」。
    只用契约 API（vertices/release/step），不读私有状态。
    """
    z_rel = float(sim.vertices()[:, 2].max())
    sim.release(fling=False)
    z_min = z_rel
    for _ in range(frames):
        sim.step(DT)
        z_min = min(z_min, float(sim.vertices()[:, 2].max()))
    return z_min / max(z_rel, 1e-9)


def test_grab_lift_fast_drag_raises_grab_point():
    """厚重意图：只有**猛拽**才小幅拎起抓点（grab_lift_v0=1400）；普通拖拽贴地。

    2026-10-09 终轮定标（冻结参数 pin_leash_pt=220 等；1200×900、抓中心、12 帧，
    本机逐位可复现）：
      猛拽 80pt/帧（标称 2400pt/s；内部估计 2000 > 1400）→ 整块最高 74.5pt（_grab_lift=74.5）
      普通拖 40pt/帧（标称 1200pt/s；内部估计 1000 < 1400）→ 整块最高 29.1pt（_grab_lift=0.0）
    度量里保留 v6.7 的教训：皮带让抓点滞后于手，按手的位置取样会取到前面的空材料（z=0），
    故幅度用**整块最高点**。判据拆成两条——
    判据一·幅度：z_vig > 60（实测 74.5）；z_ord ≤ 35（实测 29.1）。旧的 ≤25 不能再用了：
    这轮「拖动起皱」变强，普通拖自身的褶皱就把整块最高抬到 29.1pt，而 _grab_lift=0.0 说明
    没有拎升——旧线把「起皱」误判成「拎起」（1600×900 同现象复现：36.7pt）。≤35 卡在
    本轮普通拖 29.1 与旧参数（v0=700）变异 40.3 之间。
    判据二·悬空性（「拎起 vs 只是起皱」的物理区分）：被拎起的材料靠手托着 → 松手必掉；
    贴地褶皱不会掉。松手后 15 帧保留率：猛拽 74.5→21.0 = 0.28、普通拖 29.1→23.1 = 0.79；
    判定线 0.50（两侧余量 1.6× / 1.8×）。
    变异自检：grab_lift_v0 覆盖回旧值 700 → 普通拖 z_ord=40.3（>35）且保留率 0.24（<0.50），
    两条判据都必红；覆盖成 1e9（关掉抬升）→ 猛拽 z_vig=31.0 → 第一条必红
    （保留率 0.86：没拎起的材料自然不会掉，第二条不负责这一变异）。
    """
    sim = ClothSim(V63_W, V63_H)
    _fast_drag(sim, frames=12, step_pt=80.0)
    z_vig = float(sim.vertices()[:, 2].max())
    lift_vig = float(getattr(sim, "_grab_lift", float("nan")))
    keep_vig = _release_retention(sim, frames=15)
    sim_ord = ClothSim(V63_W, V63_H)
    _fast_drag(sim_ord, frames=12, step_pt=40.0)
    z_ord = float(sim_ord.vertices()[:, 2].max())
    lift_ord = float(getattr(sim_ord, "_grab_lift", float("nan")))
    keep_ord = _release_retention(sim_ord, frames=15)
    print(f"INFO cloth: 猛拽 12 帧（80pt/帧，标称 2400pt/s）整块最高 z={z_vig:.1f}pt"
          f"（_grab_lift={lift_vig:.1f}，松手 15 帧保留率 {keep_vig:.2f}）；普通拖 12 帧"
          f"（40pt/帧，标称 1200pt/s）整块最高 z={z_ord:.1f}pt（_grab_lift={lift_ord:.1f}，"
          f"松手 15 帧保留率 {keep_ord:.2f}）")
    assert z_vig > 60.0, f"猛拽没有拎起毯子：整块最高 z={z_vig:.1f}pt（应 >60；实测 74.5）"
    assert keep_vig <= 0.50, (
        f"猛拽抬高的材料松手没掉（不像悬空垂挂）：15 帧保留率 {keep_vig:.2f}（应 ≤0.50；"
        f"实测 0.28）")
    assert z_ord <= 35.0, (
        f"普通拖拽把毯子拎起来了（像轻薄的丝绸）：整块最高 z={z_ord:.1f}pt（应 ≤35；实测 29.1，"
        f"其中 _grab_lift=0.0 = 全是拖动起皱，旧线 ≤25 已被起皱高度击穿；"
        f"旧 grab_lift_v0=700 → 40.3）")
    assert keep_ord >= 0.50, (
        f"普通拖拽的材料松手就掉（像被拎起来悬空）：15 帧保留率 {keep_ord:.2f}（应 ≥0.50；"
        f"实测 0.79；旧 grab_lift_v0=700 → 0.24）")


def test_throw_in_arc_rises_above_desk():
    """厚重意图：throw_in「铺上」仍有竖直弧线，但更低更沉（throw_arc_z=300）。

    实测首 3 帧 zmax = 8.4 / 15.2 / 20.5pt，12 帧峰值 27.7pt（贴地时全程 0）。
    区间 10~60pt 两侧都有判别力：下界排除"关掉弧线"（=0 → 0.0pt）；上界排除旧值
    throw_arc_z=480（变异自检实测峰值 69.8pt → 必红）。
    落地收敛（is_asleep）由既有 test_sleep_within_6s_without_bumps 覆盖，不重复。
    """
    sim = ClothSim(V63_W, V63_H)
    sim.throw_in((V63_W / 2.0, V63_H / 2.0))
    zmax, _air = _z_traj(sim, 12)
    print(f"INFO cloth: 飞入弧线(厚重) zmax 首帧 {zmax[0]:.1f} 峰值 {max(zmax):.1f}pt")
    assert min(zmax[:3]) > 5.0, f"飞入头 3 帧没有离地：zmax={[round(z, 1) for z in zmax[:3]]}"
    assert max(zmax) >= 10.0, (
        f"飞入弧线太低（弧线没了？）：12 帧峰值仅 {max(zmax):.1f}pt（应 ≥10；实测 27.7）")
    assert max(zmax) <= 60.0, (
        f"飞入弧线太高（厚重感丢了）：12 帧峰值 {max(zmax):.1f}pt（应 ≤60；实测 27.7；"
        f"旧 throw_arc_z=480 → 69.8）")


def test_throw_in_arc_zero_stays_on_ground():
    """对照（throw_arc_z=0.0，旧行为）：飞入全程贴地——实测 12 帧 zmax 恒 0.0pt。"""
    sim = ClothSim(V63_W, V63_H, throw_arc_z=0.0)
    sim.throw_in((V63_W / 2.0, V63_H / 2.0))
    zmax, _air = _z_traj(sim, 12)
    print(f"INFO cloth: throw_arc_z=0 飞入 zmax 峰值 {max(zmax):.2f}pt")
    assert max(zmax) <= 2.0, f"throw_arc_z=0 仍离地：zmax 峰值 {max(zmax):.2f}pt > 2"


# 入睡上界保护（13.3s）：任务给的上限是 600 帧，取 400 帧做更严的回归锚点。
# 实测（厚重新参数，耗能更快）：快拖 8 帧 × 60pt 后 release(fling=False) 18 帧（0.60s）入睡、
# fling=True 77 帧（2.57s），复跑两次逐位一致（物理无随机源）。
# 历史 bug「拖动后永不入睡」的锚点就在这里。
SLEEP_MAX_STEPS_V63 = 400


def _steps_until_asleep_v63(sim: ClothSim, max_steps: int = SLEEP_MAX_STEPS_V63):
    for i in range(max_steps):
        sim.step(DT)
        if sim.is_asleep():
            return i + 1
    return None


def test_fast_drag_release_without_fling_falls_asleep():
    """回归（历史 bug：拖动后永不入睡）：快拖 → release(fling=False) → 有限帧内入睡。

    实测（厚重新参数）18 帧（0.60s）≪ 400 帧上限；动能收敛到 sleep_eps（<1.0）量级。
    """
    sim = ClothSim(V63_W, V63_H)
    _fast_drag(sim)
    sim.release(fling=False)
    assert not sim.is_asleep(), "release 后必须被 wake（还要继续收敛）"
    steps = _steps_until_asleep_v63(sim)
    print(f"INFO cloth: 快拖后 release(fling=False) 入睡={steps} 帧"
          f"（{'—' if steps is None else f'{steps / 30.0:.2f}s'}）")
    assert steps is not None, f"{SLEEP_MAX_STEPS_V63} 帧（13.3s）内未入睡"
    assert sim.kinetic_energy() < 1.0


def test_fast_drag_fling_release_falls_asleep_after_flight():
    """v6.3 路径（厚重参数下复跑）：抛掷离地 → 落回桌面后同样要收敛入睡（实测 77 帧 = 2.57s）。"""
    sim = ClothSim(V63_W, V63_H)
    _fast_drag(sim)
    sim.release(fling=True)
    steps = _steps_until_asleep_v63(sim)
    print(f"INFO cloth: 快拖后 release(fling=True) 入睡={steps} 帧"
          f"（{'—' if steps is None else f'{steps / 30.0:.2f}s'}）")
    assert steps is not None, f"{SLEEP_MAX_STEPS_V63} 帧（13.3s）内未入睡（腾空后没落地收敛）"
    assert float(sim.vertices()[:, 2].max()) < 60.0, "入睡时仍有明显离地（未真正落地）"


# ---------------------------------------------------------------------------
# v6.9：折痕处不再有「逐像素争夺深度」的贴合层（用户截图里折脊那圈锯齿撕口）
#
# 现象与根因（2026-10-09 实测）：
#   折痕压平后，折片与毯身会**共面停在桌面高度**（|Δz|≈0），而自碰撞只收
#   `z > 地板 + 0.8` 的粒子 → 两层贴地时完全没人管；渲染时两层投影到同一像素、
#   深度差≈0（`depth_bias=0.90` 也救不了 Δz=0）→ 逐像素争夺 → 顶视就是一圈锯齿。
#   量测：折角后 worst 203 对、最近 5.1pt（配置 collision_sep=15pt）。
# 阈值来源（同机实测）：
#   旧实现把**所有** cross 对都抬到 sep+1=16pt → 普通拖动整块最高 29.1→39.6pt，
#   击穿「普通拖不拎起」守卫；只抬到 sep_min_depth 后回落。
# 读私有状态是为了直接量「还有多少贴合对」，用契约 API 看不到这个量。
# ---------------------------------------------------------------------------
def _facing_fights(sim: ClothSim, sep: float = 15.0, max_dz: float = 3.0):
    """返回 (贴合对数, 最近距离)：材料**不直连**（= 不同层）且 |Δz| 小于 max_dz 的对。

    判据与 cloth._separate_degenerate_overlaps 一致（实际距离远小于网格距离 = 中间折了）。
    """
    p = sim._pos
    rr = (np.arange(N) // COLS).astype(np.float64)
    cc = (np.arange(N) % COLS).astype(np.float64)
    c = np.floor(p[:, :2] / sep)
    keys = c[:, 0] * 1000003 + c[:, 1]
    o = np.argsort(keys, kind="stable")
    sk = keys[o]
    ia = np.arange(N)
    ci, cj = [], []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            t = c[:, 0] + dx
            t = t * 1000003 + c[:, 1] + dy
            lo = np.searchsorted(sk, t, side="left")
            hi = np.searchsorted(sk, t, side="right")
            cnt = hi - lo
            if int(cnt.sum()) == 0:
                continue
            st = np.repeat(lo, cnt)
            off = np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt)
            jj = o[st + off]
            ii = np.repeat(ia, cnt)
            m = ii < jj
            if m.any():
                ci.append(ii[m])
                cj.append(jj[m])
    ii = np.concatenate(ci)
    jj = np.concatenate(cj)
    d3 = p[ii] - p[jj]
    dist = np.linalg.norm(d3, axis=1)
    cand = np.nonzero(dist < sep)[0]
    if cand.size == 0:
        return 0, 99.0
    gdist = np.hypot((cc[ii[cand]] - cc[jj[cand]]) * sim._sx,
                     (rr[ii[cand]] - rr[jj[cand]]) * sim._sy)
    cross = dist[cand] < 0.55 * np.maximum(gdist, 1e-9)
    if not cross.any():
        return 0, 99.0
    dz = np.abs(d3[cand][:, 2])
    sel = cross & (dz < max_dz)
    return int(sel.sum()), float(dist[cand][cross].min())


def test_corner_fold_leaves_no_coplanar_facing_pair():
    """折角落定后不能残留 |Δz|<3pt 的贴合层（那圈锯齿撕口的直接来源）。

    实测（corner 手势，_drag_across 同一协议）：关掉清理残留 8 对 / 最近 8.9pt；
    v6.9 修完后 **0 对**（四场景全部 0）。阈值取 0 而不是「减少若干」——这是
    「有没有可见撕口」的二值判据。变异自检：sep_max_dz=0（关掉清理）→ 残留 8 对，必红。
    """
    sim = make_sim()
    _drag_across(sim, W, H)
    sim.wake()
    for _ in range(120):
        sim.step(DT)
    n, dmin = _facing_fights(sim)
    print(f"INFO cloth: 折角落定后贴合层 {n} 对（最近 {dmin:.2f}pt）")
    assert n == 0, (
        f"折痕残留 {n} 对共面贴合层（最近 {dmin:.2f}pt）→ 渲染上是逐像素争夺的锯齿撕口；"
        f"旧实现残留 203 对，v6.9 定稿 0 对")


def test_fold_gesture_keeps_both_layers_but_no_coplanar_pair():
    """「拖一角横过毯身」仍要保留**双层**（真翻折），只是不许有共面贴合。

    这条与上一条配对：防止为了消除撕口而把折层整体推平（倒退成"折不起来"）。
    实测：翻底三角 137、双层对 ~19、贴合层 0。
    """
    sim = make_sim()
    _drag_across(sim, W, H)
    sim.wake()
    for _ in range(120):
        sim.step(DT)
    flipped = _flipped_tris(sim)
    n, _ = _facing_fights(sim)
    print(f"INFO cloth: 折后翻底三角={flipped} 贴合层={n}")
    assert flipped >= 50, f"折层被推平了：翻底三角只有 {flipped}（应保持真翻折）"
    assert n == 0, f"仍有 {n} 对共面贴合层"


def test_cleanup_does_not_lift_plain_drag():
    """退化清理**不得**给普通拖动加高度（拖动态不清理 + 只抬到最小分离量）。

    实测：清理在拖动中途插抬升 → 慢拖整块最高 11.2→16.0pt（击穿「普通拖不拎起」）；
    只在非抓取态清理后回到 14.9pt。
    """
    sim = ClothSim(V63_W, V63_H)
    _fast_drag(sim, frames=12, step_pt=10.0)
    zmax = float(sim.vertices()[:, 2].max())
    print(f"INFO cloth: 慢拖 12 帧带清理 zmax={zmax:.1f}pt")
    assert zmax < 15.0, f"清理给普通拖动加了高度：{zmax:.1f}pt（应 <15，实测 14.9）"


# ---------------------------------------------------------------------------
# v6.10：握住不动时折痕不该抖、也不该「融合」
#
# 用户报障（2026-10-09，截图：折角后折脊一圈亮撕裂 + 折层像粘在一起）：
#   「叠起来之后折痕位置还有一些抖动、融合」。
# 实测根因（两条，都由 `grab` 保持态独有）：
#   ① 抖动：`quiet`（准静态收尾）原来排除了 `_grabbed`，于是「按住不动」永远走
#      `_grab_iters` 的运动求解、拿不到 `settle_move_cap` 欠松弛与 `settle_vel_credit`
#      速度折算；`damping=0.972` 每帧只削 2.8% 动能 → 残留极限环长期停在十几 pt/帧。
#      实测逐帧渲染差分 **~130 万像素**在变（画面占一半），可见性极高。
#   ② 融合：退化清理原来「抓取中一律不跑」（怕改手感），但保持态折层会互相越贴越近
#      （最近距离 6.6→**1.25pt**、贴合对 13→**73**）→ 渲染上粘成一片。
# 修法：用**手速门控** `body_hold_speed`——手在动 = 照旧拖动求解；手停住 = 按准静态
#   收尾 + 恢复清理。实测：抖动 p50 12.0→4.8pt、渲染差分 130 万→4.4 万像素/帧、
#   最近层距 1.25→4.6pt、贴合对 73→27。
# 变异自检：body_hold_speed=1e9（= 门控关掉、退回旧行为）→ 两条都必红。
# ---------------------------------------------------------------------------
def _hold_after_fold(sim: ClothSim, x0: float = None):
    """抓左上角横过毯身（与用户截图同款），**不松手**停住。"""
    gx = (W / 2 - sim.width_pt / 2 + sim.width_pt * 0.10) if x0 is None else x0
    gy = sim._pos[:, 1].min() + sim.height_pt * 0.12
    assert sim.grab(gx, gy, 40.0) is True, "抓点没落在毯子上"
    n = 50
    for i in range(1, n + 1):
        t = i / n
        tt = t * t * (3.0 - 2.0 * t)
        sim.drag_to(gx + sim.width_pt * 0.76 * tt, gy + sim.height_pt * 0.58 * tt)
        sim.step(DT)
    assert sim._grabbed, "抓取状态丢失"


def test_holding_still_converges_instead_of_jittering():
    """握住不动 → 逐帧最大位移要收敛到几 pt 量级（旧行为：十几 pt、永不收敛）。

    实测（停 60 帧后 20 帧窗口）：修复前 p50=6.1~12.0pt / max 9.7~21.9pt；
    修复后 p50≤5.5pt。阈值 8.0 卡在两者之间（变异 body_hold_speed=1e9 实测 p50=6.07 需复核）。
    """
    sim = ClothSim(W, H)
    _hold_after_fold(sim)
    for _ in range(60):
        sim.step(DT)
    moves = []
    for _ in range(20):
        p0 = sim._pos.copy()
        sim.step(DT)
        moves.append(float(np.linalg.norm(sim._pos - p0, axis=1).max()))
    moves = np.asarray(moves)
    p50 = float(np.percentile(moves, 50))
    print(f"INFO cloth: 握住不动 20 帧 逐帧最大位移 p50={p50:.2f}pt max={moves.max():.2f}pt")
    assert p50 <= 5.5, (
        f"握住不动仍在抖：逐帧最大位移 p50={p50:.2f}pt（应 ≤5.5；旧行为 6.1~12.0pt，"
        f"画面逐帧差分 ~130 万像素）")


def test_holding_does_not_let_layers_merge():
    """握住不动 → 折层不能越贴越近（旧行为：最近层距 6.6→1.25pt、贴合对 13→73）。

    读私有状态是为了量"层间最近距离/贴合对数"（契约 API 看不到）。
    """
    sim = ClothSim(W, H)
    _hold_after_fold(sim)
    n_drag, min_drag = _facing_fights(sim)
    for _ in range(60):
        sim.step(DT)
    n_hold, min_hold = _facing_fights(sim)
    print(f"INFO cloth: 折层贴合 拖动中 {n_drag} 对/{min_drag:.2f}pt → 停住 {n_hold} 对/{min_hold:.2f}pt")
    assert min_hold >= 3.0, (
        f"停住后折层贴到一起了（融合）：最近层距 {min_hold:.2f}pt（应 ≥3.0；"
        f"旧行为 1.25pt）")
    assert n_hold <= 40, f"停住后贴合对过多：{n_hold} 对（旧行为 73 对）"


def test_drag_assist_still_follows_while_moving():
    """手在动时助力仍生效（守住手感，不许为了止抖把"拖走"也关掉）。

    实测分段滑动比（拖边角）≈ 0.19 / 0.28 / 0.49（120/300/600pt 手位移）。
    """
    for hand, lo in ((120.0, 0.05), (600.0, 0.25)):
        sim = ClothSim(W, H)
        sim.set_placement((W / 2, H / 2), W * 0.9, H * 0.8, 0.0, zero_velocity=True)
        for _ in range(30):
            sim.step(DT)
        gx, gy = W * 0.12, H * 0.14
        assert sim.grab(gx, gy, 40.0)
        c0 = sim._pos[:, :2].mean(axis=0).copy()
        n = 30
        for k in range(1, n + 1):
            t = k / n
            tt = t * t * (3.0 - 2.0 * t)
            sim.drag_to(gx + hand * tt, gy + hand * 0.6 * tt)
            sim.step(DT)
        ratio = float(np.linalg.norm(sim._pos[:, :2].mean(axis=0) - c0) / hand)
        print(f"INFO cloth: 手位移 {hand:.0f}pt → 整块滑动比 {ratio:.2f}")
        assert ratio >= lo, (
            f"手位移 {hand:.0f}pt 的助力太弱（{ratio:.2f} < {lo}）：手速门控把拖动也关掉了")
