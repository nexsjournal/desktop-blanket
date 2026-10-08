"""tests/test_overlay.py — overlay.py 回归测试：穿透闸门 + hitTest 坐标换算 + 命中语义。

锁定的规范来源（不按当前实现断言）：
  - 真机路由实测结论：`NSView.hitTest_` 返回 None **只吞事件、不穿透**；真正生效的
    「毯外点击穿透」机制是 `NSWindow.setIgnoresMouseEvents_` 的动态闸门
    （`gate_wants_ignore`，纯函数，阈值与 hitTest 的 40pt 完全一致、无死区）。
  - integrator 任务书 §1（`hitTest_` 恒 None 修复 + `_sim_xy_from_window_point`）
    与 §9a（`gate_wants_ignore(dist_pt, currently_ignoring, grabbed, foreign_press)`）。

⚠️ 并行说明：`gate_wants_ignore` 由 integrator 并行新增，未落地时本文件相关用例
`pytest.skip`（明确提示）；其余按规范断言——规范要求而实现暂时做不到的用例现在
就是红的，这是预期的回归锚点，不要为变绿放宽断言。

离屏构造：`RugRootView.alloc().initWithFrame_` 不需要窗口/面板；假 owner 只提供
`_sim` 与 `is_shown()` 两个成员（hitTest_ 的最小读取面）。
"""
from __future__ import annotations

import numpy as np
import pytest

overlay = pytest.importorskip("overlay", reason="overlay.py 依赖 AppKit/SceneKit")
cloth = pytest.importorskip("cloth")
NSMakeRect = pytest.importorskip("Foundation").NSMakeRect

SCREEN_W, SCREEN_H = 2560.0, 1440.0   # overlay 的离屏视图尺寸（真机主屏量级）
SIM_W, SIM_H = 800.0, 600.0            # 比屏幕小的真 ClothSim（毯外区域真实存在）
INNER_X, INNER_Y = 400.0, 300.0        # 毯内基准点（取最近粒子，见 _inner_points）
OUT_X, OUT_Y = 2200.0, 1400.0          # 毯外远点（最近距离 > 40pt）


# ---------------------------------------------------------------------------
# 公共构件
# ---------------------------------------------------------------------------
class _FakeOwner:
    """RugOverlay 的最小替身：hitTest_ 只读 `_sim` 与 `is_shown()`。"""

    def __init__(self, sim, shown: bool = True):
        self._sim = sim
        self._shown = shown

    def is_shown(self) -> bool:
        return self._shown


def _make_view():
    return overlay.RugRootView.alloc().initWithFrame_(
        NSMakeRect(0.0, 0.0, SCREEN_W, SCREEN_H))


def _make_hit_setup():
    """离屏 view + 真 ClothSim(800x600) + 假 owner（已挂上 _owner）。"""
    view = _make_view()
    sim = cloth.ClothSim(SIM_W, SIM_H)
    owner = _FakeOwner(sim)
    view._owner = owner
    return view, sim, owner


def _inner_points(sim):
    """毯内一个已知粒子：返回 (窗口坐标点, sim 坐标点)。

    粒子坐标直接取 `sim.vertices()` 里离 (400, 300) 最近的粒子，则
    窗口点 = (sim_x, SCREEN_H - sim_y) 对应的最近距离 ≈ 0（严格 ≤40pt 命中）。
    """
    v = np.asarray(sim.vertices())
    d2 = (v[:, 0] - INNER_X) ** 2 + (v[:, 1] - INNER_Y) ** 2
    i = int(np.argmin(d2))
    sx, sy = float(v[i, 0]), float(v[i, 1])
    return (sx, SCREEN_H - sy), (sx, sy)


def _gate():
    fn = getattr(overlay, "gate_wants_ignore", None)
    if fn is None:
        pytest.skip("gate_wants_ignore 未实现（integrator 并行中）")
    return fn


# ---------------------------------------------------------------------------
# A. gate_wants_ignore 真值表（纯函数；参数顺序 = 签名
#    gate_wants_ignore(dist_pt, currently_ignoring, grabbed, foreign_press)）
#    语义：grabbed → False（永远收鼠标）；foreign_press 且未 grabbed → 冻结为
#    currently_ignoring；否则 → dist_pt > 40.0（毯外穿透 / ≤40 收鼠标）。
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "dist_pt, currently_ignoring, grabbed, foreign_press, expected",
    [
        (5.0, False, True, False, False),     # 抓取中 → 收鼠标（距离无关）
        (999.0, False, True, False, False),   # 抓取中 → 收鼠标（抓取优先于毯外）
        (999.0, False, False, True, False),   # 外来按下 → 冻结：保持当前 False
        (5.0, True, False, True, True),       # 外来按下 → 冻结：保持当前 True
        (41.0, False, False, False, True),    # 毯外（>40）→ 穿透
        (39.9, True, False, False, False),    # 毯内（<40）→ 收鼠标
        (40.0, False, False, False, False),   # ≤40 收鼠标（与 hitTest 阈值一致）
        (40.1, True, False, False, True),     # 刚过阈值 → 穿透
    ],
    ids=[
        "grabbed-near->False",
        "grabbed-far->False",
        "foreign-press-keeps-False",
        "foreign-press-keeps-True",
        "41pt->True",
        "39.9pt->False",
        "40pt->False",
        "40.1pt->True",
    ],
)
def test_gate_wants_ignore_truth_table(dist_pt, currently_ignoring, grabbed,
                                       foreign_press, expected):
    gate = _gate()
    got = gate(dist_pt, currently_ignoring, grabbed, foreign_press)
    assert got is expected, (
        f"gate_wants_ignore({dist_pt}, {currently_ignoring}, {grabbed}, "
        f"{foreign_press}) = {got!r}，规范要求 {expected!r}")


def test_gate_wants_ignore_parameter_names_match_spec():
    """位置真值表的含义取决于参数顺序：用关键字调用核对签名名与 §9a 逐字一致。"""
    gate = _gate()
    try:
        got = gate(dist_pt=41.0, currently_ignoring=False, grabbed=False,
                   foreign_press=False)
    except TypeError as exc:
        pytest.fail(
            "gate_wants_ignore 关键字签名与规范不一致"
            f"（应为 dist_pt/currently_ignoring/grabbed/foreign_press）：{exc}")
    assert got is True


# ---------------------------------------------------------------------------
# B. RugRootView._sim_xy_from_window_point：窗口坐标（bottom-left）
#    → sim 坐标（主屏本地 top-left、y 向下）
# ---------------------------------------------------------------------------
def test_sim_xy_from_window_point_flips_y():
    view = _make_view()
    x, y = view._sim_xy_from_window_point((100.0, 200.0))
    assert x == pytest.approx(100.0)
    assert y == pytest.approx(1240.0)  # 1440 - 200：bottom-left → top-left


def test_sim_xy_from_window_point_corners():
    view = _make_view()
    x, y = view._sim_xy_from_window_point((0.0, SCREEN_H))  # 窗口左上角
    assert x == pytest.approx(0.0)
    assert y == pytest.approx(0.0)
    x, y = view._sim_xy_from_window_point((0.0, 0.0))       # 窗口左下角
    assert x == pytest.approx(0.0)
    assert y == pytest.approx(SCREEN_H)


# ---------------------------------------------------------------------------
# C. RugRootView.hitTest_：毯上（≤40pt）返回 self、毯外 None、
#    未显示 / sim 缺失 → None（异常路径不抛）——该实现曾恒返回 None（抓不住毯子）
# ---------------------------------------------------------------------------
def test_hit_test_inside_rug_returns_view():
    view, sim, _owner = _make_hit_setup()
    p, (sx, sy) = _inner_points(sim)
    assert sim.nearest_distance(sx, sy) <= 40.0  # 前提：该点确在毯上
    assert view.hitTest_(p) is view


def test_hit_test_far_outside_rug_returns_none():
    view, sim, _owner = _make_hit_setup()
    assert sim.nearest_distance(OUT_X, OUT_Y) > 40.0  # 前提：最近距离 >40pt
    p = (OUT_X, SCREEN_H - OUT_Y)
    assert view.hitTest_(p) is None


def test_hit_test_hidden_owner_returns_none():
    view, sim, owner = _make_hit_setup()
    p, _ = _inner_points(sim)
    owner._shown = False
    assert view.hitTest_(p) is None


def test_hit_test_missing_sim_returns_none():
    view, sim, owner = _make_hit_setup()
    p, _ = _inner_points(sim)
    owner._sim = None
    assert view.hitTest_(p) is None


def test_hit_test_y_flip_direction_guard():
    """y 翻转方向守卫：把毯内粒子的 sim 坐标原样当窗口坐标喂进去必须落空。

    规范换算下窗口点 (sim_x, sim_y) → sim y = SCREEN_H - sim_y ≈ 1140，
    远在毯子（y ≤ 600）之外 → None；若实现漏翻/翻错，本用例与
    test_hit_test_inside_rug_returns_view 必有一条失败。
    """
    view, sim, _owner = _make_hit_setup()
    _p, (sx, sy) = _inner_points(sim)
    assert sim.nearest_distance(sx, SCREEN_H - sy) > 40.0  # 前提：错向点确在毯外
    assert view.hitTest_((sx, sy)) is None


def test_overlay_cloth_geometry_has_material():
    """回归（2026-10-08 真机踩坑）：逐帧新建的 SCNGeometry 必须挂上材质。

    macOS 26 上「无材质的 SCNGeometry 完全不绘制」（不报错、不警告、整窗全透明）——
    演示里的表现就是「毯子可交互（抓得住）但屏幕上什么都看不见」（用户报障）。
    此测试真机 show() 一次（建的是全透明面板），断言布面/伪阴影几何的材质非空。
    """
    pytest.importorskip("overlay")
    from AppKit import NSApplication, NSScreen
    screen = NSScreen.mainScreen()
    if screen is None:
        pytest.skip("无可用屏幕")
    NSApplication.sharedApplication()
    import os
    cloth = pytest.importorskip("cloth")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tex = os.path.join(root, "assets", "rugs", "rug-01.png")
    ov = overlay.RugOverlay(screen, tex,
                            lambda w, h: cloth.ClothSim(w * 0.72, h * 0.80), fps=30)
    try:
        ov.show()
        for name, node in (("布面", ov._cloth_node), ("伪阴影", ov._shadow_node)):
            geo = node.geometry()
            assert geo is not None, f"{name} 节点没有几何"
            mats = geo.materials()
            assert mats is not None and len(mats) >= 1, \
                f"{name} 几何没有材质 → macOS 26 下不会被绘制（全透明）"
            diff = mats[0].diffuse()
            assert diff is not None and diff.contents() is not None, f"{name} 材质 diffuse 为空"
    finally:
        ov.hide()


def test_camera_projection_maps_scene_to_screen_1to1():
    """回归（2026-10-08 真机踩坑）：相机必须真正俯视 90°，场景→屏幕 1:1。

    SCNNode.rotation 是 SCNVector4 (x, y, z, 角度弧度)（角度在第四位）。旧实现写成
    (-π/2, 0, 0, 1) → 被解释为「轴=(-π/2,0,0)、角=1 rad」→ 相机只俯 57.3°，画面被
    斜视压扁（cos32.7°≈0.84）且整体下移：渲染的毯子和交互用的 sim 坐标不是同一坐标系，
    拖动手感完全错位。此测试用 SCNSceneRenderer.projectPoint_ 直接验证映射：
    场景 (x, 0, z) 必须投影到视图 (x, h - z)。
    """
    pytest.importorskip("overlay")
    from AppKit import NSApplication, NSScreen
    screen = NSScreen.mainScreen()
    if screen is None:
        pytest.skip("无可用屏幕")
    NSApplication.sharedApplication()
    import os as _os
    cloth = pytest.importorskip("cloth")
    root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    f = screen.frame()
    w, h = float(f.size.width), float(f.size.height)
    ov = overlay.RugOverlay(screen, _os.path.join(root, "assets", "rugs", "rug-01.png"),
                            lambda ww, hh: cloth.ClothSim(ww * 0.72, hh * 0.80), fps=30)
    try:
        ov.show()
        scn = ov._scn
        for x, z in ((0.0, 0.0), (w, h), (w / 2.0, h / 2.0), (w * 0.25, h * 0.75)):
            p = scn.projectPoint_((x, 0.0, z))
            assert abs(p.x - x) <= 1.0, f"x 映射偏差：scene x={x} → view x={p.x}"
            assert abs(p.y - (h - z)) <= 1.0, f"y 映射偏差：scene z={z} → view y={p.y}（期望 {h - z}）"
    finally:
        ov.hide()
