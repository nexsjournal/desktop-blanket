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


# ---------------------------------------------------------------------------
# v5：贴图留白检测（§12.17）——「毯子四周一圈白框」的修复点
# 合成贴图受 TEXTURE_MARGIN_MAX_FRAC=6% 上限保护：留白比上限还厚时只裁到上限，
# 绝不把画面内容当成边距吃掉；深色内容贴图必须完全不裁。
# ---------------------------------------------------------------------------
def _make_rep(w: int, h: int, border_px: int, border_rgb, content_rgb):
    """造一张 (h,w) RGBA 合成贴图：外圈 border_px 像素为 border_rgb，其余 content_rgb。"""
    NSData = pytest.importorskip("Foundation").NSData
    NSBitmapImageRep = pytest.importorskip("AppKit").NSBitmapImageRep
    NSDeviceRGBColorSpace = pytest.importorskip("AppKit").NSDeviceRGBColorSpace
    rep = NSBitmapImageRep.alloc().\
        initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_(
            None, w, h, 8, 4, True, False, NSDeviceRGBColorSpace, 0, 0)
    buf = np.frombuffer(rep.bitmapData(), dtype=np.uint8, count=rep.bytesPerRow() * h)
    arr = buf.reshape(h, rep.bytesPerRow() // 4, 4)[:, :w, :]
    arr[:, :, :3] = content_rgb
    arr[:, :, 3] = 255
    if border_px > 0:
        arr[:border_px, :, :3] = border_rgb
        arr[h - border_px:, :, :3] = border_rgb
        arr[:, :border_px, :3] = border_rgb
        arr[:, w - border_px:, :3] = border_rgb
    return rep


def test_texture_margin_detected_and_capped():
    """白色留白被检测到（含保险内缩），且不超过 6% 上限；深色内容图零裁切。"""
    w = h = 400
    cap = int(w * overlay.TEXTURE_MARGIN_MAX_FRAC)
    rep = _make_rep(w, h, 10, (255, 255, 255), (60, 40, 40))     # 10px 纯白边
    t, l, b, r = overlay._detect_texture_margin(rep)
    print(f"INFO overlay: 白边贴图裁切 (t,l,b,r)=({t},{l},{b},{r})，上限 {cap}")
    for v in (t, l, b, r):
        assert 10 <= v <= cap, f"裁切量 {v} 越界：应 ≥ 实际白边 10px 且 ≤ 6% 上限 {cap}"

    rep_dark = _make_rep(w, h, 0, (255, 255, 255), (60, 40, 40))
    assert overlay._detect_texture_margin(rep_dark) == (0, 0, 0, 0), "无留白贴图不应被裁"

    rep_thick = _make_rep(w, h, 60, (255, 255, 255), (60, 40, 40))   # 15% 厚白边
    t2, l2, b2, r2 = overlay._detect_texture_margin(rep_thick)
    assert max(t2, l2, b2, r2) <= cap, "厚留白必须被 6% 上限挡住（防吃内容）"


# ---------------------------------------------------------------------------
# v6「厚度」：滚边 / 接触阴影 / 静止浮雕 / 灯光重配（§12.18）
# 用户报「感觉没有厚度，就像一个薄纸片」。以下用例锁住四件事：
#   ① 边界环拓扑正确（滚边与阴影裙都挂在它上面）；
#   ② 滚边的外法线方向朝毯外、且剖面真的「向外+向下」卷；
#   ③ 灯光重配后的明暗斜率（浅起伏能看出明暗）达到设计下限；
#   ④ 浮雕场有界、低频（不得退化成格点噪声）。
# 真机像素级验收（滚边亮度、阴影 alpha、浮雕贡献）在 scripts/look_demo.py 的
# 「厚度」指标里，本文件只锁**不依赖窗口**的几何/常量不变量。
# ---------------------------------------------------------------------------
def _v6_overlay(sim_w=800.0, sim_h=600.0):
    """离屏 overlay + 平铺 ClothSim（不需要真实屏幕交互；show/hide 成对）。"""
    pytest.importorskip("overlay")
    from AppKit import NSApplication, NSScreen
    screen = NSScreen.mainScreen()
    if screen is None:
        pytest.skip("无可用屏幕")
    NSApplication.sharedApplication()
    import os as _os
    cloth = pytest.importorskip("cloth")
    root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    sim = cloth.ClothSim(sim_w, sim_h)
    ov = overlay.RugOverlay(screen, _os.path.join(root, "assets", "rugs", "rug-01.png"),
                            lambda ww, hh: sim, fps=30)
    return ov, sim


def test_boundary_loop_is_the_grid_perimeter():
    """边界环 = 网格四边一圈（顺序闭合、无重复），滚边/阴影裙的顶点数随之确定。"""
    ov, sim = _v6_overlay()
    try:
        ov.show()
        rows, cols = ov._rows, ov._cols
        loop = ov._bloop
        assert loop is not None
        assert loop.size == 2 * (cols + rows) - 4, "边界环长度应为周长"
        assert len(set(loop.tolist())) == loop.size, "边界环有重复顶点"
        perim = set(np.concatenate([[r * cols + c for c in range(cols)]
                                    for r in (0, rows - 1)]).tolist()) | \
            set(np.concatenate([[r * cols + c for r in range(rows)]
                                for c in (0, cols - 1)]).tolist())
        assert set(loop.tolist()) == perim, "边界环与网格周界不一致"
        # 滚边/阴影裙顶点数 = 环长 × 环数（布局 index = k*rings + j）
        assert ov._rim_v.shape == (loop.size * len(overlay.EDGE_ROLL_ANGLES), 3)
        assert ov._sbuf.shape == (loop.size * len(overlay.SHADOW_SKIRT_PT), 3)
    finally:
        ov.hide()


def test_rim_rolls_outward_and_down():
    """滚边剖面必须「向外 + 向下」：最外环越出布面轮廓、并沉到布面之下。

    方向用平铺态判定：外法线应远离毯心（把 -x 侧边界点取出来，看滚边是否更靠 -x）。
    """
    ov, sim = _v6_overlay()
    try:
        ov.show()
        ov._rebuild_geometry()
        rows, cols, rings = ov._rows, ov._cols, ov._rim_rings
        rim_v, rim_n = ov._rim_v, ov._rim_n
        cloth = ov._vbuf
        p = sim.vertices()
        # 平铺在 (0,0)-(w,h) 左上角：取左边中段（row=rows//2, col=0）→ 外法线应为 -x
        k = int(np.where(ov._bloop == (rows // 2) * cols + 0)[0][0])
        outer = rim_v[k * rings + (rings - 1)]          # 最外环（剖面角 90°）
        inner = cloth[(rows // 2) * cols + 0]
        assert outer[0] < inner[0] - 1.0, "滚边最外环没有向毯外（-x）卷出"
        assert outer[1] < inner[1] - 1.0, "滚边最外环没有向下沉（场景 y = 高度）"
        assert abs(outer[0] - inner[0]) <= overlay.CLOTH_THICKNESS_PT + 0.5, \
            "滚边横向超出厚度上限"
        n_outer = rim_n[k * rings + (rings - 1)]
        assert n_outer[0] < -0.9, f"最外环法线应近水平朝外，实测 {n_outer}"
        n_first = rim_n[k * rings + 0]
        assert n_first[1] > 0.9, f"首环法线应近竖直朝上（与布面法线衔接），实测 {n_first}"
    finally:
        ov.hide()


def test_contact_shadow_skirt_alpha_fades_outward():
    """接触阴影裙：贴边 alpha 最大、外圈淡出；且首环内缩（藏在布下，不漏亮缝）。"""
    ov, sim = _v6_overlay()
    try:
        ov.show()
        ov._rebuild_geometry()
        sk = ov._skirt_rings
        alphas = [float(ov._scbuf[j::sk, 3].max()) for j in range(sk)]
        assert alphas[0] > alphas[1] > alphas[2] >= alphas[3], f"阴影 alpha 未单调淡出：{alphas}"
        assert alphas[0] <= 0.5, "贴边 alpha >0.5 会被算进 coverage 统计（应留在阴影量级）"
        assert overlay.SHADOW_SKIRT_PT[0] < 0.0, "首环必须内缩到布下（否则贴边出现亮缝）"
        assert float(ov._scbuf[:, 3].min()) >= 0.0, "阴影 alpha 越界"
    finally:
        ov.hide()


def test_shadow_direction_opposite_the_light():
    """阴影方向 = 光源水平来向的反方向；lean = 1/tan(仰角)（纯函数，无窗口）。"""
    d, lean = overlay._shadow_basis_from(overlay.SUN_TO_LIGHT)
    lx, ly, lz = overlay.SUN_TO_LIGHT
    assert d[0] * lx < 0.0 and d[1] * lz < 0.0, "阴影方向没有落在光的反侧"
    assert abs(d[0] * d[0] + d[1] * d[1] - 1.0) < 1e-9, "阴影方向未归一化"
    assert 0.5 < lean < 2.0, f"lean={lean} 不在合理范围（仰角 45° 附近应 ≈1）"


def test_light_orientation_round_trip_without_euler():
    """四元数朝向 → SceneKit 实测传播方向 = -SUN_TO_LIGHT（锁住不猜欧拉约定）。"""
    pytest.importorskip("overlay")
    import SceneKit
    from AppKit import NSApplication
    NSApplication.sharedApplication()
    q = overlay.orientation_for_to_light(overlay.SUN_TO_LIGHT)
    node = SceneKit.SCNNode.node()
    node.setOrientation_(q)
    scene = SceneKit.SCNScene.scene()
    scene.rootNode().addChildNode_(node)
    d = node.convertVector_toNode_((0.0, 0.0, -1.0), scene.rootNode())
    got = np.array([float(d[0]), float(d[1]), float(d[2])])
    want = -np.array(overlay.SUN_TO_LIGHT, dtype=np.float64)
    # SceneKit 内部用 float32（实测回程误差 ~3e-6）→ 容差 1e-4
    assert float(np.abs(got - want).max()) < 1e-4, f"朝向回程偏差：{got} vs {want}"


def test_lights_rebalanced_for_visible_relief():
    """灯光重配的量化下限：平铺面总亮不变、但明暗斜率（浅起伏可见度）显著提高。

    判据（线性域）：响应 = amb_lin + sun_lin·sin(仰角) ≈ 0.86（与原版一致，不整体变暗）；
    斜率 = sun_lin·|水平分量| ≥ 0.6（原版 0.85 sRGB → 线性 0.694×0.707 ≈ 0.49）。
    NSColor 是 sRGB：SceneKit 会转线性，所以常量比对必须走 sRGB→线性换算。
    """
    def s2l(v):
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
    lx, ly, lz = overlay.SUN_TO_LIGHT
    n = (lx * lx + ly * ly + lz * lz) ** 0.5
    lx, ly, lz = lx / n, ly / n, lz / n
    horiz = (lx * lx + lz * lz) ** 0.5
    amb, sun = s2l(overlay.AMBIENT_WHITE), s2l(overlay.SUN_WHITE)
    resp = amb + sun * ly
    slope = sun * horiz
    print(f"INFO overlay: 平铺响应={resp:.3f} 明暗斜率={slope:.3f}（原版 ≈0.86 / 0.49）")
    assert 0.78 <= resp <= 0.95, f"平铺面总亮偏离原版太多：{resp:.3f}"
    assert slope >= 0.60, f"明暗斜率不足：{slope:.3f}（浅起伏会看不出来）"
    assert amb < 0.25, "环境光线性占比过高会把明暗差压平"


def test_relief_field_bounded_and_low_frequency():
    """静止微浮雕：振幅有界；沿网格的二阶差分足够小（低频，不是「揉纸」碎纹）。"""
    ov, sim = _v6_overlay()
    try:
        ov.show()
        rel = np.asarray(ov._relief, dtype=np.float64)
        n = rel.size
        assert rel.shape == (n,)
        amp_sum = float(sum(overlay.RELIEF_AMP_PT))
        assert float(np.abs(rel).max()) <= amp_sum + 1e-6, "浮雕振幅越界"
        rows, cols = ov._rows, ov._cols
        g = rel.reshape(rows, cols)
        # 低频判据（比「二阶差分有界」更本质）：沿行/列的一维自相关**首次过零**处
        # = 最短的半波长（格）。格点尺度碎纹会在 1 格处过零；设计波长 ≥7 格 → ≥3 格。
        def first_zero(a_1d: np.ndarray) -> float:
            a = a_1d - a_1d.mean()
            ac = np.correlate(a, a, mode="full")[a.size - 1:]
            ac = ac / max(float(ac[0]), 1e-12)
            for k in range(1, ac.size):
                if ac[k] <= 0.0:
                    return float(k)
            return float(ac.size)

        zeros = [first_zero(g[r]) for r in range(0, rows, 7)] + \
                [first_zero(g[:, c]) for c in range(0, cols, 11)]
        worst = min(zeros)
        print(f"INFO overlay: 浮雕最短半波长 {worst:.0f} 格（设计 ≥3 格）")
        assert worst >= 2.5, f"浮雕含格点尺度分量（最短半波长 {worst:.1f} 格）"
    finally:
        ov.hide()


def test_relief_weight_scales_uniformly_when_compressed():
    """均匀压缩（整块摆小一档）时，浮雕权重应处处 = 格距比（含 4 个网格角点）。

    首版 `rxc/ryc` 只填内圈、边界恒 1.0 → 角点在压缩态漏过缩放（复核实测 4 个角点
    权重 1.0 vs 内区 ≤0.35）。此测试用「整块缩小 40%」把格距比变成常数，从而只要
    任何一处权重偏离该常数就会红。
    """
    ov, sim = _v6_overlay()
    try:
        ov.show()
        sim.set_placement((400.0, 300.0), 480.0, 360.0, 0.0, zero_velocity=True)
        ov._rebuild_geometry()
        w = np.asarray(ov._w_relief, dtype=np.float64)
        assert w.shape == (sim.cols * sim.rows,)
        ratio = 0.6                      # 480/800 = 0.6（同比例缩小）
        lo, hi = float(w.min()), float(w.max())
        print(f"INFO overlay: 均匀压缩 0.6 下浮雕权重 min={lo:.3f} max={hi:.3f}")
        assert hi - lo < 0.05, f"权重不均匀（角点漏过缩放？）：{lo:.3f}~{hi:.3f}"
        # 只查内圈（边界行/列沿用相邻内圈值，允许首/末格有半个格子的差）
        inner = w.reshape(sim.rows, sim.cols)[1:-1, 1:-1]
        assert abs(float(inner.mean()) - ratio) < 0.05, \
            f"权重未跟随格距比：{inner.mean():.3f}（期望 ≈{ratio}）"
    finally:
        ov.hide()
