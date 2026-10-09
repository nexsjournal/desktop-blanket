"""tests/test_menu.py — 2026-10-09 新增行为回归：毯上右键菜单 + 尺寸预设落盘 + menuForEvent_。

锁定三块行为（被测实现 rug.py / overlay.py 只读，未按实现细节断言）：
  ① `rug.RugApp._make_context_menu(x, y)` 现场构建的 NSMenu 结构：第一项标题随
     `_state` / `_saved_placement`，「调整尺寸」子菜单 = 小/中/大预设 + 「拖拽调整…」
     （未铺上 disabled），设置/退出等项 target 已接线；
  ② `menuSizePreset_(sender)` 读 `representedObject()` → 改写 settings 的
     `rug_w_frac` / `rug_h_frac` 并**落盘**（小 0.34/0.42、中 0.52/0.62、大 0.70/0.80）；
  ③ `RugRootView.menuForEvent_(event)`：毯上（≤ GRAB_RADIUS_PT=40pt）→ provider(x, y) 的菜单；
     毯外 / 无 provider / 异常 / 未显示 / sim 缺失 → None（右键不弹，不卡鼠标）。
     另带 `RugOverlay(menu_provider=...)` 的注入读取面。

安全：**任何写盘都只允许写到 pytest tmp_path**——`app` 夹具把 `settings.path` 指向
tmp_path，`_guard_project_settings` 逐字节比对项目 settings.json（前后必须一致）。
AppKit：菜单对象构造需要 NSApplication 初始化（_appkit 夹具），不建窗口、不跑事件循环。
事件构造：NSView 离屏 + 只有 `locationInWindow()` 的 NSEvent 替身（现有 test_overlay.py
同款做法，实测稳定）；真正无法离屏覆盖的三条见文件末尾说明。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

rug = pytest.importorskip("rug", reason="rug.py 依赖 AppKit/Quartz")
overlay = pytest.importorskip("overlay", reason="overlay.py 依赖 AppKit/SceneKit")
cloth = pytest.importorskip("cloth")
_foundation = pytest.importorskip("Foundation")
NSMakeRect = _foundation.NSMakeRect
NSMakePoint = _foundation.NSMakePoint
NSObject = _foundation.NSObject
from AppKit import (NSApplication, NSControlStateValueOff, NSControlStateValueOn,
                    NSMenu, NSScreen)

PROJECT_SETTINGS = Path(rug.__file__).resolve().parent / "settings.json"
ROOT = Path(__file__).resolve().parent.parent
TEXTURE = str(ROOT / "assets" / "rugs" / "rug-01.png")

# 「调整尺寸」子菜单的三档预设（rug.py 的 _SIZE_PRESETS；标签里也写了百分比）
SIZE_PRESETS = {"small": (0.34, 0.42), "medium": (0.52, 0.62), "large": (0.70, 0.80)}

# 离屏 view 尺寸（与 test_overlay.py 同一屏幕量级）；毯子 800×600，毯外点最近距离 >40pt
SCREEN_W, SCREEN_H = 2560.0, 1440.0
SIM_W, SIM_H = 800.0, 600.0
IN_RUG_WINDOW = (400.0, SCREEN_H - 300.0)    # 窗口坐标 → sim (400, 300)（毯内）
OUT_RUG_WINDOW = (2200.0, SCREEN_H - 1400.0)  # 窗口坐标 → sim (2200, 1400)（毯外远点）


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module", autouse=True)
def _appkit():
    """NSMenu/NSMenuItem 构造前必须初始化 NSApplication（不建窗口、不 run）。"""
    NSApplication.sharedApplication()
    yield


@pytest.fixture(autouse=True)
def _guard_project_settings():
    """兜底：本模块任何用例都不得改动项目 settings.json（逐字节比对）。"""
    before = PROJECT_SETTINGS.read_bytes()
    yield
    after = PROJECT_SETTINGS.read_bytes()
    assert after == before, "测试改动了项目 settings.json（必须只写 tmp_path）"


@pytest.fixture
def app(_appkit, tmp_path):
    """RugApp 最小构造：__new__（alloc/init）+ __init__（PyObjC 范式，rug.py 自带）。

    `settings.path` 一律指向 tmp_path：`menuSizePreset_` 等写盘路径不会碰到项目文件。
    """
    a = rug.RugApp.__new__(rug.RugApp)
    rug.RugApp.__init__(a)
    a.settings.path = str(tmp_path / "settings.json")
    return a


@pytest.fixture
def view(_appkit):
    """离屏 RugRootView（不挂窗口；坐标换算退化为 y 翻转）。"""
    return overlay.RugRootView.alloc().initWithFrame_(
        NSMakeRect(0.0, 0.0, SCREEN_W, SCREEN_H))


class _NSEventStub(NSObject):
    """最小 NSEvent 替身：menuForEvent_ 只读 `locationInWindow()`。"""

    def initWithPoint_(self, p):
        self._p = p
        return self

    def locationInWindow(self):
        return self._p


def _event(point) -> _NSEventStub:
    return _NSEventStub.alloc().initWithPoint_(NSMakePoint(*point))


class _FakeMenuOwner:
    """RugRootView.menuForEvent_ 的最小 owner：`_sim` / `is_shown()` / `menu_provider`。"""

    def __init__(self, sim, shown: bool = True, provider=None):
        self._sim = sim
        self._shown = shown
        self.menu_provider = provider

    def is_shown(self) -> bool:
        return self._shown


def _items(menu) -> list:
    return list(menu.itemArray())


def _titles(menu) -> list[str]:
    return [str(it.title()) for it in _items(menu)]


def _find_by_title(menu, title: str):
    for it in _items(menu):
        if str(it.title()) == title:
            return it
    raise AssertionError(f"菜单里找不到「{title}」：{_titles(menu)}")


def _find_by_repr(menu, key: str):
    for it in _items(menu):
        if str(it.representedObject()) == key:
            return it
    raise AssertionError(
        f"子菜单里找不到 representedObject={key!r} 的项：{_titles(menu)}")


# ---------------------------------------------------------------------------
# A. 菜单结构：第一项标题随状态 / 尺寸子菜单 / 设置与退出项
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "state, saved, expected",
    [
        ("idle", None, "铺上毯子"),
        ("idle", (600.0, 450.0, 800.0, 600.0, 0.0), "摆放回来"),
        ("resting", None, "掀开毯子（收起）"),
        ("throwing", None, "掀开毯子（收起）"),
        ("dragging", None, "掀开毯子（收起）"),
        ("lifting", None, "掀开毯子（收起）"),
    ],
    ids=["idle-no-saved", "idle-with-saved", "resting", "throwing", "dragging", "lifting"],
)
def test_context_menu_first_item_follows_state(app, state, saved, expected):
    """第一项标题 = idle（无记忆）铺上 / idle（有记忆）摆放回来 / 非 idle 掀开收起。"""
    app._state = state
    app._saved_placement = saved
    menu = app._make_context_menu(10.0, 20.0)
    first = _items(menu)[0]
    assert str(first.title()) == expected
    assert str(first.action()) == "menuToggleRug:", "第一项必须接到 menuToggleRug:"
    assert first.target() is not None, "第一项 target 未接线"


def test_size_submenu_presets_and_edit_item(app):
    """「调整尺寸」子菜单：3 档预设（小/中/大，representedObject + 百分比标签）+「拖拽调整…」。"""
    app._state = app.STATE_IDLE
    sub = _find_by_title(app._make_context_menu(0.0, 0.0), "调整尺寸").submenu()
    assert sub is not None, "「调整尺寸」没有子菜单"
    presets = [it for it in _items(sub) if it.representedObject() is not None]
    assert len(presets) == 3, f"预设项应为 3 个，实测 {len(presets)}：{_titles(sub)}"
    for key, (pw, _ph) in SIZE_PRESETS.items():
        it = _find_by_repr(sub, key)
        assert str(it.action()) == "menuSizePreset:", f"{key} 项 action 未接线"
        assert it.target() is not None, f"{key} 项 target 未接线"
        assert f"{pw * 100:.0f}%" in str(it.title()), \
            f"{key} 的标签没写对应屏占比：{str(it.title())!r}"
    edit = _find_by_title(sub, "拖拽调整（四角缩放 / 旋转）…")
    assert str(edit.action()) == "menuEditRug:"
    assert edit.target() is not None
    # 未铺上（idle）时不能拖拽调整；且子菜单必须关掉自动启用，否则 AppKit 会把项重启用
    assert edit.isEnabled() is False, "idle 时「拖拽调整…」应 disabled"
    assert sub.autoenablesItems() is False, "子菜单需关掉 autoenables（否则 disabled 会被覆盖）"

    app._state = app.STATE_RESTING
    sub2 = _find_by_title(app._make_context_menu(0.0, 0.0), "调整尺寸").submenu()
    assert _find_by_title(sub2, "拖拽调整（四角缩放 / 旋转）…").isEnabled() is True, \
        "毯子铺上（resting）时「拖拽调整…」应可用"


def test_context_menu_actions_and_targets_wired(app):
    """抚平折痕 / 重置 / 设置… / 退出桌面毛毯 四项都在，且 target 均已接线。"""
    app._state = app.STATE_RESTING
    menu = app._make_context_menu(0.0, 0.0)
    for title, action in (("抚平折痕（摊平毯子）", "menuFlattenRug:"),
                          ("重置位置/大小/角度", "menuResetRug:"),
                          ("设置…", "menuSettings:"),
                          ("退出桌面毛毯", "menuQuit:")):
        it = _find_by_title(menu, title)
        assert str(it.action()) == action, f"「{title}」action 应为 {action}"
        assert it.target() is not None, f"「{title}」target 未接线"
    assert menu.numberOfItems() >= 6, f"菜单项太少：{_titles(menu)}"


# ---------------------------------------------------------------------------
# B. 尺寸预设：改写 settings + 落盘（只写 tmp_path）+ 勾选随 settings
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", ["small", "medium", "large"])
def test_size_preset_writes_settings_and_file(app, key):
    """menuSizePreset_：内存 settings 与 tmp 里的 JSON 同步为新预设；项目文件逐字节未变。"""
    pw, ph = SIZE_PRESETS[key]
    before = PROJECT_SETTINGS.read_bytes()
    sub = _find_by_title(app._make_context_menu(0.0, 0.0), "调整尺寸").submenu()
    app.menuSizePreset_(_find_by_repr(sub, key))

    assert float(app.settings.get("rug_w_frac")) == pytest.approx(pw, abs=1e-9)
    assert float(app.settings.get("rug_h_frac")) == pytest.approx(ph, abs=1e-9)
    assert Path(app.settings.path).exists(), "预设必须落盘到 settings.path"
    on_disk = json.loads(Path(app.settings.path).read_text(encoding="utf-8"))
    assert float(on_disk["rug_w_frac"]) == pytest.approx(pw, abs=1e-9), \
        f"落盘 JSON 未同步：{on_disk.get('rug_w_frac')}"
    assert float(on_disk["rug_h_frac"]) == pytest.approx(ph, abs=1e-9)
    assert PROJECT_SETTINGS.read_bytes() == before, "预设写到了项目 settings.json！"
    print(f"INFO menu: 预设 {key} → {pw}×{ph}（tmp JSON 已同步）")

    # 勾选跟随 settings：重建菜单后只有当前预设是 ON
    sub2 = _find_by_title(app._make_context_menu(0.0, 0.0), "调整尺寸").submenu()
    for other in SIZE_PRESETS:
        state = _find_by_repr(sub2, other).state()
        want = NSControlStateValueOn if other == key else NSControlStateValueOff
        assert state == want, f"预设 {other} 勾选状态应为 {want}，实测 {state}"


# ---------------------------------------------------------------------------
# C. menuForEvent_：毯上 -> provider 菜单；毯外/异常 -> None
# ---------------------------------------------------------------------------
def test_menu_for_event_on_rug_returns_provider_menu(view):
    """毯上（≤40pt）→ 返回 provider 给的菜单，且传给 provider 的是 sim 本地坐标（已翻 y）。"""
    sim = cloth.ClothSim(SIM_W, SIM_H)
    menu = NSMenu.alloc().init()
    calls = []

    def provider(x, y):
        calls.append((x, y))
        return menu

    owner = _FakeMenuOwner(sim, provider=provider)
    view._owner = owner
    px, py = IN_RUG_WINDOW
    assert sim.nearest_distance(px, SCREEN_H - py) <= 40.0  # 前提：该点确在毯上
    got = view.menuForEvent_(_event((px, py)))
    assert got is menu, "毯上右键应返回 provider 给的菜单"
    assert len(calls) == 1, f"provider 应被调用一次，实测 {len(calls)} 次"
    assert calls[0] == pytest.approx((px, SCREEN_H - py)), \
        f"provider 收到的应是 sim 本地坐标（y 已翻转），实测 {calls[0]}"


def test_menu_for_event_off_rug_returns_none_without_calling_provider(view):
    """毯外（>40pt）→ None，且 provider 根本不被调用（判定在调用之前）。"""
    sim = cloth.ClothSim(SIM_W, SIM_H)
    calls = []

    def provider(x, y):
        calls.append((x, y))
        return NSMenu.alloc().init()

    view._owner = _FakeMenuOwner(sim, provider=provider)
    px, py = OUT_RUG_WINDOW
    assert sim.nearest_distance(px, SCREEN_H - py) > 40.0  # 前提：该点确在毯外
    assert view.menuForEvent_(_event((px, py))) is None
    assert calls == [], "毯外不该调用 provider"


def test_menu_for_event_y_flip_guard(view):
    """y 翻转方向守卫：把毯内 sim 坐标原样当窗口坐标喂进去必然落空（→ None）。

    毯内基准点 sim (400, 300)；窗口点 (400, 300) 换算后 = sim (400, 1140)，
    落在毯子（y ≤ 600）之外 → None。若实现漏翻/翻错 y，本用例与
    test_menu_for_event_on_rug_returns_provider_menu 必有一条失败。
    """
    sim = cloth.ClothSim(SIM_W, SIM_H)
    calls = []
    view._owner = _FakeMenuOwner(
        sim, provider=lambda x, y: (calls.append((x, y)), NSMenu.alloc().init())[1])
    rug_sim = (IN_RUG_WINDOW[0], SCREEN_H - IN_RUG_WINDOW[1])   # 毯内点 sim 坐标 (400, 300)
    assert sim.nearest_distance(*rug_sim) <= 40.0               # 前提：该 sim 点确在毯上
    # 把 sim 坐标原样当窗口坐标喂进去 → 换算得 sim y = 1440-300 = 1140（毯外）→ None
    assert sim.nearest_distance(rug_sim[0], SCREEN_H - rug_sim[1]) > 40.0
    assert view.menuForEvent_(_event(rug_sim)) is None
    assert calls == []


def test_menu_for_event_none_without_provider_or_owner(view):
    """无 provider（默认）→ None；连 _owner 都没有 → None（异常路径不抛、不卡鼠标）。"""
    sim = cloth.ClothSim(SIM_W, SIM_H)
    view._owner = _FakeMenuOwner(sim, provider=None)
    assert view.menuForEvent_(_event(IN_RUG_WINDOW)) is None
    view._owner = None
    assert view.menuForEvent_(_event(IN_RUG_WINDOW)) is None


def test_menu_for_event_none_when_hidden_or_sim_missing(view):
    """owner 未显示 / sim 缺失 → None（毯子不在屏幕上时右键不该弹）。"""
    sim = cloth.ClothSim(SIM_W, SIM_H)
    view._owner = _FakeMenuOwner(sim, shown=False,
                                 provider=lambda x, y: NSMenu.alloc().init())
    assert view.menuForEvent_(_event(IN_RUG_WINDOW)) is None
    view._owner = _FakeMenuOwner(None, provider=lambda x, y: NSMenu.alloc().init())
    assert view.menuForEvent_(_event(IN_RUG_WINDOW)) is None


def test_menu_for_event_none_when_provider_raises(view):
    """provider 抛异常 → None（构建失败只告警，不能让右键把应用拖崩）。"""
    sim = cloth.ClothSim(SIM_W, SIM_H)

    def boom(x, y):
        raise RuntimeError("boom")

    view._owner = _FakeMenuOwner(sim, provider=boom)
    assert view.menuForEvent_(_event(IN_RUG_WINDOW)) is None


def test_menu_for_event_with_real_app_provider(view, app):
    """端到端接线：provider = RugApp._make_context_menu 时，毯上右键真的拿到已构建的菜单。

    （rug.py 的 _ensure_overlay 就是把这个绑定方法注入 RugOverlay.menu_provider；
    这里用同一对象验证 view → provider → NSMenu 的整条链路。）
    """
    sim = cloth.ClothSim(SIM_W, SIM_H)
    app._state = app.STATE_IDLE
    app._saved_placement = None
    view._owner = _FakeMenuOwner(sim, provider=app._make_context_menu)
    got = view.menuForEvent_(_event(IN_RUG_WINDOW))
    assert isinstance(got, NSMenu), f"毯上右键应返回 NSMenu，实测 {type(got)}"
    assert str(_items(got)[0].title()) == "铺上毯子"
    assert view.menuForEvent_(_event(OUT_RUG_WINDOW)) is None


# ---------------------------------------------------------------------------
# D. RugOverlay.menu_provider 注入读取面
# ---------------------------------------------------------------------------
def test_overlay_menu_provider_default_none_and_injected(_appkit):
    """RugOverlay(menu_provider=cb).menu_provider is cb；不传时为 None。"""
    screen = NSScreen.mainScreen()
    if screen is None:
        pytest.skip("无可用屏幕")
    cb = lambda x, y: None  # noqa: E731（只验证注入/读取面）
    ov = overlay.RugOverlay(screen, TEXTURE, lambda w, h: None, menu_provider=cb)
    assert ov.menu_provider is cb, "注入的 menu_provider 没被存下来/没暴露"
    ov2 = overlay.RugOverlay(screen, TEXTURE, lambda w, h: None)
    assert ov2.menu_provider is None, "未注入时 menu_provider 应为 None（右键不弹）"


# ---------------------------------------------------------------------------
# 未覆盖（无窗口 / 无事件循环下无法稳定断言，留给真机脚本与人工验收）：
#   a) AppKit 真实派发的右键 NSEvent（NSWindow → view）这条路由；
#   b) RugApp._ensure_overlay 里 `menu_provider=self._make_context_menu` 的那一行注入
#      （需要真建 NSPanel；本文件用 D 段验证读取面、用 test_menu_for_event_with_real_app_provider
#      验证同一绑定方法能被 view 正确调用）；
#   c) 菜单项被真实点击（NSMenu 跟踪循环）后动作生效——本文件改为直接调
#      menuSizePreset_（＝动作实现），点击→动作的派发由 AppKit 负责。
# 本文件锁的是「菜单内容 + 判定逻辑 + 落盘」；a/b/c 由真机验收覆盖。
# ---------------------------------------------------------------------------
