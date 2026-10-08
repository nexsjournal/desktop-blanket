"""settings_ui.py — 「设置」窗口（菜单栏 → 设置…）+ 持久化（项目内 settings.json）。

窗口内容（固定版式，小面板）：
  - 地毯款式：下拉（assets/rugs/rug-*.png）+「打开素材文件夹」+「重新扫描」
  - 推荐规格说明（给用户看的贴图规范）
  - 默认大小（宽/高 占屏比滑块）、默认角度（-45°~45°）
  - 图标隆起（开关 + 高度倍率）、伪阴影（开关）
  - 按钮：重置摆放 / 关闭

持久化：<项目根>/settings.json（**项目内文件**，不进系统状态；删掉即回默认值）。
所有控件改动即时生效并落盘（由 rug.py 注入的回调应用）。
"""
from __future__ import annotations

import json
import os
import subprocess

import objc

from AppKit import (NSButton, NSButtonTypeSwitch, NSColor, NSFont, NSPanel, NSPopUpButton,
                    NSSlider, NSTextField, NSView,
                    NSWindowStyleMaskTitled, NSWindowStyleMaskClosable,
                    NSWindowStyleMaskUtilityWindow)
from Foundation import NSMakeRect, NSObject

__all__ = ["Settings", "SettingsController", "load_settings", "save_settings", "SPEC_TEXT"]

SPEC_TEXT = (
    "推荐贴图规格：正俯视（从上往下拍/扫描）、光照均匀无强阴影与反光；"
    "横向长方形，长宽比约 1:0.6~0.7；宽度 ≥2048px；边缘最好带透明通道（PNG）。"
    "带背景的照片也行——放进文件夹后我来抠图裁剪。命名 rug-01.png、rug-02.png…"
)

DEFAULTS = {
    "rug_w_frac": 0.52,     # 默认宽度占屏比
    "rug_h_frac": 0.62,     # 默认高度占屏比
    "angle_deg": 0.0,       # 默认摆放角度
    "texture": "",          # 选中的贴图文件名（""=自动取第一个 rug-*.png）
    "bump_enabled": True,   # 图标隆起
    "bump_scale": 1.0,      # 隆起高度倍率
    "shadow_enabled": True, # 伪阴影
}


class Settings:
    """设置的读写（项目内 settings.json；文件缺失/损坏 → 默认值）。"""

    def __init__(self, path: str) -> None:
        self.path = path
        self.data = dict(DEFAULTS)
        try:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                for k, v in DEFAULTS.items():
                    if k not in raw:
                        continue
                    val = raw[k]
                    if isinstance(v, bool):
                        if isinstance(val, bool):
                            self.data[k] = val
                    elif isinstance(v, float):
                        if isinstance(val, (int, float)) and not isinstance(val, bool):
                            self.data[k] = float(val)
                    elif isinstance(v, str):
                        if isinstance(val, str):
                            self.data[k] = val
        except (OSError, json.JSONDecodeError):
            pass

    def get(self, key: str):
        return self.data.get(key, DEFAULTS.get(key))

    def set(self, key: str, value) -> None:
        self.data[key] = value
        save_settings(self)


def load_settings(project_dir: str) -> Settings:
    return Settings(os.path.join(project_dir, "settings.json"))


def save_settings(settings: Settings) -> None:
    """原子写（临时文件 + rename）；失败只告警不抛（设置窗不能把应用搞崩）。"""
    tmp = settings.path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(settings.data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, settings.path)
    except OSError as exc:
        print(f"RUG-WARN 设置保存失败：{exc}", flush=True)


def _label(text: str, x: float, y: float, w: float, h: float = 20.0,
           size: float = 12.0, color=None, bold: bool = False) -> NSTextField:
    t = NSTextField.alloc().initWithFrame_(NSMakeRect(x, y, w, h))
    t.setStringValue_(text)
    t.setBezeled_(False)
    t.setDrawsBackground_(False)
    t.setEditable_(False)
    t.setSelectable_(False)
    if bold:
        t.setFont_(NSFont.boldSystemFontOfSize_(size))
    else:
        t.setFont_(NSFont.systemFontOfSize_(size))
    if color is not None:
        t.setTextColor_(color)
    return t


class SettingsController(NSObject):
    """设置窗控制器：建窗 + 控件回调（回调由 RugApp 注入）。"""

    def __new__(cls):
        # PyObjC 范式（同 rug.py RugApp）：alloc().init() 由 __new__ 内部完成，
        # 外部用普通 Python 构造 SettingsController() 才会走到 __init__。
        return cls.alloc().init()

    def __init__(self) -> None:
        self.panel = None
        self.app = None            # RugApp（鸭子类型）
        self.texture_paths: list[str] = []
        self._control_refs = {}

    # ---- 由 RugApp 调用 ----
    @objc.python_method
    def _configure(self, app, texture_paths: list[str]) -> None:
        self.app = app
        self.texture_paths = list(texture_paths)

    @objc.python_method
    def _show(self) -> None:
        if self.panel is None:
            self._build()
        self._sync_controls()
        # accessory 应用平时不在前台：先把应用拉到前台，再显示（否则窗口可能拿不到 key）
        try:
            from AppKit import NSApplication
            NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
        except Exception:
            pass
        self.panel.orderFrontRegardless()
        self.panel.makeKeyAndOrderFront_(None)
        print(f"RUG-INFO 设置窗已显示：visible={bool(self.panel.isVisible())} "
              f"frame={self.panel.frame()}", flush=True)

    # ---- 建窗 ----
    @objc.python_method
    def _build(self) -> None:
        W, H = 400.0, 472.0
        # 注意（真机踩坑 2026-10-08）：NSPanel + UtilityWindow 默认 hidesOnDeactivate=YES，
        # accessory 应用（不常激活）下窗口一显示就被藏起来 = 用户「点了设置没反应」。
        # 这里不用 UtilityWindow 风格，并显式关掉 hidesOnDeactivate / 打开浮动面板语义。
        p = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(200.0, 160.0, W, H),
            NSWindowStyleMaskTitled | NSWindowStyleMaskClosable,
            2, False)  # NSBackingStoreBuffered
        p.setTitle_("桌面毛毯 · 设置")
        p.setLevel_(0)                 # 普通窗口层级（在毯子之上）
        p.setHidesOnDeactivate_(False)
        try:
            p.setFloatingPanel_(False)
            p.setBecomesKeyOnlyIfNeeded_(False)
        except Exception:
            pass
        p.setReleasedWhenClosed_(False)
        p.setCollectionBehavior_(1 << 0)   # NSWindowCollectionBehaviorDefault（跟随当前 Space）
        v = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, W, H))

        y = H - 44.0
        v.addSubview_(_label("地毯款式", 20, y, 100, bold=True))
        pop = NSPopUpButton.alloc().initWithFrame_pullsDown_(NSMakeRect(20, y - 30, 220, 26), False)
        pop.setTarget_(self)
        pop.setAction_("onPickTexture:")
        v.addSubview_(pop)
        self._control_refs["texture_popup"] = pop
        btn_open = NSButton.alloc().initWithFrame_(NSMakeRect(250, y - 30, 130, 26))
        btn_open.setTitle_("打开素材文件夹")
        btn_open.setBezelStyle_(1)     # rounded
        btn_open.setTarget_(self)
        btn_open.setAction_("onOpenFolder:")
        v.addSubview_(btn_open)
        btn_rescan = NSButton.alloc().initWithFrame_(NSMakeRect(250, y - 62, 130, 26))
        btn_rescan.setTitle_("重新扫描")
        btn_rescan.setBezelStyle_(1)
        btn_rescan.setTarget_(self)
        btn_rescan.setAction_("onRescan:")
        v.addSubview_(btn_rescan)
        y -= 74.0

        spec = _label(SPEC_TEXT, 20, y - 62, 360, 60, size=10.5,
                      color=NSColor.secondaryLabelColor())
        spec.setLineBreakMode_(0)      # NSLineBreakByWordWrapping
        try:
            spec.setUsesSingleLineMode_(False)
        except Exception:
            pass
        v.addSubview_(spec)
        y -= 84.0

        v.addSubview_(_label("默认大小（占屏幕宽/高）", 20, y, 240, bold=True))
        y -= 26.0
        v.addSubview_(_label("宽", 20, y + 4, 24))
        self._control_refs["w_slider"] = self._slider(v, 48, y, 250, 0.30, 0.95, "onSize:")
        self._control_refs["w_value"] = _label("—", 306, y + 4, 70)
        v.addSubview_(self._control_refs["w_value"])
        y -= 30.0
        v.addSubview_(_label("高", 20, y + 4, 24))
        self._control_refs["h_slider"] = self._slider(v, 48, y, 250, 0.30, 0.95, "onSize:")
        self._control_refs["h_value"] = _label("—", 306, y + 4, 70)
        v.addSubview_(self._control_refs["h_value"])
        y -= 36.0

        v.addSubview_(_label("默认摆放角度", 20, y, 240, bold=True))
        y -= 26.0
        self._control_refs["ang_slider"] = self._slider(v, 48, y, 250, -45.0, 45.0, "onAngle:")
        self._control_refs["ang_value"] = _label("—", 306, y + 4, 70)
        v.addSubview_(self._control_refs["ang_value"])
        y -= 40.0

        chk = NSButton.alloc().initWithFrame_(NSMakeRect(20, y, 220, 22))
        chk.setButtonType_(NSButtonTypeSwitch)
        chk.setTitle_("图标在毯下顶起隆包")
        chk.setTarget_(self)
        chk.setAction_("onBumpToggle:")
        v.addSubview_(chk)
        self._control_refs["bump_check"] = chk
        y -= 30.0
        v.addSubview_(_label("隆起高度", 20, y + 4, 60))
        self._control_refs["bump_slider"] = self._slider(v, 88, y, 210, 0.4, 2.0, "onBumpScale:")
        self._control_refs["bump_value"] = _label("—", 306, y + 4, 70)
        v.addSubview_(self._control_refs["bump_value"])
        y -= 34.0

        chk2 = NSButton.alloc().initWithFrame_(NSMakeRect(20, y, 240, 22))
        chk2.setButtonType_(NSButtonTypeSwitch)
        chk2.setTitle_("伪阴影（毯子边缘投影）")
        chk2.setTarget_(self)
        chk2.setAction_("onShadowToggle:")
        v.addSubview_(chk2)
        self._control_refs["shadow_check"] = chk2
        y -= 40.0

        b_reset = NSButton.alloc().initWithFrame_(NSMakeRect(20, y, 150, 28))
        b_reset.setTitle_("重置位置/大小/角度")
        b_reset.setBezelStyle_(1)
        b_reset.setTarget_(self)
        b_reset.setAction_("onResetPlacement:")
        v.addSubview_(b_reset)
        b_close = NSButton.alloc().initWithFrame_(NSMakeRect(W - 130.0, y, 110, 28))
        b_close.setTitle_("关闭")
        b_close.setBezelStyle_(1)
        b_close.setKeyEquivalent_("\r")
        b_close.setTarget_(self)
        b_close.setAction_("onClose:")
        v.addSubview_(b_close)

        p.setContentView_(v)
        self.panel = p

    @objc.python_method
    def _slider(self, parent, x, y, w, lo, hi, action):
        s = NSSlider.alloc().initWithFrame_(NSMakeRect(x, y, w, 24))
        s.setMinValue_(lo)
        s.setMaxValue_(hi)
        s.setTarget_(self)
        s.setAction_(action)
        s.setContinuous_(True)
        parent.addSubview_(s)
        return s

    # ---- 控件状态同步 ----
    @objc.python_method
    def _sync_controls(self) -> None:
        if self.app is None or self.panel is None:
            return
        st = self.app.settings
        refs = self._control_refs
        pop = refs.get("texture_popup")
        if pop is not None:
            pop.removeAllItems()
            for p in self.texture_paths:
                pop.addItemWithTitle_(os.path.basename(p))
            cur = os.path.basename(str(st.get("texture") or ""))
            names = [os.path.basename(p) for p in self.texture_paths]
            if cur in names:
                pop.selectItemWithTitle_(cur)
        ws = refs.get("w_slider")
        if ws is not None:
            ws.setDoubleValue_(float(st.get("rug_w_frac")))
        hs = refs.get("h_slider")
        if hs is not None:
            hs.setDoubleValue_(float(st.get("rug_h_frac")))
        an = refs.get("ang_slider")
        if an is not None:
            an.setDoubleValue_(float(st.get("angle_deg")))
        bc = refs.get("bump_check")
        if bc is not None:
            bc.setState_(1 if st.get("bump_enabled") else 0)
        bs = refs.get("bump_slider")
        if bs is not None:
            bs.setDoubleValue_(float(st.get("bump_scale")))
        sc = refs.get("shadow_check")
        if sc is not None:
            sc.setState_(1 if st.get("shadow_enabled") else 0)
        self._update_value_labels()

    @objc.python_method
    def _update_value_labels(self) -> None:
        st = self.app.settings
        refs = self._control_refs
        if "w_value" in refs:
            refs["w_value"].setStringValue_(f"{float(st.get('rug_w_frac')) * 100:.0f}%")
        if "h_value" in refs:
            refs["h_value"].setStringValue_(f"{float(st.get('rug_h_frac')) * 100:.0f}%")
        if "ang_value" in refs:
            refs["ang_value"].setStringValue_(f"{float(st.get('angle_deg')):.0f}°")
        if "bump_value" in refs:
            refs["bump_value"].setStringValue_(f"×{float(st.get('bump_scale')):.2f}")

    # ---- 控件回调 ----
    def onPickTexture_(self, sender) -> None:
        pop = self._control_refs.get("texture_popup") or sender
        idx = int(pop.indexOfSelectedItem())
        if 0 <= idx < len(self.texture_paths):
            self.app.apply_settings_texture(self.texture_paths[idx])

    def onOpenFolder_(self, sender) -> None:
        try:
            subprocess.Popen(["open", self.app.assets_dir()])
        except Exception as exc:
            print(f"RUG-WARN 打开素材文件夹失败：{exc}", flush=True)

    def onRescan_(self, sender) -> None:
        self.texture_paths = self.app.rescan_textures()
        self._sync_controls()

    def onSize_(self, sender) -> None:
        refs = self._control_refs
        self.app.settings.set("rug_w_frac", round(float(refs["w_slider"].doubleValue()), 3))
        self.app.settings.set("rug_h_frac", round(float(refs["h_slider"].doubleValue()), 3))
        self._update_value_labels()
        self.app.apply_settings_default_size()

    def onAngle_(self, sender) -> None:
        slider = self._control_refs.get("ang_slider") or sender
        self.app.settings.set("angle_deg", round(float(slider.doubleValue()), 1))
        self._update_value_labels()
        self.app.apply_settings_angle()

    def onBumpToggle_(self, sender) -> None:
        chk = self._control_refs.get("bump_check") or sender
        self.app.settings.set("bump_enabled", bool(chk.state()))
        self.app.apply_settings_bumps()

    def onBumpScale_(self, sender) -> None:
        slider = self._control_refs.get("bump_slider") or sender
        self.app.settings.set("bump_scale", round(float(slider.doubleValue()), 2))
        self._update_value_labels()
        self.app.apply_settings_bumps()

    def onShadowToggle_(self, sender) -> None:
        chk = self._control_refs.get("shadow_check") or sender
        self.app.settings.set("shadow_enabled", bool(chk.state()))
        self.app.apply_settings_shadow()

    def onResetPlacement_(self, sender) -> None:
        self.app.menuResetRug_(None)

    def onClose_(self, sender) -> None:
        if self.panel is not None:
            self.panel.orderOut_(None)
