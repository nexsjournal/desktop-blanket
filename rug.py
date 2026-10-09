"""rug.py — 菜单栏入口 + 总控状态机（contracts/interfaces.py 的 RugApp/parse_args/main）。

职责（contract.md §3D / 开发文档 §7.5）：
  - accessory NSApplication + NSStatusItem 菜单：铺上/掀开 ⌘\、地毯款式子菜单
    （glob assets/rugs/rug-*.png，切换 = hide 旧 RugOverlay、以新贴图重建）、退出。
  - 状态机：idle --lay_down--> throwing --落定(is_asleep 或超时≈4s)--> resting
    --on_grab--> dragging --on_release--> resting --lift--> lifting --完成/超时--> idle；
    throwing/dragging 中允许直接 lift。
  - IconSensor 生命周期（主线程）：query() → calibration() → BumpField →
    overlay.sim.set_bumps + wake()；on_change 只置脏标志，0.5s 防抖 timer 重建链路。
  - 权限降级（S5）：query() 抛 IconSensorError → degraded=True（无隆起、物理照常、
    菜单标注），不 start sensor、不二次弹权限；calibration() 失败只回退
    (0.0, 24.0) + RUG-WARN，隆起继续可用（契约 interfaces.py:295）。
  - 退出（S6）：stop sensor → hide overlay → terminate → os._exit(0) 兜底。
  - --selftest：跳过 IconSensor（不碰 Finder、无人值守不弹权限），lay_down 后
    t≈2s 用 CGWindowListCopyWindowInfo 探针找本进程面板打 SELFTEST-LEVEL，
    t≈5s 打 SELFTEST-OK / SELFTEST-FAIL 后 os._exit(0/1)。

启动行为（2026-10-09 打包为 .app 后调整）：
  - 默认「启动即铺上」（即旧 --demo）：双击 app 屏幕上立刻出现毯子，不必先去
    菜单栏点「铺上」。旧默认（只驻留菜单栏、屏幕零反馈）被用户反馈为「打开应用
    没反应」；--start-hidden 可恢复旧行为。
  - 双击启动没有终端可看：非 TTY 启动时 stdout/stderr 追加到
    ~/Library/Logs/DesktopRug.log（终端里跑不受影响）。
  - 单实例锁：~/Library/Application Support/DesktopRug/instance.lock（flock），
    重复启动只留一条日志后退出，避免出现两个菜单栏项/两张毯子。
"""
from __future__ import annotations

import argparse
import fcntl
import glob
import math
import os
import signal
import sys
import time

import objc
import numpy as np
from Foundation import NSObject, NSTimer
from AppKit import (NSApplication, NSApplicationActivationPolicyAccessory,
                    NSApplicationActivationPolicyRegular, NSCommandKeyMask,
                    NSControlStateValueOff, NSControlStateValueOn, NSImage, NSMenu,
                    NSMenuItem, NSScreen, NSShiftKeyMask, NSStatusBar,
                    NSVariableStatusItemLength)
from Quartz import (CGWindowListCopyWindowInfo, kCGNullWindowID,
                    kCGWindowListOptionAll)

from bumps import BumpField
from settings_ui import SettingsController, load_settings
from cloth import ClothSim
from icons import IconSensor, IconSensorError
import menubar
from overlay import RugOverlay, icon_level_base

PANEL_LEVEL_DELTA = 8          # kCGDesktopIconWindowLevel + 8（contract.md §4）
THROW_TIMEOUT_S = 4.0          # throwing 落定兜底（契约 ≈4s）
LIFT_TIMEOUT_S = 3.0           # lifting 完成兜底
LIFT_DRAG_S = 0.6              # 掀开的拖拽动画时长
LIFT_TARGET_FRAC = -0.3        # 掀开拖到屏外（-0.3w, -0.3h）
DIRTY_DEBOUNCE_S = 0.5         # on_change 脏标志轮询（G4：总延迟 < 2s）
# 毯子默认大小（屏宽/高占比）与默认摆放角度。
# 0.52x0.62：用户反馈 0.72x0.80 偏大 → 调小；角度 0 = 正放（原落地带斜角被反馈不好看）。
RUG_W_FRAC = 0.52
RUG_H_FRAC = 0.62
RUG_DEFAULT_ANGLE_DEG = 0.0
# 右键菜单「调整尺寸」预设：(宽占屏比, 高占屏比)。中 = DEFAULTS 的出厂值。
_SIZE_PRESETS = {
    "small": (0.34, 0.42),
    "medium": (0.52, 0.62),
    "large": (0.70, 0.80),
}


def _log(level: str, msg: str) -> None:
    print(f"{level} {msg}", flush=True)


APP_SUPPORT_DIR = os.path.expanduser("~/Library/Application Support/DesktopRug")
LOG_PATH = os.path.expanduser("~/Library/Logs/DesktopRug.log")
_LOG_ROTATE_BYTES = 1 << 20   # 超过 1MB 轮转一次（只留 .1 一份）


def _setup_file_logging() -> None:
    """Finder 双击启动（stdin/stdout 都不是终端）时把 stdout/stderr 追加到日志文件。

    双击启动的进程没有可看的 stdout——出问题时用户只能看到「点了没反应」，
    日志是唯一线索。终端里跑、以及被脚本用管道读 stdout（scripts/verify_routing.py
    等）都不受影响：只要 stdin 或 stdout 有一个是终端就原样输出。
    RUG_NO_FILE_LOG=1 可强制关闭落盘。
    """
    if os.environ.get("RUG_NO_FILE_LOG") == "1":
        return
    try:
        if sys.stdout is not None and sys.stdout.isatty():
            return
        if sys.stdin is not None and sys.stdin.isatty():
            return
    except Exception:
        pass
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        try:
            if os.path.getsize(LOG_PATH) > _LOG_ROTATE_BYTES:
                os.replace(LOG_PATH, LOG_PATH + ".1")
        except OSError:
            pass
        fd = os.open(LOG_PATH, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        if fd > 2:
            os.close(fd)
    except Exception:
        return
    _log("RUG-INFO", f"--- 启动 pid={os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')} "
                     f"日志={LOG_PATH} ---")


def _acquire_instance_lock():
    """单实例锁（flock，非阻塞）：已有实例在跑 → 返回 None。

    调用方拿到锁后须保持引用（文件对象被回收则锁释放）。锁文件只写 pid，
    不做进程存活判断——内核在进程退出时自动释放 flock。
    """
    try:
        os.makedirs(APP_SUPPORT_DIR, exist_ok=True)
        f = open(os.path.join(APP_SUPPORT_DIR, "instance.lock"), "a+")
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return None
    try:
        f.seek(0)
        f.truncate()
        f.write(str(os.getpid()))
        f.flush()
    except OSError:
        pass
    return f


class RugApp(NSObject):
    """accessory 菜单栏应用 + 总控状态机（契约公开面见 contracts/interfaces.py）。"""

    STATE_IDLE = "idle"
    STATE_THROWING = "throwing"
    STATE_RESTING = "resting"
    STATE_DRAGGING = "dragging"
    STATE_LIFTING = "lifting"

    # ---- 构造（PyObjC：__new__ 走 alloc().init()，随后 Python __init__ 接管） ----
    def __new__(cls, selftest: bool = False, assets_dir: str | None = None,
                demo: bool = False, menubar_handoff: str | None = None):
        return cls.alloc().init()

    def __init__(self, selftest: bool = False, assets_dir: str | None = None,
                 demo: bool = False, menubar_handoff: str | None = None) -> None:
        self._selftest = bool(selftest)
        self._demo = bool(demo)
        self._assets_dir = assets_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "assets", "rugs")
        self._state = self.STATE_IDLE
        self._degraded = False
        self._saved_placement = None      # 掀开前记住的摆放（摆放回来用）
        self.settings = load_settings(os.path.dirname(os.path.abspath(__file__)))
        self._settings_ctrl = None        # 设置窗控制器（懒建）
        self._sensor = None
        self._overlay = None
        self._bumps = None
        self._icons_dirty = False
        self._texture_paths = sorted(
            _p for _p in glob.glob(os.path.join(self._assets_dir, "rug-*.png"))
            if not _p.endswith("-back.png"))   # -back.png 是材质背面（v4），不是款式
        self._texture_idx = 0
        _want = os.path.basename(str(self.settings.get("texture") or ""))
        for _i, _p in enumerate(self._texture_paths):
            if os.path.basename(_p) == _want:
                self._texture_idx = _i
        self._status_item = None
        self._toggle_item = None
        self._style_submenu = None
        self._edit_item = None            # 「调整大小/角度」勾选项
        self._reset_item = None           # 「重置位置/大小/角度」
        self._degraded_item = None
        self._handoff_token = menubar_handoff      # 本进程是「菜单栏交接子进程」时的 token
        self._handoff_report = menubar.report_path(self.settings.path)
        self._instance_lock = None                 # main() 里的单实例锁（交接前要放开）
        self._menubar_ok = None                    # True/False：菜单栏项落位判定结果
        self._poll_timer = None
        self._poll_deadline = 0.0
        self._lift_timer = None
        self._lift_t0 = 0.0
        self._lift_from = (0.0, 0.0)
        self._dirty_timer = None
        self._screen = None
        self._w = 0.0
        self._h = 0.0
        self._selftest_ok = False

    # ---- 契约公开面 ----
    @property
    def state(self) -> str:
        return self._state

    @property
    def degraded(self) -> bool:
        return self._degraded

    def run(self) -> int:
        """进入主循环（正常路径经 os._exit 兜底，不返回）。"""
        app = NSApplication.sharedApplication()
        # 常驻状态栏应用（Clash 那种）：平时 Accessory（**不占 Dock**，只有菜单栏图标）；
        # 打开设置窗时才临时切到 Regular（Dock 图标出现、主菜单可用），关窗后切回。
        # 2026-10-09 真机教训：纯菜单栏项在多屏/菜单栏拥挤时可能落在另一块屏上，用户
        # 「看不到图标、也关不掉」——所以设置窗期间给出 Dock 图标 + ⌘Q 这条确定的路。
        app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
        app.setDelegate_(self)
        screen = NSScreen.mainScreen()
        if screen is None:
            _log("RUG-ERROR", "没有可用屏幕，退出")
            os._exit(1)
        self._screen = screen
        f = screen.frame()
        self._w, self._h = float(f.size.width), float(f.size.height)
        self._build_main_menu()
        self._build_menu()
        self._set_dock_icon(app)
        _log("RUG-INFO", f"激活策略=Accessory（常驻菜单栏、不占 Dock），主屏 {self._w:.0f}x{self._h:.0f}")
        if self._selftest:
            # 契约：selftest 跳过 IconSensor，不碰 Finder（无人值守不弹权限窗）
            _log("RUG-INFO", "selftest 模式：跳过 IconSensor，直接 lay_down")
            self._arm_selftest()
        else:
            self._init_sensor()
            self._start_dirty_timer()
            # 菜单栏项落位判定（≤1.2s）：正常落位就继续；被系统隐藏则尝试交接给子进程
            #（macOS 26 + LaunchServices 启动路径踩坑，见 menubar.py 顶部）。放在铺毯之前，
            # 交接成功时本进程直接退出、不留多余窗口。
            self._menubar_settle()
            if self._demo:
                # --demo：启动即铺上，便于「直接看效果」；正常驻留不退出
                _log("RUG-INFO", "demo 模式：启动即自动铺上（菜单里仍可掀开/退出）")
                self.lay_down()
        self._install_sigint()  # Ctrl-C → quit()（app.run() 外层的 except 收不到 KI，见 §12.12）
        try:
            app.run()
        except KeyboardInterrupt:
            _log("RUG-INFO", "Ctrl-C，退出")
        self._cleanup()
        os._exit(0)  # 兜底：app.run() 不可靠返回（contract.md §4）
        return 0     # pragma: no cover

    def lay_down(self) -> None:
        """「铺上」：仅 idle 态有效（其他态幂等忽略）→ throwing（从角落飞入并正放）。"""
        if self._state != self.STATE_IDLE:
            return
        self._ensure_overlay()
        self._overlay.show()
        sim = self._overlay.sim
        if sim is None:
            _log("RUG-ERROR", "overlay.show 后 sim 缺失")
            return
        # 摆放状态：默认大小 + 屏中心 + 默认角度（只记状态，不动 sim，让飞入动画照常）
        self._overlay.apply_placement(self._default_placement(), to_sim=False)
        sim.throw_in((self._w / 2.0, self._h / 2.0), "top_left")
        self._set_state(self.STATE_THROWING)
        self._arm_poll(THROW_TIMEOUT_S)

    def _default_placement(self):
        """默认摆放 (cx, cy, w, h, angle)：屏中心 + 默认尺寸 + 默认角度。"""
        return (self._w / 2.0, self._h / 2.0,
                self._w * float(self.settings.get("rug_w_frac")),
                self._h * float(self.settings.get("rug_h_frac")),
                math.radians(float(self.settings.get("angle_deg"))))

    def place_back(self) -> None:
        """「摆放回来」：把上次掀开前记住的位置/大小/角度原样摆回来（不走飞入动画）。"""
        if self._state != self.STATE_IDLE:
            return
        self._ensure_overlay()
        self._overlay.show()
        sim = self._overlay.sim
        if sim is None:
            _log("RUG-ERROR", "overlay.show 后 sim 缺失")
            return
        p = self._saved_placement or self._default_placement()
        try:
            self._overlay.apply_placement(p)     # 同时写 sim（zero_velocity → 直接落定）
            sim.wake()
        except Exception as exc:
            _log("RUG-WARN", f"摆放回来失败（退回默认）: {exc}")
            self._overlay.apply_placement(self._default_placement())
        self._set_state(self.STATE_RESTING)
        _log("RUG-INFO", f"摆放回来：中心 ({p[0]:.0f},{p[1]:.0f}) 尺寸 {p[2]:.0f}x{p[3]:.0f} "
                         f"{math.degrees(p[4]):.1f}°")

    def lift(self) -> None:
        """「掀开」：非 idle 态 → lifting：抓毯角 drag 至屏外后 release；落定/超时 → idle。

        掀开前记住当前摆放（位置/大小/角度），菜单随后提供「摆放回来」原样恢复。
        """
        if self._state == self.STATE_IDLE or self._overlay is None:
            return
        # 记住摆放 + 退出编辑模式
        try:
            if self._overlay.placement is not None:
                self._saved_placement = tuple(self._overlay.placement)
        except Exception:
            pass
        if self._overlay.edit_mode:
            self._overlay.set_edit_mode(False)
            if self._edit_item is not None:
                self._edit_item.setState_(NSControlStateValueOff)
        self._cancel_poll()
        # 先取消可能还在跑的 lift timer：LIFTING 中重复触发若不取消会泄漏一个
        # 每帧 drag_to(-0.3w,-0.3h)+release(fling=True) 的 timer（毁掉后续用户抓拖）
        self._cancel_lift_timer()
        sim = self._overlay.sim
        if sim is None:
            self._finish_lift()
            return
        if self._state == self.STATE_DRAGGING:
            # 用户还按着鼠标：先替他松开（release 幂等），再走程序化掀开
            try:
                sim.release(fling=False)
            except Exception as exc:
                _log("RUG-WARN", f"lift 前释放用户抓点异常：{exc}")
        # 抓"最靠角落"的粒子：x+y 最大者（必然命中，grab 的是粒子自身坐标）
        try:
            V = sim.vertices()
            i = int(np.argmax(V[:, 0] + V[:, 1]))
            px, py = float(V[i, 0]), float(V[i, 1])
            sim.grab(px, py, 40.0)
            sim.wake()
        except Exception as exc:
            _log("RUG-ERROR", f"lift 抓角异常：{exc}")
            self._finish_lift()
            return
        self._lift_from = (px, py)
        self._lift_t0 = time.monotonic()
        self._set_state(self.STATE_LIFTING)
        self._lift_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            1.0 / 30.0, self, "liftStep:", None, True)
        self._arm_poll(LIFT_TIMEOUT_S)

    def quit(self) -> None:
        """彻底退出（S6）：清理 → terminate → os._exit(0) 兜底。"""
        _log("RUG-INFO", "退出：清理 sensor/overlay → terminate → os._exit")
        self._cleanup()
        try:
            NSApplication.sharedApplication().terminate_(None)
        except Exception:
            pass
        os._exit(0)

    def _install_sigint(self) -> None:
        """Ctrl-C 走与菜单「退出」同一条路（S6）。

        不能依赖 `try: app.run() except KeyboardInterrupt`：NSApp.run() 是 C 循环，
        PyObjC 又会吞掉 objc 回调内抛出的 KI——外层 except 永不触发（2026-10-08 实测：
        SIGINT 发出 5s 后进程仍在）。信号处理器由解释器在字节码边界执行（闸门 timer
        5~30Hz 保证边界持续出现），这里直接调 quit()（清理 → terminate → os._exit 兜底）。
        """

        def _on_sigint(_sig, _frm):
            _log("RUG-INFO", "Ctrl-C，退出")
            try:
                self.quit()
            except Exception:
                os._exit(0)  # 兜底：quit() 半途异常也必须退出

        signal.signal(signal.SIGINT, _on_sigint)

    # ---- 状态机内部 ----
    def _set_state(self, s: str) -> None:
        if self._state == s:
            return
        self._state = s
        self._refresh_toggle_item()
        if self._edit_item is not None:
            self._edit_item.setEnabled_(s != self.STATE_IDLE)
        _log("RUG-INFO", f"state -> {s}")

    def _refresh_toggle_item(self) -> None:
        """主菜单项标题：铺上（没铺过）→ 掀开（已铺上）→ 摆放回来（掀开后还记得位置）。"""
        if self._toggle_item is None:
            return
        if self._state != self.STATE_IDLE:
            title = "掀开"
        elif self._saved_placement is not None:
            title = "摆放回来"
        else:
            title = "铺上"
        self._toggle_item.setTitle_(title)

    def _arm_poll(self, timeout: float) -> None:
        self._cancel_poll()
        self._poll_deadline = time.monotonic() + timeout
        self._poll_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            0.2, self, "pollState:", None, True)

    def _cancel_poll(self) -> None:
        if self._poll_timer is not None:
            self._poll_timer.invalidate()
            self._poll_timer = None

    def _cancel_lift_timer(self) -> None:
        if self._lift_timer is not None:
            self._lift_timer.invalidate()
            self._lift_timer = None

    def pollState_(self, timer) -> None:
        """throwing/lifting 的落定轮询：is_asleep 或超时兜底。"""
        if self._state not in (self.STATE_THROWING, self.STATE_LIFTING):
            self._cancel_poll()
            return
        sim = self._overlay.sim if self._overlay is not None else None
        now = time.monotonic()
        if self._state == self.STATE_THROWING:
            if (sim is not None and sim.is_asleep()) or now >= self._poll_deadline:
                self._cancel_poll()
                self._set_state(self.STATE_RESTING)
        else:  # lifting
            if sim is None or sim.is_asleep() or now >= self._poll_deadline:
                self._cancel_poll()
                self._finish_lift()

    def _finish_lift(self) -> None:
        self._cancel_lift_timer()
        if self._overlay is not None:
            self._overlay.hide()
        self._set_state(self.STATE_IDLE)

    def liftStep_(self, timer) -> None:
        """掀开动画：0.6s 内把抓点沿直线拖到屏外左上，然后 release(fling=True)。"""
        sim = self._overlay.sim if self._overlay is not None else None
        if sim is None:
            self._cancel_lift_timer()
            return
        t = (time.monotonic() - self._lift_t0) / LIFT_DRAG_S
        tx, ty = self._w * LIFT_TARGET_FRAC, self._h * LIFT_TARGET_FRAC
        if t >= 1.0:
            self._cancel_lift_timer()
            try:
                sim.drag_to(tx, ty)
                sim.release(fling=True)
            except Exception as exc:
                _log("RUG-ERROR", f"lift release 异常：{exc}")
        else:
            fx, fy = self._lift_from
            try:
                sim.drag_to(fx + (tx - fx) * t, fy + (ty - fy) * t)
            except Exception as exc:
                _log("RUG-ERROR", f"lift drag 异常：{exc}")

    # ---- overlay 回调（主线程） ----
    def _on_grab(self) -> None:
        if self._state == self.STATE_RESTING:
            self._set_state(self.STATE_DRAGGING)

    def _on_release(self) -> None:
        if self._state == self.STATE_DRAGGING:
            self._set_state(self.STATE_RESTING)

    # ---- overlay / sim_factory ----
    def _sim_factory(self):
        """闭包注入当前 BumpField；每次 show() 由 overlay 调用新建 sim（契约）。

        毯子小于屏幕（设计变更 A）：ClothSim(w*RUG_W_FRAC, h*RUG_H_FRAC)，其余
        （throw_in 落点 = 屏中心、菜单、状态机）不变。
        """
        def factory(w: float, h: float):
            return ClothSim(float(w) * float(self.settings.get("rug_w_frac")),
                            float(h) * float(self.settings.get("rug_h_frac")),
                            bumps=self._bumps)
        return factory

    def _ensure_overlay(self) -> None:
        if self._overlay is None:
            self._overlay = RugOverlay(
                self._screen, self._current_texture(), self._sim_factory(),
                on_grab=self._on_grab, on_release=self._on_release, fps=30,
                menu_provider=self._make_context_menu)

    def _current_texture(self) -> str:
        if self._texture_paths:
            return self._texture_paths[min(self._texture_idx, len(self._texture_paths) - 1)]
        return os.path.join(self._assets_dir, "rug-01.png")  # overlay 会 WARN 并回退纯色

    def _switch_texture(self, path: str) -> None:
        """地毯款式切换：hide 旧 RugOverlay、以新贴图重建（契约）。"""
        try:
            self._texture_idx = self._texture_paths.index(path)
        except ValueError:
            self._texture_paths.append(path)
            self._texture_idx = len(self._texture_paths) - 1
        self._refresh_style_menu()
        _log("RUG-INFO", f"切换地毯款式：{os.path.basename(path)}")
        if self._overlay is None:
            return
        was_shown = self._overlay.is_shown()
        keep = None
        try:
            if was_shown:
                keep = tuple(self._overlay.placement)   # 保留用户调好的摆放
        except Exception:
            keep = None
        self._cancel_poll()
        self._cancel_lift_timer()
        self._overlay.hide()
        self._overlay = None  # 以新贴图重建
        if was_shown:
            self._set_state(self.STATE_IDLE)
            self.lay_down()  # 重新走一遍铺上动画，状态机自洽
            if keep is not None:                        # 摆回原位（换图不丢位置/大小/角度）
                try:
                    self._overlay.apply_placement(keep)
                except Exception:
                    pass

    # ---- IconSensor 链路（主线程） ----
    def _init_sensor(self) -> None:
        """query 失败 → 降级；calibration 失败 → 回退 (0.0, 24.0)，隆起继续可用。

        契约 interfaces.py:295：calibration 失败由 RugApp 回退近似值 (0.0, 24.0)
        并打 RUG-WARN，**不**整体降级；只有 query / 权限失败才 degraded。
        """
        try:
            self._sensor = IconSensor()
            icons = self._sensor.query()
        except IconSensorError as exc:
            self._enter_degraded(f"权限/感知失败（{exc}）")
            return
        except Exception as exc:  # 意外异常同样降级，不崩（contract.md §5）
            self._enter_degraded(f"感知异常（{exc}）")
            return
        try:
            cal = self._sensor.calibration()
        except Exception as exc:
            cal = (0.0, 24.0)  # 菜单栏高度近似（契约回退值）
            _log("RUG-WARN", f"Finder 标定失败，回退 calibration=({cal[0]:.1f}, {cal[1]:.1f})，"
                             f"图标隆起继续可用：{exc}")
        self._bumps = BumpField(icons, self._w, self._h, calibration=cal)
        self._degraded = False
        _log("RUG-INFO", f"图标隆起就绪：{len(icons)} 个图标，calibration=({cal[0]:.1f}, {cal[1]:.1f})")
        try:
            self._sensor.start(self._on_icons_changed)
        except Exception as exc:
            _log("RUG-WARN", f"sensor.start 失败，桌面变化通知不可用（隆起保留）：{exc}")

    def _enter_degraded(self, reason: str) -> None:
        self._degraded = True
        self._bumps = None
        self._sensor = None
        if self._degraded_item is not None:
            self._degraded_item.setHidden_(False)
        _log("RUG-WARN", f"图标隆起不可用，进入降级模式（无隆起、物理照常、不二次弹权限）：{reason}")

    def _on_icons_changed(self) -> None:
        """sensor 回调：保证主线程、须快速返回 → 只置脏标志（契约）。"""
        self._icons_dirty = True

    def _start_dirty_timer(self) -> None:
        self._dirty_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            DIRTY_DEBOUNCE_S, self, "pollDirty:", None, True)

    def pollDirty_(self, timer) -> None:
        if not self._icons_dirty:
            return
        self._icons_dirty = False
        self._rebuild_bumps()

    def _rebuild_bumps(self) -> None:
        """重新 query + 重建 BumpField → set_bumps + wake()（G4：2s 内更新）。"""
        if self._degraded or self._sensor is None:
            return
        try:
            icons = self._sensor.query()
            cal = self._sensor.calibration()  # 实例内已缓存，快
            self._bumps = self._make_bumps(icons, cal)
        except IconSensorError as exc:
            _log("RUG-WARN", f"图标重查失败，沿用旧隆起：{exc}")
            return
        except Exception as exc:
            _log("RUG-WARN", f"图标重查异常，沿用旧隆起：{exc}")
            return
        sim = self._overlay.sim if self._overlay is not None else None
        if sim is not None:
            sim.set_bumps(self._bumps)
            sim.wake()
        _log("RUG-INFO", f"隆起重建：{len(icons)} 个图标")

    # ---- 设置（settings_ui.py 的窗口回调这些方法） ----
    def _make_bumps(self, icons, cal):
        """按设置构造 BumpField（隆起开关/高度倍率）；关闭时返回 None（无隆起）。"""
        if not bool(self.settings.get("bump_enabled")):
            return None
        scale = float(self.settings.get("bump_scale") or 1.0)
        return BumpField(icons, self._w, self._h, calibration=cal,
                         base_height_pt=10.0 * scale)

    def assets_dir(self) -> str:
        return self._assets_dir

    def rescan_textures(self) -> list:
        """重新扫描 assets/rugs/rug-*.png，重建款式菜单，返回路径列表。"""
        self._texture_paths = sorted(
            _p for _p in glob.glob(os.path.join(self._assets_dir, "rug-*.png"))
            if not _p.endswith("-back.png"))   # -back.png 是材质背面（v4），不是款式
        if self._texture_idx >= len(self._texture_paths):
            self._texture_idx = 0
        if self._style_submenu is not None:
            self._style_submenu.removeAllItems()
            if self._texture_paths:
                for i, p in enumerate(self._texture_paths):
                    it = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                        os.path.basename(p), "menuPickRug:", "")
                    it.setTarget_(self)
                    it.setRepresentedObject_(p)
                    it.setState_(NSControlStateValueOn if i == self._texture_idx
                                 else NSControlStateValueOff)
                    self._style_submenu.addItem_(it)
            else:
                it = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                    "（无 rug-*.png，见 assets/rugs/README）", None, "")
                it.setEnabled_(False)
                self._style_submenu.addItem_(it)
        _log("RUG-INFO", f"重新扫描地毯贴图：{len(self._texture_paths)} 款")
        return list(self._texture_paths)

    @objc.python_method
    def apply_settings_texture(self, path: str) -> None:
        """设置窗选中的贴图：走款式切换（换图保留摆放）。"""
        self.settings.set("texture", os.path.basename(path))
        self._switch_texture(path)

    @objc.python_method
    def apply_settings_default_size(self) -> None:
        """默认大小改了：如果毯子正铺着且不在编辑模式，立刻按新尺寸摆一次。"""
        if self._overlay is None or self._state == self.STATE_IDLE or self._overlay.edit_mode:
            return
        p = self._default_placement()
        cur = self._overlay.placement
        if cur is not None:
            p = (cur[0], cur[1], p[2], p[3], cur[4])   # 位置/角度不打扰，只改尺寸
        self._overlay.apply_placement(p)

    @objc.python_method
    def apply_settings_angle(self) -> None:
        """默认角度改了：正铺着且不在编辑模式时立刻应用。"""
        if self._overlay is None or self._state == self.STATE_IDLE or self._overlay.edit_mode:
            return
        cur = self._overlay.placement
        if cur is None:
            return
        self._overlay.apply_placement((cur[0], cur[1], cur[2], cur[3],
                                       math.radians(float(self.settings.get("angle_deg")))))

    @objc.python_method
    def apply_settings_bumps(self) -> None:
        """隆起开关/高度倍率改了：重建 BumpField 并唤醒。"""
        if self._degraded or self._sensor is None:
            return
        try:
            icons = self._sensor.query()
            self._bumps = self._make_bumps(icons, self._sensor.calibration())
        except Exception as exc:
            _log("RUG-WARN", f"隆起设置应用失败（沿用旧值）：{exc}")
            return
        sim = self._overlay.sim if self._overlay is not None else None
        if sim is not None:
            sim.set_bumps(self._bumps)
            sim.wake()
        _log("RUG-INFO", f"隆起设置：enabled={bool(self.settings.get('bump_enabled'))} "
                         f"×{float(self.settings.get('bump_scale')):.2f}")

    @objc.python_method
    def apply_settings_shadow(self) -> None:
        """接触阴影开关：直接切场景里阴影节点（贴边投影裙）的可见性。"""
        if self._overlay is None:
            return
        try:
            self._overlay.set_shadow_visible(bool(self.settings.get("shadow_enabled")))
        except Exception as exc:
            _log("RUG-WARN", f"接触阴影设置应用失败：{exc}")

    def menuSettings_(self, sender) -> None:
        """菜单栏「设置…」：打开设置窗（懒建 + 每次打开同步控件状态）。

        打开期间临时把应用切到 Regular（Dock 图标出现、⌘Q 可用）；关窗由
        SettingsController 回调 _settings_window_closed() 收回。
        """
        try:
            if self._settings_ctrl is None:
                self._settings_ctrl = SettingsController()
            self._settings_ctrl._configure(self, self._texture_paths)
            self._set_dock_visible(True, reason="设置窗")
            self._settings_ctrl._show()
        except Exception as exc:
            _log("RUG-ERROR", f"打开设置窗失败：{exc}")

    # ---- 菜单 ----
    def _icon_file(self) -> str | None:
        """毛毯图标文件：打包后取包内 AppIcon.icns，开发时取项目 icon.icns。"""
        here = os.path.dirname(os.path.abspath(__file__))
        for path in (os.path.join(here, "..", "AppIcon.icns"),      # .app 内（Resources/app → Resources）
                     os.path.join(here, "icon.icns")):              # 开发目录
            if os.path.isfile(path):
                return path
        return None

    def _logo_dir(self) -> str:
        """品牌图标目录（assets/logo）：打包后与源码运行都是同一个相对位置。"""
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "logo")

    def _status_icon_image(self):
        """菜单栏图标（18pt 模板图标，来自品牌 SVG 的变体）；没有就用 icns，再没有返回 None。

        模板图标（template）是菜单栏的规矩做法：纯 alpha 蒙版 + 黑色，系统按当前菜单栏
        深浅色自动反色；用彩色 icns 会在浅色菜单栏上糊成一团。图标生成见
        scripts/make_logo_assets.py（源 SVG assets/logo/applogo.svg）。

        状态栏项本身必须存在：多屏 / 菜单栏拥挤时系统可能把它放到别的屏或收进溢出区
        （macOS 26 + LaunchServices 启动路径见 menubar.py 顶部说明与 _menubar_settle），
        所以 Dock 图标（见 _set_dock_icon）与「毯上右键」才是「一定找得到」的那条路。
        """
        img = menubar.template_image(self._logo_dir())
        if img is not None:
            return img
        path = self._icon_file()
        if path is None:
            return None
        try:
            img = NSImage.alloc().initWithContentsOfFile_(path)
            if img is None:
                return None
            img.setSize_((18.0, 18.0))
            return img
        except Exception:
            return None

    def _set_dock_icon(self, app) -> None:
        """Dock 图标 = 毛毯图标（源码运行时也一致，不显示 Python 火箭图标）。"""
        path = self._icon_file()
        if path is None:
            return
        try:
            img = NSImage.alloc().initWithContentsOfFile_(path)
            if img is not None:
                app.setApplicationIconImage_(img)
        except Exception as exc:
            _log("RUG-WARN", f"Dock 图标设置失败：{exc}")

    def _build_main_menu(self) -> None:
        """主菜单：Dock 图标常驻后，用户点 Dock 图标/激活应用时菜单栏会切成它。

        纯菜单栏应用（Accessory）时代不需要主菜单；现在必须给出可发现的「退出 ⌘Q」
        与常用动作，否则用户还是只能去找菜单栏小图标。
        """
        try:
            app = NSApplication.sharedApplication()
            main = NSMenu.alloc().init()

            app_menu = NSMenu.alloc().init()
            app_menu.setAutoenablesItems_(False)
            settings_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                "设置…", "menuSettings:", ",")                       # ⌘,
            settings_item.setTarget_(self)
            app_menu.addItem_(settings_item)
            app_menu.addItem_(NSMenuItem.separatorItem())
            quit_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                "退出桌面毛毯", "menuQuit:", "q")                     # ⌘Q
            quit_item.setTarget_(self)
            app_menu.addItem_(quit_item)
            app_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("桌面毛毯", None, "")
            app_item.setSubmenu_(app_menu)
            main.addItem_(app_item)

            rug_menu = NSMenu.alloc().init()
            rug_menu.setAutoenablesItems_(False)
            for title, action, key, mask in (
                ("铺上／掀开", "menuToggleRug:", "\\", None),
                ("调整毯子（大小/角度）…", "menuEditRug:", "e", None),
                ("抚平折痕（摊平毯子）", "menuFlattenRug:", "f", None),
                ("重置位置/大小/角度", "menuResetRug:", "r", NSCommandKeyMask | NSShiftKeyMask),
            ):
                it = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, action, key)
                it.setTarget_(self)
                if mask is not None:
                    it.setKeyEquivalentModifierMask_(mask)
                rug_menu.addItem_(it)
            rug_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("毯子", None, "")
            rug_item.setSubmenu_(rug_menu)
            main.addItem_(rug_item)

            app.setMainMenu_(main)
        except Exception as exc:
            _log("RUG-WARN", f"主菜单构建失败（Dock 图标仍可用，「退出」可用菜单栏项）：{exc}")

    def applicationShouldHandleReopen_hasVisibleWindows_(self, sender, has_visible_windows) -> bool:
        """点 Dock 图标（应用重开）：设置窗开着就把它提到前面，否则切换毯子。

        正常态（Accessory）没有 Dock 图标，所以这条只在设置窗打开过（Dock 图标在）时触发。
        """
        try:
            ctrl = self._settings_ctrl
            panel = getattr(ctrl, "panel", None) if ctrl is not None else None
            if panel is not None and panel.isVisible():
                panel.makeKeyAndOrderFront_(None)
                return True
            self.menuToggleRug_(None)
        except Exception as exc:
            _log("RUG-WARN", f"Dock 重开处理失败：{exc}")
        return True

    def _set_dock_visible(self, visible: bool, reason: str = "") -> None:
        """切换「Dock 图标是否出现」：设置窗打开 → Regular；关闭 → Accessory。

        常驻状态栏应用的形态（Clash 那种）：平时只占菜单栏、不占 Dock；唤起窗口时才
        临时进 Dock（并让主菜单/⌘Q 可用），关窗即收回。
        """
        try:
            app = NSApplication.sharedApplication()
            want = (NSApplicationActivationPolicyRegular if visible
                    else NSApplicationActivationPolicyAccessory)
            if app.activationPolicy() != want:
                app.setActivationPolicy_(want)
                _log("RUG-INFO", f"激活策略 -> {'Regular（Dock 图标出现）' if visible else 'Accessory（不占 Dock）'}"
                                 + (f"：{reason}" if reason else ""))
        except Exception as exc:
            _log("RUG-WARN", f"激活策略切换失败：{exc}")

    def _settings_window_closed(self) -> None:
        """设置窗关闭（「关闭」按钮或红点都算）：把 Dock 图标收回去。"""
        self._set_dock_visible(False, reason="设置窗已关")

    def _build_menu(self) -> None:
        item = NSStatusBar.systemStatusBar().statusItemWithLength_(NSVariableStatusItemLength)
        # autosaveName：用户在菜单栏里 ⌘ 拖动过之后，系统按这个名字记住位置
        # （NSStatusItem Preferred Position <名字>），下次启动回到用户放的地方。
        try:
            item.setAutosaveName_("RugStatusItem")
        except Exception:
            pass
        proj_icon = self._status_icon_image()
        try:
            if proj_icon is not None:
                item.button().setImage_(proj_icon)
                item.button().setToolTip_("桌面毛毯")
            else:
                item.button().setTitle_("Rug")
            # 辅助功能名字：macOS 26 的「系统设置 → 控制中心 → 菜单栏」里就是按这个名字列项，
            # 不设的话那一栏只会显示系统默认的「状态菜单」，用户根本认不出哪个是本应用。
            item.button().setAccessibilityLabel_("桌面毛毯")
        except Exception:
            try:
                item.setTitle_("Rug")
            except Exception:
                pass

        menu = NSMenu.alloc().init()
        menu.setAutoenablesItems_(False)  # accessory 应用防菜单被自动禁用

        toggle = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "铺上", "menuToggleRug:", "\\")  # ⌘\
        toggle.setTarget_(self)
        menu.addItem_(toggle)
        self._toggle_item = toggle

        edit = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "调整毯子（大小/角度）…", "menuEditRug:", "e")  # ⌘E
        edit.setTarget_(self)
        edit.setEnabled_(False)          # 铺上后才可用
        menu.addItem_(edit)
        self._edit_item = edit

        flatten = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "抚平折痕（摊平毯子）", "menuFlattenRug:", "f")  # ⌘F
        flatten.setTarget_(self)
        flatten.setKeyEquivalentModifierMask_(NSCommandKeyMask)
        menu.addItem_(flatten)

        reset = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "重置位置/大小/角度", "menuResetRug:", "r")  # ⌘R
        reset.setTarget_(self)
        reset.setKeyEquivalentModifierMask_(NSCommandKeyMask | NSShiftKeyMask)  # ⌘⇧R
        menu.addItem_(reset)
        self._reset_item = reset

        settings_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "设置…", "menuSettings:", ",")  # ⌘,
        settings_item.setTarget_(self)
        menu.addItem_(settings_item)

        style = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("地毯款式", None, "")
        submenu = NSMenu.alloc().init()
        submenu.setAutoenablesItems_(False)
        if self._texture_paths:
            for i, p in enumerate(self._texture_paths):
                it = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                    os.path.basename(p), "menuPickRug:", "")
                it.setTarget_(self)
                it.setRepresentedObject_(p)
                it.setState_(NSControlStateValueOn if i == self._texture_idx
                             else NSControlStateValueOff)
                submenu.addItem_(it)
        else:
            it = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                "（无 rug-*.png，见 assets/rugs/README）", None, "")
            it.setEnabled_(False)
            submenu.addItem_(it)
        style.setSubmenu_(submenu)
        menu.addItem_(style)
        self._style_submenu = submenu

        degraded = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "图标隆起：不可用（需自动化权限）", None, "")
        degraded.setEnabled_(False)
        degraded.setHidden_(True)
        menu.addItem_(degraded)
        self._degraded_item = degraded

        menu.addItem_(NSMenuItem.separatorItem())
        quit_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("退出", "menuQuit:", "q")
        quit_item.setTarget_(self)
        menu.addItem_(quit_item)

        item.setMenu_(menu)
        self._status_item = item
        # 落位判定与「被系统隐藏」的处理统一在 _menubar_settle()（run() 里紧跟着调用）：
        # 这里读到的 frame 在系统布局完成前是 34×0 @ (0,0) 的假值，别在这里下结论。

    def _menubar_settle(self) -> None:
        """菜单栏项落位判定 + 必要时交接（背景与实测数据见 menubar.py 顶部）。

        三种角色：
          · 普通进程（本机从 shell/launchctl 启动，或系统本来就没挤）——落位成功，继续；
          · 父进程（LaunchServices 启动，= 双击/登录项）——落位失败则拉起子进程接棒，
            子进程拿到落位就退出自己（子进程是新的一等公民：它自己会铺毯、建菜单）；
          · 交接子进程（`--menubar-handoff <token>`）——落位失败就把结果写回报告文件并退出，
            让父进程继续（父进程的项虽然被隐藏，但至少应用还在跑，右键毯子照常可用）。
        """
        if self._status_item is None:
            return
        ok, dt = menubar.await_slot(self._status_item)
        self._menubar_ok = ok
        self._log_status_item_geometry()      # 落位后的真实坐标（多屏排障用）
        if ok:
            _log("RUG-INFO", f"菜单栏项已落位（{dt:.1f}s）")
            if self._handoff_token:
                menubar.write_report(self._handoff_report, self._handoff_token, True,
                                     menubar.read_frame(self._status_item))
            return
        if self._handoff_token:
            _log("RUG-WARN", f"交接子进程也未拿到菜单栏落位（{dt:.1f}s），退出交回父进程")
            menubar.write_report(self._handoff_report, self._handoff_token, False, None)
            self._cleanup()
            os._exit(0)
        fr = menubar.read_frame(self._status_item)
        _log("RUG-WARN", f"菜单栏项未落位（{dt:.1f}s 后 frame={fr}）——macOS 把它放到了屏幕外；"
                         f"右键毯子仍可打开菜单")
        if os.environ.get("RUG_NO_MENUBAR_HANDOFF") == "1":
            _log("RUG-INFO", "RUG_NO_MENUBAR_HANDOFF=1：不交接（保持本进程）")
            return
        self._handoff_to_child()

    def _handoff_argv(self) -> list[str]:
        """再拉起一份自己的命令行（同一个 Python 运行时、同一个脚本，只带启动模式）。"""
        argv = [sys.executable, os.path.abspath(__file__)]
        if not self._demo:
            argv.append("--start-hidden")
        return argv

    def _handoff_to_child(self) -> None:
        """父进程侧：把应用交给「非 LaunchServices 启动」的子进程（见 menubar.py 顶部）。"""
        if self._instance_lock is not None:
            try:
                self._instance_lock.close()      # 让出单实例锁，子进程才起得来
            except OSError:
                pass
            self._instance_lock = None
        token, pid = menubar.spawn_handoff(self._handoff_argv(), self._handoff_report)
        if pid is None:
            _log("RUG-WARN", "菜单栏交接：子进程没起来")
            self._menubar_hint()
            return
        _log("RUG-INFO", f"菜单栏交接：拉起子进程 pid={pid} token={token}")
        rep = menubar.wait_report(self._handoff_report, token)
        if rep and rep.get("visible"):
            _log("RUG-INFO", f"菜单栏交接成功：子进程 pid={rep.get('pid')} 已拿到菜单栏项，"
                             f"本进程退出（frame={rep.get('frame')}）")
            self._cleanup()
            os._exit(0)
        _log("RUG-WARN", f"菜单栏交接失败：子进程报告={rep}")
        self._menubar_hint()

    def _menubar_hint(self) -> None:
        """交接也失败时的兜底说明（每版本一次）。"""
        menubar.show_hidden_hint(self.settings, log=_log)

    def _log_status_item_geometry(self) -> None:
        """记录菜单栏项落在哪块屏/哪个坐标（多屏排障用；2026-10-09 踩坑）。

        注意：macOS 26 下这个窗口由系统（控制中心）渲染，用 CGWindowListCopyWindowInfo
        按本进程 PID 查是查不到的——只能在进程内用 button.window() 查；而且**未落位时
        读到的是 34×0 @ (0,0) 的假值**，可见性结论一律以 menubar.await_slot() 为准。
        """
        try:
            if self._status_item is None:
                return
            w = self._status_item.button().window()
            if w is None:
                _log("RUG-WARN", "菜单栏项没有窗口（系统还没给它建窗口）")
                return
            f = w.frame()
            scr = w.screen()
            scr_desc = (f"{scr.frame().size.width:.0f}x{scr.frame().size.height:.0f}"
                        if scr is not None else "?")
            _log("RUG-INFO", f"菜单栏项 x={f.origin.x:.0f} y={f.origin.y:.0f} "
                             f"{f.size.width:.0f}x{f.size.height:.0f} "
                             f"visible={bool(w.isVisible())} 所在屏={scr_desc}")
        except Exception as exc:
            _log("RUG-WARN", f"菜单栏项定位失败：{exc}")

    def _refresh_style_menu(self) -> None:
        if self._style_submenu is None:
            return
        current = self._current_texture()
        for it in self._style_submenu.itemArray():
            try:
                it.setState_(NSControlStateValueOn if str(it.representedObject()) == current
                             else NSControlStateValueOff)
            except Exception:
                pass

    def _context_toggle_title(self) -> str:
        """右键菜单第一项标题（随状态）：已铺上 = 掀开收起，否则 = 铺上/摆放回来。"""
        if self._state != self.STATE_IDLE:
            return "掀开毯子（收起）"
        if self._saved_placement is not None:
            return "摆放回来"
        return "铺上毯子"

    def _make_context_menu(self, x: float, y: float):
        """毯子上右键（或 Ctrl+左键）弹出的上下文菜单。

        每次现建：勾选/标题/可用性都按当前状态给，不维护第二份菜单状态。
        (x, y) 是主屏本地点击点（当前版本不按点位做内容差异，留给后续扩展）。
        """
        menu = NSMenu.alloc().init()
        menu.setAutoenablesItems_(False)

        toggle = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            self._context_toggle_title(), "menuToggleRug:", "")
        toggle.setTarget_(self)
        menu.addItem_(toggle)
        menu.addItem_(NSMenuItem.separatorItem())
        _log("RUG-INFO", f"毯上右键菜单：({x:.0f},{y:.0f}) 状态={self._state}")

        size_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("调整尺寸", None, "")
        size_menu = NSMenu.alloc().init()
        size_menu.setAutoenablesItems_(False)
        cur_w = float(self.settings.get("rug_w_frac"))
        for key, label, (pw, _ph) in (
            ("small", "小（占屏宽 34%）", _SIZE_PRESETS["small"]),
            ("medium", "中（52%）", _SIZE_PRESETS["medium"]),
            ("large", "大（70%）", _SIZE_PRESETS["large"]),
        ):
            it = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                label, "menuSizePreset:", "")
            it.setTarget_(self)
            it.setRepresentedObject_(key)
            it.setState_(NSControlStateValueOn if abs(cur_w - pw) < 0.005
                         else NSControlStateValueOff)
            size_menu.addItem_(it)
        size_menu.addItem_(NSMenuItem.separatorItem())
        edit = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "拖拽调整（四角缩放 / 旋转）…", "menuEditRug:", "")
        edit.setTarget_(self)
        edit.setEnabled_(self._state != self.STATE_IDLE)
        size_menu.addItem_(edit)
        size_item.setSubmenu_(size_menu)
        menu.addItem_(size_item)

        for title, action in (("抚平折痕（摊平毯子）", "menuFlattenRug:"),
                              ("重置位置/大小/角度", "menuResetRug:")):
            it = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, action, "")
            it.setTarget_(self)
            menu.addItem_(it)

        menu.addItem_(NSMenuItem.separatorItem())
        for title, action in (("设置…", "menuSettings:"), ("退出桌面毛毯", "menuQuit:")):
            it = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, action, "")
            it.setTarget_(self)
            menu.addItem_(it)
        return menu

    def menuSizePreset_(self, sender) -> None:
        """右键菜单「调整尺寸 → 小/中/大」：改写默认大小；已铺上且不在编辑态时立刻生效。"""
        key = str(sender.representedObject())
        w, h = _SIZE_PRESETS.get(key, _SIZE_PRESETS["medium"])
        self.settings.set("rug_w_frac", w)
        self.settings.set("rug_h_frac", h)
        self.apply_settings_default_size()
        _log("RUG-INFO", f"尺寸预设 {key}：宽 {w*100:.0f}% × 高 {h*100:.0f}% 屏占比")

    def menuToggleRug_(self, sender) -> None:
        if self._state != self.STATE_IDLE:
            self.lift()
        elif self._saved_placement is not None:
            self.place_back()      # 掀开后 → 原样摆回来
        else:
            self.lay_down()

    def menuEditRug_(self, sender) -> None:
        """「调整毯子（大小/角度）」：切换编辑模式（四角缩放柄 + 旋转柄）。"""
        if self._overlay is None or self._state == self.STATE_IDLE:
            return
        on = not self._overlay.edit_mode
        try:
            self._overlay.set_edit_mode(on)
        except Exception as exc:
            _log("RUG-ERROR", f"切换编辑模式失败：{exc}")
            return
        if self._edit_item is not None:
            self._edit_item.setState_(NSControlStateValueOn if on else NSControlStateValueOff)
            self._edit_item.setTitle_("完成调整（大小/角度）" if on else "调整毯子（大小/角度）…")

    def menuFlattenRug_(self, sender) -> None:
        """「抚平折痕」：把压缩产生的折痕清掉（摊平），位置/大小/角度不变。"""
        if self._overlay is None or self._state == self.STATE_IDLE:
            return
        sim = self._overlay.sim
        if sim is None:
            return
        try:
            sim.flatten_folds()
        except Exception as exc:
            _log("RUG-WARN", f"抚平折痕失败：{exc}")
            return
        _log("RUG-INFO", "抚平折痕：塑性折痕场已清零")

    def menuResetRug_(self, sender) -> None:
        """「重置位置/大小/角度」：回到默认摆放（屏中心 + 默认尺寸 + 默认角度）。"""
        p = self._default_placement()
        self._saved_placement = None
        if self._overlay is not None and self._state != self.STATE_IDLE:
            try:
                self._overlay.apply_placement(p)
            except Exception as exc:
                _log("RUG-WARN", f"重置摆放失败：{exc}")
        self._refresh_toggle_item()
        _log("RUG-INFO", f"重置摆放：{p[2]:.0f}x{p[3]:.0f} @ ({p[0]:.0f},{p[1]:.0f}) "
                         f"{float(self.settings.get('angle_deg')):.0f}°")

    def menuPickRug_(self, sender) -> None:
        try:
            path = str(sender.representedObject())
        except Exception:
            return
        if path:
            self._switch_texture(path)

    def menuQuit_(self, sender) -> None:
        self.quit()

    # ---- selftest（契约：探针行无 RUG- 前缀，供脚本 grep） ----
    def _arm_selftest(self) -> None:
        self.lay_down()
        NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            2.0, self, "selftestProbe:", None, False)
        NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            5.0, self, "selftestFinish:", None, False)

    def selftestProbe_(self, timer) -> None:
        level = -1
        expected = icon_level_base() + PANEL_LEVEL_DELTA
        try:
            infos = CGWindowListCopyWindowInfo(kCGWindowListOptionAll, kCGNullWindowID)
            pid = os.getpid()
            fallback = -1
            for info in infos:
                try:
                    if int(info.get("kCGWindowOwnerPID", -1)) != pid:
                        continue
                    layer = int(info.get("kCGWindowLayer", 0))
                except Exception:
                    continue
                if layer == expected:
                    level = layer
                    break
                if layer < 0 and fallback == -1:
                    fallback = layer  # 本进程的其他负层窗口（兜底线索）
            if level == -1 and fallback != -1:
                level = fallback
        except Exception as exc:
            _log("RUG-ERROR", f"selftest 窗口探针失败：{exc}")
        print(f"SELFTEST-LEVEL {level}", flush=True)
        self._selftest_ok = (level == expected)

    def selftestFinish_(self, timer) -> None:
        print("SELFTEST-OK" if self._selftest_ok else "SELFTEST-FAIL", flush=True)
        os._exit(0 if self._selftest_ok else 1)

    # ---- 清理 ----
    def _cleanup(self) -> None:
        self._cancel_poll()
        self._cancel_lift_timer()
        if self._dirty_timer is not None:
            self._dirty_timer.invalidate()
            self._dirty_timer = None
        if self._sensor is not None:
            try:
                self._sensor.stop()
            except Exception as exc:
                _log("RUG-WARN", f"sensor.stop 异常：{exc}")
        if self._overlay is not None:
            try:
                self._overlay.hide()
            except Exception as exc:
                _log("RUG-WARN", f"overlay.hide 异常：{exc}")


def parse_args(argv: list[str] | None = None):
    """解析命令行：--selftest（无人值守自检）、--start-hidden（不自动铺上）。

    默认启动即铺上（双击 app 屏幕立刻有毯子）；--demo 是历史选项，保留兼容
    （等价默认行为）。
    """
    p = argparse.ArgumentParser(prog="rug", description="桌面毛毯（macOS Demo）")
    p.add_argument("--selftest", action="store_true",
                   help="无人值守自检：跳过 IconSensor，验证覆盖窗口层级后自动退出")
    p.add_argument("--demo", action="store_true",
                   help="兼容选项：启动即自动铺上（现在是默认行为）")
    p.add_argument("--start-hidden", dest="start_hidden", action="store_true",
                   help="启动时不铺上，只驻留菜单栏（旧默认行为）")
    p.add_argument("--menubar-handoff", dest="menubar_handoff", default=None,
                   metavar="TOKEN",
                   help="内部选项（交接子进程用）：菜单栏项落位结果写回 TOKEN 对应的报告文件")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """rug.py 入口：日志 → 单实例锁 → parse_args → RugApp(...).run()。"""
    _setup_file_logging()
    args = parse_args(argv)
    lock = None
    if not args.selftest:   # selftest 是诊断模式：允许与常驻实例并存
        lock = _acquire_instance_lock()
        if lock is None:
            _log("RUG-INFO", "已有实例在运行（单实例锁），本次启动直接退出")
            return 0
    app = RugApp(selftest=bool(getattr(args, "selftest", False)),
                 demo=not bool(getattr(args, "start_hidden", False)),
                 menubar_handoff=getattr(args, "menubar_handoff", None))
    app._instance_lock = lock     # 菜单栏交接要先把锁让给子进程（见 _handoff_to_child）
    try:
        return app.run()
    except KeyboardInterrupt:
        try:
            app._cleanup()
        except Exception:
            pass
        os._exit(0)
    return 0  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main() or 0)
