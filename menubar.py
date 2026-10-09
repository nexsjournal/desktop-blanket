"""menubar.py — 菜单栏项：模板图标、落位判定，以及 macOS 26 的「LaunchServices 启动拿不到
菜单栏项」交接（handoff）。

背景（2026-10-09 真机实证，macOS 26 + 刘海内建屏 1800×1169，单屏）：
  · 同一份代码，同一 bundle id：
      - `open -a 桌面毛毯.app`（= 双击 / 登录项 / LaunchServices）启动 → 菜单栏项窗口
        34×0 @ (0,0)、`isVisible()=False`、AX 读到 x≈-1115：被系统挤到屏外，用户看不到图标；
      - 从 shell 或 `launchctl` 启动 → 同一项落在 x≈1101、34×39，正常可见。
  · 该屏菜单栏并未满：探针进程当场能拿到 999/1033/1067/1101 四个位置；也不是图标类型问题
    （.icns 非 template / template PNG / 纯画出来的图，四种都试过）。
  · 把「LS 启动的进程」作为父进程再拉起一个子进程，**子进程的菜单栏项能正常落位** →
    这就是 handoff：父进程发现自己拿不到落位时，把活交给子进程，自己退出。
  · 进程内读数的可信度：只有「已落位」时 `button().window().frame()` 才是真值（高 ≥10）；
    未落位时读到的是 34×0 @ (0,0) 的假值。AX（System Events）读到的坐标才是外部真相。

对外只暴露函数，不建单例：调用点全在 rug.py（见 `_menubar_settle` / `_handoff_to_child`）。
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import uuid

import objc  # noqa: F401
from AppKit import NSImage
from Foundation import NSData, NSDate, NSRunLoop

__all__ = ["template_image", "read_frame", "settled", "await_slot", "spawn_handoff",
           "write_report", "wait_report", "report_path", "show_hidden_hint", "version_string"]

MENUBAR_PT = 18.0          # 状态栏标准图标尺寸
GATE_MAX_S = 1.2           # 落位判定上限：正常落位在 1s 内完成，超过即认为被系统隐藏
GATE_STEP_S = 0.2          # 每次泵 runloop 的时长（系统的布局消息要走 runloop）
HANDOFF_WAIT_S = 15.0      # 父进程等子进程报告的时限
HINT_VERSION_KEY = "menubar_hint_version"   # 每个版本只弹一次提示
SETTINGS_URL = "x-apple.systempreferences:com.apple.ControlCenter-Settings.extension"


def template_image(logo_dir: str):
    """菜单栏模板图标（assets/logo/menubar{,@2x,@3x}.png）→ 18pt NSImage(template)。

    缺文件返回 None，调用方退回旧的 icns / 文字方案。三档 rep 一次性装进同一个 NSImage，
    1x/2x/3x 屏各自取最清晰的那档。
    """
    try:
        from AppKit import NSBitmapImageRep
        img = NSImage.alloc().initWithSize_((MENUBAR_PT, MENUBAR_PT))
        reps = 0
        for suffix in ("@3x", "@2x", ""):
            path = os.path.join(logo_dir, f"menubar{suffix}.png")
            if not os.path.isfile(path):
                continue
            with open(path, "rb") as fh:
                raw = fh.read()
            rep = NSBitmapImageRep.imageRepWithData_(
                NSData.dataWithBytes_length_(raw, len(raw)))
            if rep is None:
                continue
            img.addRepresentation_(rep)
            reps += 1
        if reps == 0:
            return None
        img.setSize_((MENUBAR_PT, MENUBAR_PT))
        img.setTemplate_(True)   # 纯 alpha 蒙版，系统按菜单栏配色自动反色
        return img
    except Exception:
        return None


def read_frame(item):
    """→ (x, y, w, h, is_visible) 或 None。未落位时读到 34×0@(0,0) 的假值，别当真值用。"""
    try:
        win = item.button().window() if item is not None else None
        if win is None:
            return None
        f = win.frame()
        return (float(f.origin.x), float(f.origin.y), float(f.size.width),
                float(f.size.height), bool(win.isVisible()))
    except Exception:
        return None


def settled(item) -> bool:
    """菜单栏项是否已落位（窗口有真实高度且可见）。"""
    fr = read_frame(item)
    return bool(fr and fr[3] >= 10.0 and fr[4])


def await_slot(item, max_s: float = GATE_MAX_S) -> tuple[bool, float]:
    """轮询等落位，返回 (是否落位, 用时秒)。期间泵 runloop —— 系统的布局消息要靠它派发。"""
    t0 = time.monotonic()
    while True:
        if settled(item):
            return True, time.monotonic() - t0
        if time.monotonic() - t0 >= max_s:
            return False, time.monotonic() - t0
        NSRunLoop.currentRunLoop().runUntilDate_(
            NSDate.dateWithTimeIntervalSinceNow_(GATE_STEP_S))


def report_path(settings_path: str) -> str:
    """交接报告文件：与 settings.json 同目录（打包 → Application Support；源码 → 项目内）。"""
    return os.path.join(os.path.dirname(settings_path), "menubar-handoff.json")


def spawn_handoff(argv: list[str], report_path_: str) -> tuple[str, int | None]:
    """父进程侧：拉起子进程接棒（子进程不是 LS 启动，能拿到菜单栏落位）。

    子进程 `start_new_session=True` 独立于父进程，父进程退出它照跑；stdout/stderr 丢弃
    （它自己会写 RUG-INFO 日志到同一个日志文件）。
    """
    token = uuid.uuid4().hex[:12]
    try:
        proc = subprocess.Popen([*argv, "--menubar-handoff", token],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                stdin=subprocess.DEVNULL, start_new_session=True)
    except OSError:
        return token, None
    return token, proc.pid


def write_report(path: str, token: str, visible: bool, frame=None) -> None:
    """子进程侧：写回「我拿到落位了吗」。原子写（临时文件 + rename）。"""
    data = {"token": token, "pid": os.getpid(), "visible": bool(visible),
            "frame": list(frame) if frame else None,
            "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)
        os.replace(tmp, path)
    except OSError:
        pass


def wait_report(path: str, token: str, timeout: float = HANDOFF_WAIT_S):
    """父进程侧：等子进程写回同一 token 的报告（期间泵 runloop）。超时/无文件 → None。"""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict) and data.get("token") == token:
                return data
        except (OSError, json.JSONDecodeError):
            pass
        NSRunLoop.currentRunLoop().runUntilDate_(
            NSDate.dateWithTimeIntervalSinceNow_(0.2))
    return None


def version_string() -> str:
    """当前版本号（打包 → Info.plist；源码 → "dev"）。"""
    try:
        from AppKit import NSBundle
        info = NSBundle.mainBundle().infoDictionary() or {}
        return str(info.get("CFBundleShortVersionString") or "dev")
    except Exception:
        return "dev"


def show_hidden_hint(settings, log=print) -> bool:
    """菜单栏项被系统隐藏且交接也失败时，弹一次说明（每版本一次）。返回是否弹了。"""
    version = version_string()
    if settings.get(HINT_VERSION_KEY) == version:
        return False
    try:
        from AppKit import (NSAlert, NSApplication, NSWorkspace,
                            NSAlertFirstButtonReturn, NSAlertSecondButtonReturn)
        from Foundation import NSURL
        alert = NSAlert.alloc().init()
        alert.setMessageText_("菜单栏图标被 macOS 收起来了")
        alert.setInformativeText_(
            "macOS 在菜单栏拥挤时会把新图标放出屏幕外（本机当前就是这样），"
            "应用自己没法强行占位。\n\n"
            "想让图标出现，任选一种：\n"
            "① 系统设置 → 控制中心 → 菜单栏：把「桌面毛毯」放行；\n"
            "② 退出/减少几个其它菜单栏图标（如 clash-verge、PixPin、ChatGPT），再重启本应用。\n\n"
            "不影响使用：在毯子上点右键就是完整菜单（掀开／调整尺寸／设置／退出）。")
        alert.addButtonWithTitle_("打开系统设置")
        alert.addButtonWithTitle_("知道了")
        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
        resp = alert.runModal()
        log("RUG-INFO", f"菜单栏图标隐藏提示已展示（用户选择={resp}）")
        if resp == NSAlertFirstButtonReturn:
            NSWorkspace.sharedWorkspace().openURL_(NSURL.URLWithString_(SETTINGS_URL))
        try:
            settings.set(HINT_VERSION_KEY, version)
            from settings_ui import save_settings
            save_settings(settings)
        except Exception:
            pass
        return True
    except Exception as exc:
        log("RUG-WARN", f"菜单栏图标隐藏提示弹窗失败：{exc}")
        return False
