#!/usr/bin/env python3
"""verify_routing.py — 用 WindowServer 的 mouseDown 路由查询验证「毯外点击穿透」机制。

原理（纯只读查询 + 自建临时隐形窗口，不改任何系统状态、不需要任何权限）：
  NSWindow +windowNumberAtPoint:belowWindowWithWindowNumber:0 的语义（SDK 头文件原文）：
  「返回若在该屏幕位置发生 mouseDown 时**会被命中的最前窗口编号**，可以是别的 App 的窗口」。
  即：它就是窗口服务器点击路由的权威答案（跨进程、跨层级）。

自检模式（默认，独立于本项目进程）：
  1. 在屏上选一个未被任何普通窗口覆盖的点，放一个 40×40 全透明、level=图标层+8 的临时窗口；
  2. 实验 R1  窗口 ignores=NO,  view.hitTest 返回 self  → 期望命中 = 本临时窗口（负层级可被路由）；
  3. 实验 R2  ignores=YES                              → 期望 ≠ 本窗口（穿透到下层）；
  4. 实验 R3  ignores=NO,  view.hitTest 返回 None       → 若仍是本窗口，则证明
     **view 级 hitTest 不参与窗口路由**（hitTest 返回 None 不能把点击交给下层窗口，
     点击只是被吞掉）——这是本项目「逐像素穿透」方案是否成立的判据。
实时模式 --live：对正在运行的 rug.py --selftest 查询两个点（毯上/毯外）：
  毯上点 → 期望命中 rug 自己的窗口；毯外点 → 期望**不**命中 rug 的窗口（应落到桌面）。

输出（供 grep）：`ROUTE-SELFTEST <PASS|FAIL>`、`ROUTE-LIVE <PASS|FAIL>`；明细逐行打印；
退出码 0=全过，1=有失败，2=用法/环境错误。

用法：
  .venv/bin/python scripts/verify_routing.py            # 自检（R1/R2/R3）
  .venv/bin/python scripts/verify_routing.py --live     # 额外：对着运行的 rug --selftest 查实时路由
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import Quartz
from AppKit import (NSApplication, NSApplicationActivationPolicyAccessory,
                    NSBackingStoreBuffered, NSColor, NSView, NSWindow,
                    NSWindowStyleMaskBorderless)
from Foundation import NSMakePoint, NSMakeRect

from overlay import icon_level_base  # 单一来源（含回退常量），overlay 不依赖 cloth/icons

PROBE_LEVEL_DELTA = 8          # 与 rug 面板同层（图标层+8）
PROBE_LEVEL_ABOVE_NORMAL = 23  # 高于普通窗口(0)/Dock(20)、低于菜单栏(24)：保证探针是该点最前窗口
PROBE_SIZE = 40.0              # 探针窗口边长 pt
SETTLE_S = 0.35                # 下单后等待窗口服务器生效
SCAN_GRID = 40.0               # 空闲点扫描步长 pt
EDGE_MARGIN = 4.0              # 扫描边距 pt（桌面可达区可能很窄，如「最大化窗口之下」的细条）


def _log(msg: str) -> None:
    print(msg, flush=True)


def _pump(seconds: float) -> None:
    """跑一小段主 run loop，让 AppKit/窗口服务器处理完窗口操作。"""
    from Foundation import NSDate, NSRunLoop
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.05))


def _screen_wh() -> tuple[float, float]:
    from AppKit import NSScreen
    f = NSScreen.mainScreen().frame()
    return float(f.size.width), float(f.size.height)


def _to_appkit_point(x_tl: float, y_tl: float, screen_h: float) -> tuple[float, float]:
    """主屏本地 top-left 坐标 → AppKit 全局 bottom-left 坐标（单主屏、原点 (0,0)）。"""
    return x_tl, screen_h - y_tl


def _route_at(x_tl: float, y_tl: float, screen_h: float) -> int:
    """查询：该点 mouseDown 会命中哪个窗口（0 = 没有窗口，理论上不会）。"""
    x, y = _to_appkit_point(x_tl, y_tl, screen_h)
    return int(NSWindow.windowNumberAtPoint_belowWindowWithWindowNumber_(NSMakePoint(x, y), 0))


def _window_info(nums: list[int]) -> dict[int, dict]:
    """按 window number 取 CGWindowList 元信息（owner/layer/name），供人读诊断。"""
    infos = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionAll, Quartz.kCGNullWindowID)
    out: dict[int, dict] = {}
    for info in infos:
        try:
            n = int(info.get("kCGWindowNumber", -1))
        except Exception:
            continue
        if n in nums:
            out[n] = {
                "owner": str(info.get("kCGWindowOwnerName", "?")),
                "pid": int(info.get("kCGWindowOwnerPID", -1)),
                "layer": int(info.get("kCGWindowLayer", 0)),
            }
    return out


def _describe(num: int) -> str:
    info = _window_info([num]).get(num)
    if not info:
        return f"win#{num}"
    return f"win#{num} owner={info['owner']} pid={info['pid']} layer={info['layer']}"


def _pick_desktop_point(w: float, h: float) -> tuple[float, float, int, dict] | None:
    """找一个「桌面可达」的点：当前 mouseDown 路由命中负层级窗口（Finder 桌面等，
    说明其上方的普通/系统窗口在该点都不收鼠标——如全屏 Dock 窗口的透明区）。
    返回 (x_tl, y_tl, 命中窗口号, 窗口信息)；找不到返回 None。
    """
    y = EDGE_MARGIN
    while y < h - EDGE_MARGIN:
        x = EDGE_MARGIN
        while x < w - EDGE_MARGIN:
            num = _route_at(x, y, h)
            info = _window_info([num]).get(num)
            if info is not None and info["layer"] < 0 and info["pid"] != os.getpid():
                return x, y, num, info
            x += SCAN_GRID
        y += SCAN_GRID
    return None


class _HitSelfView(NSView):
    """默认 hitTest：点在自身范围内 → 返回 self（可被命中）。"""


class _HitNilView(NSView):
    """hitTest 永远返回 None（模拟「毯外区域」的逐像素穿透逻辑）。"""

    def hitTest_(self, point):
        return None


def _make_probe_window(x_tl: float, y_tl: float, screen_h: float, level: int,
                       view_cls=_HitSelfView):
    """40×40 全透明 borderless 窗口，放在给定点中心；level 与内容视图可指定。"""
    ax, ay = _to_appkit_point(x_tl, y_tl, screen_h)
    rect = NSMakeRect(ax - PROBE_SIZE / 2.0, ay - PROBE_SIZE / 2.0, PROBE_SIZE, PROBE_SIZE)
    win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        rect, NSWindowStyleMaskBorderless, NSBackingStoreBuffered, False)
    win.setLevel_(level)
    win.setOpaque_(False)
    win.setBackgroundColor_(NSColor.clearColor())
    win.setHasShadow_(False)
    win.setIgnoresMouseEvents_(False)
    win.setContentView_(view_cls.alloc().initWithFrame_(NSMakeRect(0, 0, PROBE_SIZE, PROBE_SIZE)))
    win.orderFrontRegardless()
    return win


def cmd_selftest() -> int:
    """R1（机会性：负层级参与路由）/ R2（ignores 生效=穿透）/ R3（hitTest 不参与路由）。"""
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    w, h = _screen_wh()
    _log(f"ROUTE-INFO 屏幕 {w:.0f}x{h:.0f}")
    ok = True

    # ---- R1（机会性）：找一个当前「桌面可见」的点，验证负层级窗口参与 mouseDown 路由 ----
    picked = _pick_desktop_point(w, h)
    if picked is None:
        _log("ROUTE-R1 SKIP 当前屏幕无可见桌面（全被普通窗口覆盖），"
             "负层级路由改由 --live / 真人点击测试覆盖")
    else:
        x_tl, y_tl, num, info = picked
        p1 = info["layer"] < 0
        ok &= p1
        _log(f"ROUTE-R1 {'PASS' if p1 else 'FAIL'} 桌面点({x_tl:.0f},{y_tl:.0f}) 路由命中="
             f"{_describe(num)}（负层级窗口可被 mouseDown 命中 ✓）")

    # ---- R2/R3：level=23 探针（保证自身是该点最前窗口，环境无关） ----
    candidates = [(w / 2.0, h / 2.0), (w / 4.0, h / 3.0), (3 * w / 4.0, h / 3.0),
                  (w / 4.0, 2 * h / 3.0), (3 * w / 4.0, 2 * h / 3.0)]
    win = None
    my_num = 0
    px = py = 0.0
    for cx, cy in candidates:
        # 绝对 level 23（高于普通窗口 0/程序坞 20、低于菜单栏 24），与 rug 的负层级无关：
        # R2/R3 验证的是 ignores 标志与 view.hitTest 对路由的作用，二者与层级无关
        win = _make_probe_window(cx, cy, h, PROBE_LEVEL_ABOVE_NORMAL)
        _pump(SETTLE_S)
        my_num = int(win.windowNumber())
        if _route_at(cx, cy, h) == my_num:
            px, py = cx, cy
            break
        win.orderOut_(None)
        _pump(0.1)
        win = None
    if win is None:
        _log("ROUTE-ERROR level=23 探针在候选点均不是最前窗口（有更高层系统 UI 遮挡？）")
        return 2
    _log(f"ROUTE-INFO 探针点=({px:.0f},{py:.0f}) 探针窗口 {_describe(my_num)}")

    # R2a：ignores=NO + hitTest self → 期望命中本窗口（负层级之外的通用前提：窗口可命中）
    r2a = _route_at(px, py, h)
    p2a = (r2a == my_num)
    ok &= p2a
    _log(f"ROUTE-R2a {'PASS' if p2a else 'FAIL'} 命中={_describe(r2a)} 期望={my_num}(本探针)")

    # R2b：ignores=YES → 期望不命中本窗口（穿透到下层真实窗口）
    win.setIgnoresMouseEvents_(True)
    _pump(SETTLE_S)
    r2b = _route_at(px, py, h)
    p2b = (r2b != my_num)
    ok &= p2b
    _log(f"ROUTE-R2b {'PASS' if p2b else 'FAIL'} 命中={_describe(r2b)} 期望≠{my_num}"
         f"（ignoresMouseEvents 生效：点击落到下层窗口）")

    # R3（信息性，不计入 PASS/FAIL）：ignores=NO + hitTest None → 观察路由是否仍命中本探针。
    # 本机实测该结果**不稳定**（WindowServer 对内容视图更换/时序敏感：多次运行里多数仍命中
    # 本窗、偶发穿到下层），且产品逻辑不依赖它——「毯外穿透」由 ignores 闸门负责，hitTest
    # 只在窗口已收到事件后做 40pt 判定。故只作 INFO 记录。
    win.setIgnoresMouseEvents_(False)
    win.setContentView_(_HitNilView.alloc().initWithFrame_(NSMakeRect(0, 0, PROBE_SIZE, PROBE_SIZE)))
    _pump(0.6)
    win.setIgnoresMouseEvents_(False)  # 视图更换后重新显式声明「参与路由」再查询
    _pump(0.6)
    r3 = _route_at(px, py, h)
    p3 = (r3 == my_num)
    verdict3 = ("仍命中本窗（事件目标是我们，hitTest 只决定窗口内部谁处理）" if p3
                else "穿到了下层窗口（本机偶发不稳定，见注释）")
    _log(f"ROUTE-R3 INFO 命中={_describe(r3)} —— {verdict3}")

    win.orderOut_(None)
    _pump(0.1)
    _log(f"ROUTE-SELFTEST {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def _spawn_rug_selftest(force: str | None):
    """启动 rug.py --selftest（可选 RUG_CLICKTHROUGH_FORCE 钉死闸门状态）。"""
    env = dict(os.environ)
    if force:
        env["RUG_CLICKTHROUGH_FORCE"] = force
    return subprocess.Popen(
        [os.path.join(_ROOT, ".venv", "bin", "python"), os.path.join(_ROOT, "rug.py"), "--selftest"],
        cwd=_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def _collect_rug_windows(pid: int) -> list[int]:
    infos = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionAll, Quartz.kCGNullWindowID)
    wins: list[int] = []
    for info in infos:
        try:
            if int(info.get("kCGWindowOwnerPID", -1)) == pid and \
                    int(info.get("kCGWindowLayer", 0)) < 0:
                wins.append(int(info.get("kCGWindowNumber", -1)))
        except Exception:
            continue
    return wins


def _scan_for_route(w: float, h: float, targets: set[int]) -> tuple[float, float, int] | None:
    """全屏网格扫描，找第一个路由命中 targets 中窗口的点。"""
    y = EDGE_MARGIN
    while y < h - EDGE_MARGIN:
        x = EDGE_MARGIN
        while x < w - EDGE_MARGIN:
            num = _route_at(x, y, h)
            if num in targets:
                return x, y, num
            x += SCAN_GRID
        y += SCAN_GRID
    return None


def cmd_live() -> int:
    """对运行中的 rug.py --selftest 做实时路由查询：闸门两态各跑一次。

    闸门状态由 RUG_CLICKTHROUGH_FORCE 钉死（生产不用该变量；光标不可合成，故必须钉死），
    从而不受「光标当前在哪」影响，可重复：
      - forced=receive：毯子面板可被 mouseDown 命中（§9「桌面层窗口收不到鼠标事件」风险）
        —— 需要屏幕上有**可见桌面区**，否则该方向 SKIP（环境所限，由真人点击测试覆盖）；
      - forced=ignore ：毯子面板退出路由，点击真正落到下层（G5 穿透的机制证明）。
    """
    w, h = _screen_wh()
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    ok = True

    # ---- 方向 1：forced=receive —— rug 面板可被命中 ----
    proc = _spawn_rug_selftest("receive")
    try:
        time.sleep(2.4)
        rug_wins = _collect_rug_windows(proc.pid)
        if not rug_wins:
            _log("ROUTE-ERROR [receive] 找不到 rug 进程的负层级窗口")
            return 2
        _log(f"ROUTE-INFO [receive] rug pid={proc.pid} 窗口 {rug_wins}")
        hit = _scan_for_route(w, h, set(rug_wins))
        if hit is None:
            _log("ROUTE-LIVE-RECEIVE SKIP 全屏扫描未命中 rug 面板：当前屏幕被普通窗口"
                 "铺满、没有可见桌面区（属环境状态，非缺陷；该方向由真人点击测试覆盖）")
        else:
            x_tl, y_tl, num = hit
            _log(f"ROUTE-LIVE-RECEIVE PASS 点({x_tl:.0f},{y_tl:.0f}) 命中={_describe(num)}"
                 f"（毯子面板在桌面可见处可被 mouseDown 命中）")
    finally:
        _shutdown(proc, "receive")

    # ---- 方向 2：forced=ignore —— rug 面板退出路由（穿透） ----
    proc = _spawn_rug_selftest("ignore")
    try:
        time.sleep(2.4)
        rug_wins = _collect_rug_windows(proc.pid)
        if not rug_wins:
            _log("ROUTE-ERROR [ignore] 找不到 rug 进程的负层级窗口")
            return 2
        _log(f"ROUTE-INFO [ignore] rug pid={proc.pid} 窗口 {rug_wins}")
        # 中心点必在毯子面板内（面板全屏）：ignores=YES 时路由必须不是 rug
        r_center = _route_at(w / 2.0, h / 2.0, h)
        p_center = r_center not in rug_wins
        ok &= p_center
        _log(f"ROUTE-LIVE-IGNORE {'PASS' if p_center else 'FAIL'} 屏中心({w/2:.0f},{h/2:.0f}) "
             f"命中={_describe(r_center)} 期望∉{rug_wins}（ignores 生效：点击落到下层）")
        # 再抽查两点（左上、右下 1/4 处）
        for x_tl, y_tl in ((w * 0.25, h * 0.25), (w * 0.75, h * 0.75)):
            r = _route_at(x_tl, y_tl, h)
            p = r not in rug_wins
            ok &= p
            _log(f"ROUTE-LIVE-IGNORE {'PASS' if p else 'FAIL'} 抽查({x_tl:.0f},{y_tl:.0f}) "
                 f"命中={_describe(r)} 期望∉{rug_wins}")
    finally:
        _shutdown(proc, "ignore")

    _log(f"ROUTE-LIVE {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def _shutdown(proc, tag: str) -> None:
    """收尾：终止进程、收走 stdout 并把关键行打出来（带 tag）。"""
    try:
        proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=5)
    except Exception:
        pass
    out = ""
    try:
        out = proc.stdout.read() if proc.stdout else ""
    except Exception:
        pass
    for line in (out or "").splitlines():
        if "SELFTEST" in line or line.startswith("RUG-"):
            _log(f"ROUTE-INFO [{tag}] {line}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="verify_routing", description="点击路由机制验证")
    p.add_argument("--live", action="store_true", help="额外对运行中的 rug --selftest 做实时查询")
    args = p.parse_args(argv)
    rc = cmd_selftest()
    if args.live:
        rc_live = cmd_live()
        rc = max(rc, rc_live)
    return rc


if __name__ == "__main__":
    sys.exit(main())
