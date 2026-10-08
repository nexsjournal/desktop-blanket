#!/usr/bin/env python3
"""运行中实例的线上体检（§12.11 首次使用，2026-10-08）。

回答三个具体问题：
  1. 毯子渲染出来了吗？（抓「我们自己的窗口表面」像素，不受遮挡影响——§12.10 的验证手法）
  2. 毯子现在看得见吗？（上层窗口对毯子墨迹区域的遮挡覆盖率；被普通窗口盖住是设计行为，不是 bug）
  3. 某个点上的点击会打到谁？（+[NSWindow windowNumberAtPoint:belowWindowWithWindowNumber:] 权威路由查询）

用法：
    .venv/bin/python scripts/live_probe.py            # 自动找运行中的 rug.py 实例
    .venv/bin/python scripts/live_probe.py --pid 1234 # 指定进程
    .venv/bin/python scripts/live_probe.py --save     # 把窗口表面抓图存到 /tmp/rug_window_surface.png
    .venv/bin/python scripts/live_probe.py --icons    # 附带：桌面图标位置 vs 毯子覆盖（需要「自动化→Finder」）

判读：
    LIVE-PROBE-RENDER OK       窗口表面不透明占比 ≥15%——布面画出来了
    LIVE-PROBE-RENDER SUSPECT  几乎全透明——渲染回归，对照 §12.10 排查（材质/几何）
    LIVE-PROBE-OCCLUSION DONE  遮挡统计；>99% = 此刻屏幕上完全看不到毯子（挪开窗口/显示桌面）
"""
import argparse
import subprocess
import sys

import numpy as np
from AppKit import (NSApplication, NSBitmapImageRep, NSPNGFileType, NSWindow,
                    NSMakePoint, NSScreen)
from Quartz import (
    CGWindowListCopyWindowInfo, kCGWindowListOptionAll, kCGWindowListOptionOnScreenOnly,
    kCGWindowListOptionIncludingWindow, CGWindowListCreateImage, CGRectNull,
    kCGWindowImageBoundsIgnoreFraming, kCGWindowImageDefault, kCGNullWindowID,
    CGImageGetWidth, CGImageGetHeight, CGImageGetBytesPerRow, CGImageGetBitsPerPixel,
    CGImageGetDataProvider, CGDataProviderCopyData)


def _log(tag: str, msg: str) -> None:
    print(f"{tag} {msg}")


def find_rug_pids() -> list[int]:
    out = subprocess.run(["pgrep", "-fl", "rug.py"], capture_output=True, text=True).stdout
    pids = []
    for line in out.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and "rug.py" in parts[1] and "live_probe" not in parts[1]:
            try:
                pids.append(int(parts[0]))
            except ValueError:
                pass
    return pids


def windows_of(pid: int):
    infos = CGWindowListCopyWindowInfo(kCGWindowListOptionAll, 0)
    return [w for w in infos if w.get("kCGWindowOwnerPID") == pid]


def grab_surface(wnum: int):
    img = CGWindowListCreateImage(CGRectNull, kCGWindowListOptionIncludingWindow, wnum,
                                  kCGWindowImageBoundsIgnoreFraming)
    if img is None:
        return None, None
    w, h = int(CGImageGetWidth(img)), int(CGImageGetHeight(img))
    bpr, bpp = int(CGImageGetBytesPerRow(img)), int(CGImageGetBitsPerPixel(img))
    data = CGDataProviderCopyData(CGImageGetDataProvider(img))
    buf = bytes(data)
    if bpp != 32 or len(buf) < bpr * h:
        return None, img
    arr = np.frombuffer(buf, dtype=np.uint8)[: bpr * h].reshape(h, bpr // 4, 4)[:, :w, :]
    return arr, img


def save_png(img, path: str) -> None:
    rep = NSBitmapImageRep.alloc().initWithCGImage_(img)
    data = rep.representationUsingType_properties_(NSPNGFileType, {})
    _log("LIVE-PROBE-SAVE", f"{path}: {bool(data.writeToFile_atomically_(path, True))}")


def main() -> int:
    ap = argparse.ArgumentParser(description="运行中 rug.py 实例的线上体检")
    ap.add_argument("--pid", type=int, default=None)
    ap.add_argument("--save", action="store_true", help="窗口表面抓图存 /tmp/rug_window_surface.png")
    ap.add_argument("--icons", action="store_true", help="附带桌面图标位置 vs 毯子覆盖（需 Finder 权限）")
    args = ap.parse_args()

    NSApplication.sharedApplication()  # windowNumberAtPoint 需要 app 上下文，否则恒返回 0
    screen_h = float(NSScreen.mainScreen().frame().size.height)

    pids = [args.pid] if args.pid else find_rug_pids()
    if not pids:
        _log("LIVE-PROBE-ERROR", "没找到运行中的 rug.py（先 ./run.sh 铺上毯子）")
        return 2

    pid = None
    panel = None
    for p in pids:
        ws = windows_of(p)
        if ws:
            panel = max(ws, key=lambda w: (w["kCGWindowBounds"]["Width"] * w["kCGWindowBounds"]["Height"]))
            pid = p
            break
    if panel is None:
        _log("LIVE-PROBE-ERROR", f"找到进程 {pids} 但没有窗口（毯子未铺上？）")
        return 2

    wnum = int(panel["kCGWindowNumber"])
    lay = int(panel["kCGWindowLayer"])
    b = panel["kCGWindowBounds"]
    start = subprocess.run(["ps", "-p", str(pid), "-o", "lstart="],
                           capture_output=True, text=True).stdout.strip()
    _log("LIVE-PROBE-WINDOW",
         f"pid={pid} win#{wnum} level={lay} onscreen={bool(panel.get('kCGWindowIsOnscreen'))} "
         f"bounds={b['Width']:.0f}x{b['Height']:.0f}@{b['X']:.0f},{b['Y']:.0f} 启动={start}")

    arr, img = grab_surface(wnum)
    bbox = None
    if arr is None:
        _log("LIVE-PROBE-RENDER", "SUSPECT 拿不到窗口表面（权限？）")
    else:
        h, w, _ = arr.shape
        ys, xs = np.where(arr[:, :, 3] > 8)
        frac = len(ys) / (w * h) * 100
        tag = "OK" if frac >= 15 else ("SUSPECT" if frac <= 5 else "LOW")
        _log("LIVE-PROBE-RENDER", f"{tag} 窗口表面 {w}x{h}px 不透明占比 {frac:.1f}%")
        if len(ys):
            bbox = (xs.min() / 2.0, xs.max() / 2.0, ys.min() / 2.0, ys.max() / 2.0)
            strong = arr[ys, xs][arr[ys, xs][:, 3] > 128]
            if len(strong):
                bb, gg, rr = (float(strong[:, i].mean()) for i in (0, 1, 2))
                _log("LIVE-PROBE-RENDER",
                     f"毯子墨迹 bbox(pt) x[{bbox[0]:.0f},{bbox[1]:.0f}] y[{bbox[2]:.0f},{bbox[3]:.0f}] "
                     f"均色 R={rr:.0f} G={gg:.0f} B={bb:.0f}（BGRA 内存序）")
        if frac <= 5:
            _log("LIVE-PROBE-RENDER", "对照 §12.10：几何未挂材质时整窗 0% 不透明；像素近零即同类回归")
        if args.save and img is not None:
            save_png(img, "/tmp/rug_window_surface.png")

    # ---- 遮挡 ----
    infos = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, 0)
    above = []
    for w in infos:
        wl = w.get("kCGWindowLayer")
        if wl is None or wl <= lay or w.get("kCGWindowOwnerPID") == pid:
            continue
        wb = w.get("kCGWindowBounds")
        above.append((wl, str(w.get("kCGWindowOwnerName")), str(w.get("kCGWindowName") or ""),
                      wb["X"], wb["Y"], wb["Width"], wb["Height"], int(w["kCGWindowNumber"])))
    if bbox is not None:
        gx, gy = np.meshgrid(np.linspace(bbox[0], bbox[1], 160), np.linspace(bbox[2], bbox[3], 160))
        cov = np.zeros(gx.shape, dtype=bool)
        for _, _, _, X, Y, W, H, _n in above:
            cov |= (gx >= X) & (gx <= X + W) & (gy >= Y) & (gy <= Y + H)
        pct = cov.mean() * 100
        sup = ">99% = 此刻屏幕上看不到毯子（被普通窗口盖住是设计；挪开窗口/显示桌面）" if pct > 99 else ""
        _log("LIVE-PROBE-OCCLUSION", f"DONE 毯子墨迹被上层窗口覆盖 {pct:.1f}% {sup}")
        for row in above:
            _log("LIVE-PROBE-OCCLUSION", f"  上层: layer={row[0]} {row[1]!r} {row[2]!r} "
                                         f"{row[5]:.0f}x{row[6]:.0f}@{row[3]:.0f},{row[4]:.0f}")

        # ---- 路由 ----
        cx, cy = (bbox[0] + bbox[1]) / 2, (bbox[2] + bbox[3]) / 2
        wmap = {int(w.get("kCGWindowNumber")): (str(w.get("kCGWindowOwnerName")),
                                                str(w.get("kCGWindowName") or "")) for w in infos}
        wn_hit = int(NSWindow.windowNumberAtPoint_belowWindowWithWindowNumber_(
            NSMakePoint(cx, screen_h - cy), 0))
        owner, name = wmap.get(wn_hit, ("?", "?"))
        verdict = "点击打在毯子上" if wn_hit == wnum else f"点击被 {owner!r} {name!r} 截获（层级设计如此）"
        _log("LIVE-PROBE-ROUTE", f"毯子中心 ({cx:.0f},{cy:.0f}) -> win#{wn_hit} == {verdict}")

    # ---- 可选：图标 vs 毯子 ----
    if args.icons:
        try:
            sys.path.insert(0, ".")
            from icons import IconSensor
            items = IconSensor().query()
            if bbox is not None and items:
                inside = [it.name for it in items
                          if bbox[0] <= it.x_pt <= bbox[1] and bbox[2] - 90 <= it.y_pt <= bbox[3]]
                _log("LIVE-PROBE-ICONS",
                     f"{len(items)} 个图标，其中 {len(inside)} 个在毯子覆盖范围内"
                     f"（隆包只在毯子盖到图标时可见）：{inside}")
        except Exception as exc:  # 权限拒绝等一律降级为提示
            _log("LIVE-PROBE-ICONS", f"跳过（{exc}）")

    return 0


if __name__ == "__main__":
    sys.exit(main())
