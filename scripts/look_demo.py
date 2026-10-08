#!/usr/bin/env python3
"""look_demo.py — v5 观感评测台（离线四场景渲染 + 数值指标）。

用途：改 cloth/overlay 的观感后，用**同一套 App 材质**离线渲出四个真机同款场景，
对每个场景给出「像不像参考视频」的数值判据，并出图供人工/评审对照。

四场景：
    rest    铺上后静置 3s（参考视频：毯子应平铺、无白边、无破洞）
    drag    抓住毯身横拖（参考视频：大而光滑的波浪，不碎皱）
    fold    抓一角横过布身 + 抛掷（应有**圆角、有体积**的翻折，不是平面剪纸）
    corner  抓右下角拖向中央（参考视频截图 2 的折角：折片压在毯身上、折脊有高度）

每个场景输出 `LOOK-*` 指标行与 `/tmp/look_<场景>.png`：
    coverage   画出来的不透明像素 / 毯子标称面积（平铺应 ≈ ≥0.9）
    holes      毯子轮廓**内部**的透明像素占比（应 <0.5%；剪纸式折痕会大量出现）
    white_edge 轮廓边缘 6px 带内的近白像素占比（白边贴图的直接判据，应 <2%）
    zmax/lifted/flipped/two_layer  几何量（翻折是否有体积）
    lum_ratio  画面平铺区亮度 / 贴图亮度（灯光是否把毯子压暗）

用法：
    .venv/bin/python scripts/look_demo.py                 # 四个场景
    .venv/bin/python scripts/look_demo.py rest fold       # 只跑指定场景
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from AppKit import NSApplication, NSApplicationActivationPolicyAccessory, NSScreen
from Foundation import NSDate, NSRunLoop
from Quartz import (
    CGWindowListCreateImage, CGRectNull, kCGWindowListOptionIncludingWindow,
    kCGWindowImageBoundsIgnoreFraming, CGImageGetWidth, CGImageGetHeight,
    CGImageGetBytesPerRow, CGImageGetBitsPerPixel, CGImageGetDataProvider,
    CGDataProviderCopyData)

from cloth import ClothSim
from overlay import RugOverlay, _detect_texture_margin

DT = 1.0 / 30.0
TEX = os.path.join(ROOT, "assets", "rugs", "rug-01.png")


def pump(seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.05))


def surface_of_window(wnum: int):
    img = CGWindowListCreateImage(CGRectNull, kCGWindowListOptionIncludingWindow, wnum,
                                  kCGWindowImageBoundsIgnoreFraming)
    if img is None:
        return None
    w, h = int(CGImageGetWidth(img)), int(CGImageGetHeight(img))
    bpr, bpp = int(CGImageGetBytesPerRow(img)), int(CGImageGetBitsPerPixel(img))
    buf = bytes(CGDataProviderCopyData(CGImageGetDataProvider(img)))
    if bpp != 32 or len(buf) < bpr * h:
        return None
    arr = np.frombuffer(buf, dtype=np.uint8)[: bpr * h].reshape(h, bpr // 4, 4)[:, :w, :]
    return arr.copy()


# ---------------------------------------------------------------- 场景构造

def scenario_rest(sim: ClothSim, w: float, h: float) -> str:
    sim.set_placement((w / 2, h / 2), sim.width_pt, sim.height_pt, 0.0, zero_velocity=True)
    for _ in range(90):
        sim.step(DT)
    return "rest"


def scenario_drag(sim: ClothSim, w: float, h: float) -> str:
    sim.set_placement((w / 2, h / 2), sim.width_pt, sim.height_pt, 0.0, zero_velocity=True)
    for _ in range(30):
        sim.step(DT)
    gx, gy = w / 2, h / 2
    sim.grab(gx, gy, 40.0)
    tx, ty = w / 2 - 0.22 * w, h / 2 + 0.10 * h
    n = 30
    for i in range(1, n + 1):
        t = i / n
        tt = t * t * (3.0 - 2.0 * t)
        sim.drag_to(gx + (tx - gx) * tt, gy + (ty - gy) * tt)
        sim.step(DT)
    sim.release(fling=False)
    for _ in range(60):
        sim.step(DT)
    return "drag"


def scenario_fold(sim: ClothSim, w: float, h: float) -> str:
    rw, rh = sim.width_pt, sim.height_pt
    sim.set_placement((w / 2, h / 2), rw, rh, 0.0, zero_velocity=True)
    gx0, gy0 = w / 2 - rw / 2 + 0.10 * rw, h / 2 - rh / 2 + 0.12 * rh
    assert sim.grab(gx0, gy0, 40.0)
    tx, ty = w / 2 + 0.55 * rw, h / 2 + 0.34 * rh
    n = 50
    for i in range(1, n + 1):
        t = i / n
        tt = t * t * (3.0 - 2.0 * t)
        sim.drag_to(gx0 + (tx - gx0) * tt, gy0 + (ty - gy0) * tt)
        sim.step(DT)
    sim.release(fling=False)
    for _ in range(150):
        sim.step(DT)
        if sim.is_asleep():
            break
    return "fold"


def scenario_corner(sim: ClothSim, w: float, h: float) -> str:
    """抓右下角拖向毯心偏左（参考截图 2 的折角手势）。"""
    rw, rh = sim.width_pt, sim.height_pt
    sim.set_placement((w / 2, h / 2), rw, rh, 0.0, zero_velocity=True)
    for _ in range(30):
        sim.step(DT)
    gx0, gy0 = w / 2 + 0.46 * rw, h / 2 + 0.44 * rh
    assert sim.grab(gx0, gy0, 40.0)
    tx, ty = w / 2 - 0.10 * rw, h / 2 - 0.05 * rh
    n = 44
    for i in range(1, n + 1):
        t = i / n
        tt = t * t * (3.0 - 2.0 * t)
        sim.drag_to(gx0 + (tx - gx0) * tt, gy0 + (ty - gy0) * tt)
        sim.step(DT)
    sim.release(fling=False)
    for _ in range(150):
        sim.step(DT)
        if sim.is_asleep():
            break
    return "corner"


SCENARIOS = {
    "rest": scenario_rest,
    "drag": scenario_drag,
    "fold": scenario_fold,
    "corner": scenario_corner,
}


# ---------------------------------------------------------------- 指标

def geom_stats(sim: ClothSim) -> dict:
    pos = np.asarray(sim._pos)
    tri = np.asarray(sim.indices()).reshape(-1, 3).astype(np.int64)
    e1 = pos[tri[:, 1]] - pos[tri[:, 0]]
    e2 = pos[tri[:, 2]] - pos[tri[:, 0]]
    nz = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    flipped = int((nz < -1e-6).sum())
    lifted = int((pos[:, 2] > 3.0).sum())
    # 双层对（xy 近、z 差大、非网格近邻）= 折片压在自己身上
    n = pos.shape[0]
    rr = np.arange(n) // sim.cols
    cc = np.arange(n) % sim.cols
    two = 0
    for i0 in range(0, n, 800):
        i1 = min(i0 + 800, n)
        dxy2 = ((pos[i0:i1, None, 0] - pos[None, :, 0]) ** 2
                + (pos[i0:i1, None, 1] - pos[None, :, 1]) ** 2)
        dz = pos[i0:i1, None, 2] - pos[None, :, 2]
        near_grid = (np.abs(rr[i0:i1, None] - rr[None, :]) <= 2) & \
                    (np.abs(cc[i0:i1, None] - cc[None, :]) <= 2)
        two += int(((dxy2 < 36.0) & (np.abs(dz) > 4.0) & ~near_grid).sum())
    return {
        "zmax": float(pos[:, 2].max()),
        "z_p99": float(np.percentile(pos[:, 2], 99.0)),
        "lifted": lifted,
        "flipped": flipped,
        "two_layer": two // 2,
    }


def render_stats(rgba: np.ndarray, sim: ClothSim, scale: float, tex_mean_lum: float) -> dict:
    a = rgba[..., 3]
    rgb = rgba[..., :3].astype(np.int16)
    op = a > 128
    n = op.size
    nominal = (sim.width_pt * sim.height_pt) * (scale ** 2)   # 标称毯身像素数
    holes = 0.0
    white_edge = 0.0
    if op.any():
        im = Image.fromarray((op * 255).astype(np.uint8), "L")
        # 内部空洞：从四角 flood fill 背景，仍为 0 的透明像素 = 被毯子包住
        rc = im.copy()
        for xy in ((0, 0), (im.width - 1, 0), (0, im.height - 1), (im.width - 1, im.height - 1)):
            if rc.getpixel(xy) == 0:
                ImageDraw.floodfill(rc, xy, 128)
        filled = np.asarray(rc) == 128
        holes = float(((~op) & ~filled).sum()) / float(nominal)
        # 轮廓边缘带（6px）：膨胀 - 腐蚀
        dil = np.asarray(im.filter(ImageFilter.MaxFilter(13))) > 128
        ero = np.asarray(im.filter(ImageFilter.MinFilter(13))) > 128
        band = dil & ~ero
        near_white = (rgb.min(axis=-1) > 225) & op
        white_edge = float((band & near_white).sum()) / max(float(band.sum()), 1.0)
    flat_lum = float("nan")
    flat = op & (rgb.min(axis=-1) > 0)
    if flat.any():
        lum = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])
        flat_lum = float(lum[flat].mean()) / 255.0
    return {
        "coverage": float(op.sum()) / nominal,
        "holes": holes,
        "white_edge": white_edge,
        "lum_ratio": (flat_lum / tex_mean_lum) if tex_mean_lum > 1e-6 else float("nan"),
    }


def main() -> int:
    which = sys.argv[1:] or list(SCENARIOS)
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    screen = NSScreen.mainScreen()
    f = screen.frame()
    W, H = float(f.size.width), float(f.size.height)

    tex = Image.open(TEX).convert("RGB")
    tarr = np.asarray(tex).astype(np.float64)
    # 亮度基准取**裁掉留白后**的贴图：加载期会自动裁掉白边，用含白边的均值会低估基准
    from Foundation import NSData
    from AppKit import NSBitmapImageRep
    rep = NSBitmapImageRep.alloc().initWithData_(NSData.dataWithContentsOfFile_(TEX))
    t, l, b, r = _detect_texture_margin(rep)
    if t or l or b or r:
        tarr = tarr[t:tarr.shape[0] - b, l:tarr.shape[1] - r]
    tex_lum = float((0.299 * tarr[..., 0] + 0.587 * tarr[..., 1] + 0.114 * tarr[..., 2]).mean()) / 255.0

    print(f"LOOK 屏幕 {W:.0f}x{H:.0f}  贴图亮度 {tex_lum:.3f}")
    for name in which:
        fn = SCENARIOS[name]
        rw, rh = W * 0.626, H * 0.62        # 与 settings.json 当前一致
        sim = ClothSim(rw, rh)
        fn(sim, W, H)
        # 取景：把毯子刚体平移回屏幕中央（不影响折层几何与物理，只为出图可比）
        c = sim._pos[:, :2].mean(axis=0)
        sim._pos[:, 0] += W / 2 - c[0]
        sim._pos[:, 1] += H / 2 - c[1]
        sim.wake()
        for _ in range(10):
            sim.step(DT)
        g = geom_stats(sim)
        ov = RugOverlay(screen, TEX, lambda w, h: sim, fps=1)
        try:
            ov.show()
            pump(0.35)
            ov._rebuild_geometry()
            pump(0.15)
            wnum = ov._panel.windowNumber()
            rgba = surface_of_window(int(wnum))
            if rgba is None:
                print(f"LOOK-{name} 抓窗口表面失败")
                continue
            r = render_stats(rgba, sim, float(screen.backingScaleFactor()), tex_lum)
            Image.fromarray(rgba[..., [2, 1, 0, 3]], "RGBA").save(f"/tmp/look_{name}.png")
        finally:
            ov.hide()
            pump(0.15)
        print(f"LOOK-{name:6s} coverage={r['coverage']:.3f} holes={r['holes']*100:.2f}% "
              f"white_edge={r['white_edge']*100:.2f}% lum_ratio={r['lum_ratio']:.2f} | "
              f"zmax={g['zmax']:.0f} z99={g['z_p99']:.0f} lifted={g['lifted']} "
              f"flipped={g['flipped']} two_layer={g['two_layer']} asleep={sim.is_asleep()} "
              f"-> /tmp/look_{name}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
