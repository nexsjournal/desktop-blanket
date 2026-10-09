#!/usr/bin/env python3
"""fold_demo.py — v4「真翻折」离线验证 + 目验图（§12.16）。

逼真拖动：抓住毯子左上角附近 → 平滑横过布身到右下 → 甩手 → 落定 4 秒。
输出两样东西：
  1) 数值验证（FOLD-* 行）：抬手区粒子数 / **双层区对数**（xy 投影近、z 相差 ≥4pt、
     非网格近邻 = 毯子翻折压在自己身上）/ 高度剖面 / 最高点。2.5D 高度场做不到双层。
  2) 渲染快照 /tmp/rug_fold_demo.png（与 App 同一套材质：顶/背两张贴图 + 接触阴影）——
     翻折露底处应能看到变暗的背面贴图。

用法：.venv/bin/python scripts/fold_demo.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from AppKit import (NSApplication, NSApplicationActivationPolicyAccessory, NSScreen,
                    NSBitmapImageRep, NSPNGFileType)
from Foundation import NSDate, NSRunLoop

from cloth import ClothSim
from overlay import RugOverlay

DT = 1.0 / 30.0
OUT_PNG = "/tmp/rug_fold_demo.png"


def pump(seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.05))


def two_layer_pairs(pos: np.ndarray, cols: int, xy_tol: float = 6.0,
                    z_min: float = 4.0) -> int:
    """数「翻折双层」粒子对：xy 投影近、z 差大、且不是网格近邻（|Δr|/|Δc| > 2）。"""
    n = pos.shape[0]
    rr = np.arange(n) // cols
    cc = np.arange(n) % cols
    count = 0
    step = 600
    for i0 in range(0, n, step):
        i1 = min(i0 + step, n)
        dxy2 = ((pos[i0:i1, None, 0] - pos[None, :, 0]) ** 2
                + (pos[i0:i1, None, 1] - pos[None, :, 1]) ** 2)
        dz = pos[i0:i1, None, 2] - pos[None, :, 2]
        near_grid = (np.abs(rr[i0:i1, None] - rr[None, :]) <= 2) & \
                    (np.abs(cc[i0:i1, None] - cc[None, :]) <= 2)
        m = (dxy2 < xy_tol * xy_tol) & (np.abs(dz) > z_min) & ~near_grid
        count += int(m.sum())
    return count // 2


def flipped_tris(pos: np.ndarray, tri: np.ndarray) -> int:
    """几何环绕方向朝下（sim 系 z 分量 < 0）的三角数 = **真翻折**（露底、渲染用背贴图）。

    注意必须按几何环绕算，不能用渲染法线缓冲（渲染法线被主动翻正以保持受光观感）。
    """
    e1 = pos[tri[:, 1]] - pos[tri[:, 0]]
    e2 = pos[tri[:, 2]] - pos[tri[:, 0]]
    nz = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    return int((nz < -1e-6).sum())


def main() -> int:
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    screen = NSScreen.mainScreen()
    if screen is None:
        print("FOLD-ERROR 无可用屏幕")
        return 2
    f = screen.frame()
    W, H = float(f.size.width), float(f.size.height)

    rw, rh = W * 0.72, H * 0.80           # 与 App 相同的毯子尺寸
    sim = ClothSim(rw, rh)
    sim.set_placement((W / 2, H / 2), rw, rh, 0.0, zero_velocity=True)
    print(f"FOLD-DEMO 屏幕 {W:.0f}x{H:.0f} 毯子 {rw:.0f}x{rh:.0f}")

    # 抓左上角附近
    gx0, gy0 = W / 2 - rw / 2 + 0.10 * rw, H / 2 - rh / 2 + 0.12 * rh
    ok = sim.grab(gx0, gy0, 40.0)
    print(f"FOLD grab@({gx0:.0f},{gy0:.0f}) -> {ok}")
    if not ok:
        print("FOLD-ERROR 没抓住")
        return 2

    # 平滑横过布身（ease-in-out，50 帧 ≈ 1.7s 真人速度；行程超过毯身 → 材料必须折叠）
    tx, ty = W / 2 + 0.55 * rw, H / 2 + 0.34 * rh
    N = 50
    for i in range(1, N + 1):
        t = i / N
        tt = t * t * (3.0 - 2.0 * t)
        sim.drag_to(gx0 + (tx - gx0) * tt, gy0 + (ty - gy0) * tt)
        sim.step(DT)
    sim.release(fling=True)
    print("FOLD release（带抛掷）")
    for i in range(120):
        sim.step(DT)
        if sim.is_asleep():
            print(f"FOLD asleep@{i + 1} 步")
            break

    pos = sim._pos
    tri = sim.indices().reshape(-1, 3).astype(np.int64)
    flipped = flipped_tris(pos, tri)
    lifted = int(np.count_nonzero(pos[:, 2] > 1.5))
    layers = two_layer_pairs(pos, sim.cols)
    zmax = float(pos[:, 2].max())

    # 层对（i 在上层、j 在下方）供出图/像素核对用
    n = pos.shape[0]
    rr = np.arange(n) // sim.cols
    cc = np.arange(n) % sim.cols
    upper = np.zeros(n, dtype=bool)
    for i0 in range(0, n, 600):
        i1 = min(i0 + 600, n)
        dxy2 = ((pos[i0:i1, None, 0] - pos[None, :, 0]) ** 2
                + (pos[i0:i1, None, 1] - pos[None, :, 1]) ** 2)
        dz = pos[i0:i1, None, 2] - pos[None, :, 2]
        near_grid = (np.abs(rr[i0:i1, None] - rr[None, :]) <= 2) & \
                    (np.abs(cc[i0:i1, None] - cc[None, :]) <= 2)
        upper[i0:i1] |= ((dxy2 < 36.0) & (dz > 4.0) & ~near_grid).any(axis=1)

    print(f"FOLD-RESULT flipped_tris={flipped}（翻底露出的三角数）  lifted(>1.5pt)={lifted}  "
          f"two_layer_pairs={layers}  zmax={zmax:.1f}pt  last_move={sim._last_move:.3f}")
    if upper.any():
        fx = float(pos[upper, 0].mean())
        fy = float(pos[upper, 1].mean())
        print(f"FOLD 折层区域中心 ≈({fx:.0f},{fy:.0f})")

    # 高度图（7×7 粗采样毯身覆盖区，直观看到折脊在哪）
    cx, cy = float(pos[:, 0].mean()), float(pos[:, 1].mean())
    grid = []
    for r in range(7):
        row = []
        for c in range(7):
            px = cx + (c / 6.0 - 0.5) * rw * 0.9
            py = cy + (r / 6.0 - 0.5) * rh * 0.9
            d2 = (pos[:, 0] - px) ** 2 + (pos[:, 1] - py) ** 2
            m = d2 < 40.0 * 40.0
            row.append(f"{float(pos[m, 2].max()):3.0f}" if m.any() else "  .")
        grid.append(" ".join(row))
    print("FOLD-HEIGHTMAP (7x7, pt):")
    for row in grid:
        print("   ", row)

    verdict = "PASS（出现真翻折：大面积翻底保持）" if flipped >= 100 else \
        ("WEAK（有折层但翻底很少）" if (layers > 20 or lifted > 50) else "FAIL（没折起来）")
    print(f"FOLD-VERDICT {verdict}")

    # 出图前把毯子刚体平移回屏幕中央（演示取景；不影响折层几何与物理）
    shift = (np.array([W / 2, H / 2]) - pos[:, :2].mean(axis=0)).astype(np.float64)
    pos[:, 0] += shift[0]
    pos[:, 1] += shift[1]
    for _ in range(20):
        sim.step(DT)
    pump(0.05)

    # ---- 渲染快照（与 App 同套材质）----
    tex = os.path.join(ROOT, "assets", "rugs", "rug-01.png")
    ov = RugOverlay(screen, tex, lambda w, h: sim, fps=1)
    try:
        ov.show()
        pump(0.4)
        ov._rebuild_geometry()
        img = ov._scn.snapshot()
        if img is not None:
            rep = NSBitmapImageRep.imageRepWithData_(img.TIFFRepresentation())
            Wp, Hp = int(rep.pixelsWide()), int(rep.pixelsHigh())
            data = rep.representationUsingType_properties_(NSPNGFileType, {})
            saved = bool(data.writeToFile_atomically_(OUT_PNG, True))
            print(f"FOLD-RENDER 快照 {Wp}x{Hp} 已存={saved} -> {OUT_PNG}")

            # 折层区域应看到「背面贴图」（变暗）：按**几何环绕**找翻底三角（scene 系
            # 绕行法线朝上 = 从相机看是正面朝向 = 渲染时用背面材质）。
            vb = ov._vbuf
            t = ov._tri
            e1 = vb[t[:, 1]] - vb[t[:, 0]]
            e2 = vb[t[:, 2]] - vb[t[:, 0]]
            n_y = e1[:, 2] * e2[:, 0] - e1[:, 0] * e2[:, 2]     # 环绕法线的 scene-y 分量
            back_tris = np.nonzero(n_y > 0.0)[0]
            front_pts_idx = np.nonzero((pos[:, 2] < 1.0))[0][::50]
            scale = float(screen.backingScaleFactor())

            def lum(pts) -> float:
                vals = []
                for px, py in pts:
                    c = rep.colorAtX_y_(int(min(max(px * scale, 0), Wp - 1)),
                                        int(min(max(py * scale, 0), Hp - 1)))
                    if c is not None:
                        vals.append(0.299 * c.redComponent() + 0.587 * c.greenComponent()
                                    + 0.114 * c.blueComponent())
                return float(np.mean(vals)) if vals else float("nan")

            back_idx = np.unique(t[back_tris].ravel())
            up_pts = [(float(pos[i, 0]), float(pos[i, 1])) for i in back_idx[::3]]
            flat_pts = [(float(pos[i, 0]), float(pos[i, 1])) for i in front_pts_idx]
            l_up, l_flat = lum(up_pts), lum(flat_pts)
            ratio = (l_up / l_flat) if l_flat and l_flat == l_flat and l_flat > 1e-6 else float("nan")
            print(f"FOLD-RENDER 翻底三角数={back_tris.size}（0 = 画面上没有露底的面）  "
                  f"露底亮度 {l_up:.3f} vs 平铺亮度 {l_flat:.3f} → 比值 {ratio:.2f}"
                  f"（露底区域应 <0.85：看到的是变暗的背面贴图）")
        else:
            print("FOLD-RENDER 快照失败")
    finally:
        ov.hide()
        pump(0.2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
