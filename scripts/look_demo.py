#!/usr/bin/env python3
"""look_demo.py — 观感评测台（离线四场景渲染 + 数值指标；v5 建，v6 加「厚度」组）。

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
    settled    冻结前物理是否收敛（入睡）；冻结后 asleep 恒 False（show 会 wake）

    —— v6「厚度」行（§12.18）：逐特征 A/B（关掉特征 → 重抓窗口表面 → 取差）——
    rim_dlum     滚边贡献：轮廓最外 0..14px（厚度带）平均 |Δ亮度|
    rim_lit/dim  滚边带在迎光侧 / 背光侧的平均亮度（应 lit > dim：滚边随光转向）
    relief_dlum  静止微浮雕贡献：毯身核心平均 |Δ亮度|（设计 ~2.5-3 亮阶，四场景一致）
    ao_dlum(%)   折层遮蔽贡献：受影响像素（|Δ|>2）平均 |Δ亮度| 与面积占比（静置 0）
    shadow_dalp  接触阴影贡献：轮廓外侧 1..24px 平均 |Δalpha|
    halo_min     布与影子之间不得出现亮缝（轮廓外 1..5px 的最小 alpha；折层压布处可为 0）
    edge_px      轮廓过渡宽度（px；纸片边 1~2，带滚边+贴边阴影 20+）
    outline_grow 轮廓外扩（px/边；≈滚边半径×2 = 厚度几何在）

⚠️ 抓图前会先 settle 再**冻结 `ov._on_tick`**：RugOverlay 把 fps clamp 到 ≥30，
不冻结的话 A/B 之间位姿会漂移、指标全是假象（§12.18.3 实测踩过）。

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


# ---------------------------------------------------------------- v6 厚度指标（§12.18）
# 边缘亮度被贴图花纹掩盖（实测：毯子边缘的图案本身就有明暗差），因此厚度特征一律用
# **A/B 关掉特征再抓一次**来量化：贡献 = 两次渲染的像素差。这同时是「特征真的在画」的锚点。

def _lum_alpha(rgba: np.ndarray):
    x = rgba.astype(np.float64)
    return (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]), x[..., 3]


def _erode(m: np.ndarray, it: int) -> np.ndarray:
    for _ in range(it):
        m = m & np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1)
    return m


def _dilate(m: np.ndarray, it: int) -> np.ndarray:
    for _ in range(it):
        m = m | np.roll(m, 1, 0) | np.roll(m, -1, 0) | np.roll(m, 1, 1) | np.roll(m, -1, 1)
    return m


def _edge_transition_px(al: np.ndarray, op: np.ndarray) -> float:
    """轮廓不透明过渡宽度（px，中位数）：alpha 由 <40 升到 >250 的横向跨度。

    注意：外侧阴影（黑+alpha）也在这个跨度里，所以它同时反映「边缘带 + 贴边阴影」的总宽度；
    纸片边（无滚边无阴影）实测 1~3px。
    """
    ys, _ = np.nonzero(op)
    if ys.size == 0:
        return -1.0
    out = []
    for frac in (0.2, 0.35, 0.5, 0.65, 0.8):
        y = int(ys.min() + frac * (ys.max() - ys.min()))
        row = al[y]
        idx = np.nonzero(row > 250)[0]
        if idx.size == 0:
            continue
        s = int(idx.min())
        j = s
        while j > 0 and row[j] > 40.0:
            j -= 1
        out.append(s - j)
    return float(np.median(out)) if out else -1.0


def thickness_stats(ov, sim: ClothSim, base: np.ndarray, scale: float) -> dict:
    """逐特征 A/B（关掉 → 重抓窗口表面 → 取差）量化 v6 厚度的可见贡献。

    调用前提：**物理已冻结且收敛**（main 里先 settle 再冻结 tick）——否则 A/B 之间位姿
    漂移会伪造出巨大差（实测拖动场景 relief 贡献被抬到 22、AO 到 26 全是假象）。

    指标（都在基图自家掩膜上取；A/B 两次同机位同材质）：
      rim_dlum     轮廓最外 0..14px 带内平均 |Δ亮度|（滚边被画出来的直接证据）
      rim_lit/dim  滚边带在「迎光半圈」与「背光半圈」的平均亮度（应 lit > dim：滚边随光转向）
      relief_dlum  毯身核心（离轮廓 ≥60px）平均 |Δ亮度|（静止微浮雕贡献，应 ~2-4）
      ao_dlum      关掉折层遮蔽场后 **受影响像素**（|Δ|>2）的平均 |Δ亮度|；ao_area=占比%
      shadow_dalp  轮廓外侧 1..24px 带内平均 |Δalpha|（接触阴影贡献）
      edge_px      轮廓过渡宽度（px）；outline_grow 轮廓外扩（px/边，标称尺寸对比）
    """
    wnum = int(ov._panel.windowNumber())
    lum_b, al_b = _lum_alpha(base)
    op = al_b > 250
    if not op.any():
        return {}
    core = _erode(op, 60)
    band = op & ~_erode(op, 14)              # 滚边所在的外圈（宽 = R×scale ≈14px）
    outer = _dilate(op, 24) & ~_dilate(op, 1)

    def shot() -> np.ndarray:
        ov._rebuild_geometry()
        pump(0.15)
        return surface_of_window(wnum)

    def affected(lum_a: np.ndarray, m: np.ndarray) -> tuple[float, float]:
        d = np.abs(lum_b - lum_a)[m]
        hot = d > 2.0
        return (float(d[hot].mean()) if hot.any() else 0.0,
                float(hot.mean()) * 100.0 if d.size else 0.0)

    out = {}
    if getattr(ov, "_rim_node", None) is not None:
        ov._rim_node.setHidden_(True)
        alt = shot()
        ov._rim_node.setHidden_(False)
        shot()
        lum_a, _ = _lum_alpha(alt)
        out["rim_dlum"] = float(np.abs(lum_b - lum_a)[band].mean())
        # 迎光/背光两侧（1:1 划分）：光在屏幕左上 → 上边 + 左边 = 迎光，下边 + 右边 = 背光。
        # 角上归属两侧都算 → 用互斥条件排除（只看纯边中段），避免纹理在角上干扰判据。
        op_ys, op_xs = np.nonzero(op)
        y_top, y_bot = op_ys.min(), op_ys.max()
        x_lef, x_rig = op_xs.min(), op_xs.max()
        ys, xs = np.nonzero(band)
        if ys.size:
            px = 14.0
            near_top = ys < y_top + px
            near_bot = ys > y_bot - px
            near_lef = xs < x_lef + px
            near_rig = xs > x_rig - px
            lit = (near_top | near_lef) & ~(near_bot | near_rig)
            dim = (near_bot | near_rig) & ~(near_top | near_lef)
            lb = lum_b[band]
            out["rim_lit"] = float(lb[lit].mean()) if lit.any() else float("nan")
            out["rim_dim"] = float(lb[dim].mean()) if dim.any() else float("nan")
    if getattr(ov, "_relief", None) is not None:
        keep = ov._relief.copy()
        ov._relief[:] = 0.0
        alt = shot()
        ov._relief[:] = keep
        shot()
        lum_a, _ = _lum_alpha(alt)
        out["relief_dlum"] = float(np.abs(lum_b - lum_a)[core].mean())
    try:
        keep_ao = np.array(sim._ao, copy=True)
        sim._ao[:] = 0.0
        alt = shot()
        sim._ao[:] = keep_ao
        shot()
        lum_a, _ = _lum_alpha(alt)
        out["ao_dlum"], out["ao_area"] = affected(lum_a, core)
    except Exception:
        out["ao_dlum"] = out["ao_area"] = float("nan")
    ov.set_shadow_visible(False)
    alt = shot()
    ov.set_shadow_visible(True)
    shot()
    _, al_a = _lum_alpha(alt)
    out["shadow_dalp"] = float(np.abs(al_b - al_a)[outer].mean())
    # 贴边亮缝守卫：轮廓外 1..5px 的 alpha 不能掉到很低（布与接触阴影之间不允许出现亮缝）
    halo = _dilate(op, 5) & ~_dilate(op, 1)
    out["halo_min"] = float(al_b[halo].min()) if halo.any() else float("nan")
    out["edge_px"] = _edge_transition_px(al_b, op)
    ys, xs = np.nonzero(op)
    out["outline_grow"] = (float(xs.max() - xs.min() + 1) - sim.width_pt * scale) * 0.5
    return out


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
        # 先让物理收敛再冻结：A/B 抓图必须来自同一（静止）位姿，否则位姿漂移会伪造差异。
        # 注意 show() 会 sim.wake()（防出生即休眠）→ 冻结期间 asleep 恒 False，
        # 所以「是否入睡」必须在这里读（冻结后读到的 False 是假象）。
        settle_steps = 0
        for settle_steps in range(1, 301):
            if sim.is_asleep():
                break
            sim.step(DT)
        settled = bool(sim.is_asleep())
        g = geom_stats(sim)
        ov = RugOverlay(screen, TEX, lambda w, h: sim, fps=1)
        try:
            ov.show()
            pump(0.35)
            # ⚠️ 关键：RugOverlay 构造把 fps clamp 到 ≥30 → 渲染 timer 30fps 会一直
            # sim.step()，A/B 抓图之间位姿会漂移（实测拖动场景 relief 贡献被抬高到 22、
            # AO 到 26 全靠这个假象）。冻结 tick：所有抓图必须来自同一位姿。
            tick_keep = ov._on_tick
            ov._on_tick = lambda: None
            ov._rebuild_geometry()
            pump(0.15)
            wnum = ov._panel.windowNumber()
            rgba = surface_of_window(int(wnum))
            if rgba is None:
                print(f"LOOK-{name} 抓窗口表面失败")
                continue
            r = render_stats(rgba, sim, float(screen.backingScaleFactor()), tex_lum)
            th = thickness_stats(ov, sim, rgba, float(screen.backingScaleFactor()))
            Image.fromarray(rgba[..., [2, 1, 0, 3]], "RGBA").save(f"/tmp/look_{name}.png")
        finally:
            try:
                ov._on_tick = tick_keep
            except Exception:
                pass
            ov.hide()
            pump(0.15)
        print(f"LOOK-{name:6s} coverage={r['coverage']:.3f} holes={r['holes']*100:.2f}% "
              f"white_edge={r['white_edge']*100:.2f}% lum_ratio={r['lum_ratio']:.2f} | "
              f"zmax={g['zmax']:.0f} z99={g['z_p99']:.0f} lifted={g['lifted']} "
              f"flipped={g['flipped']} two_layer={g['two_layer']} "
              f"settled={settled}({settle_steps * DT:.1f}s) "
              f"-> /tmp/look_{name}.png")
        print(f"LOOK-{name:6s} 厚度：rim_dlum={th.get('rim_dlum', float('nan')):.2f} "
              f"rim_lit/dim={th.get('rim_lit', float('nan')):.0f}/{th.get('rim_dim', float('nan')):.0f} "
              f"relief_dlum={th.get('relief_dlum', float('nan')):.2f} "
              f"ao_dlum={th.get('ao_dlum', float('nan')):.1f}({th.get('ao_area', 0.0):.2f}%) "
              f"shadow_dalp={th.get('shadow_dalp', float('nan')):.1f} "
              f"halo_min={th.get('halo_min', float('nan')):.0f} "
              f"edge_px={th.get('edge_px', -1):.1f} outline_grow={th.get('outline_grow', float('nan')):.1f}px")
    return 0


if __name__ == "__main__":
    sys.exit(main())
