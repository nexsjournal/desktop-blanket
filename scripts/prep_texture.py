#!/usr/bin/env python3
"""贴图预处理：assets/rugs/blanket.webp → assets/rugs/rug-01.png

全部用数值分析（边缘纯色检测/内容包围盒/旋转/缩放），不涉及人工看图。
输出报告到 stdout，数字说话。
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SRC = Path("assets/rugs/blanket.webp")
DST = Path("assets/rugs/rug-01.png")
TARGET_W = 2048
BORDER = 24          # 边框采样宽度（px）
UNIFORM_STD = 14.0   # 边框各通道 std 低于此值 → 判定为纯色背景
COLOR_DIST = 42.0    # 与背景色的距离阈值（判定"内容"像素）

img = Image.open(SRC)
print(f"source: {img.size} mode={img.mode}")
if img.mode != "RGB":
    img = img.convert("RGB")
arr = np.asarray(img).astype(np.int16)
h, w = arr.shape[:2]

# 1) 边框纯色检测
frame = np.concatenate([
    arr[:BORDER].reshape(-1, 3), arr[-BORDER:].reshape(-1, 3),
    arr[:, :BORDER].reshape(-1, 3), arr[:, -BORDER:].reshape(-1, 3),
])
border_std = frame.std(axis=0)
bg = np.median(frame, axis=0)
is_uniform = bool((border_std < UNIFORM_STD).all())
print(f"border std RGB = {border_std.round(1).tolist()} median = {bg.tolist()} "
      f"uniform={is_uniform}")

# 2) 内容包围盒（仅当边框是纯色背景时裁剪）
if is_uniform:
    dist = np.linalg.norm(arr - bg[None, None, :], axis=2)
    mask = dist > COLOR_DIST
    row_frac = mask.mean(axis=1)
    col_frac = mask.mean(axis=0)
    rows = np.where(row_frac > 0.015)[0]
    cols = np.where(col_frac > 0.015)[0]
    if len(rows) and len(cols):
        box = (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)
        area_ratio = (box[2] - box[0]) * (box[3] - box[1]) / (w * h)
        print(f"content bbox={box} area_ratio={area_ratio:.2f}")
        if area_ratio >= 0.25:
            m = 8
            box = (max(0, box[0] - m), max(0, box[1] - m),
                   min(w, box[2] + m), min(h, box[3] + m))
            img = img.crop(box)
            arr = arr[box[1]:box[3], box[0]:box[2]]
            print(f"cropped -> {img.size}")
        else:
            print("content bbox too small / noisy -> keep full frame")
    else:
        print("no content detected -> keep full frame")
else:
    print("border textured (blanket fills frame) -> keep full frame")

# 3) 方向：竖版转横版（毛毯贴图需要 landscape）
w2, h2 = img.size
if h2 > w2:
    img = img.transpose(Image.ROTATE_90)
    print(f"rotated 90° -> {img.size}")

# 4) 缩放到 TARGET_W 宽
w3, h3 = img.size
if w3 > TARGET_W:
    img = img.resize((TARGET_W, round(h3 * TARGET_W / w3)), Image.LANCZOS)
print(f"final size={img.size} ratio=1:{img.size[1] / img.size[0]:.2f}")

DST.parent.mkdir(parents=True, exist_ok=True)
img.save(DST)
print(f"saved -> {DST} ({DST.stat().st_size // 1024} KB)")
sys.exit(0)
