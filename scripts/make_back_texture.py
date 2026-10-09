#!/usr/bin/env python3
"""make_back_texture.py — 从毯子正面贴图生成「背面」贴图（v4 真翻折用）。

背面 = 水平镜像 + 压暗 + 轻微模糊（模拟地毯背面的织底：图案反了、颜色暗、纹理糊）。
输出：<名字>-back.png（与正面同目录）；RugOverlay 会自动加载同目录的 -back.png，
不存在则回退用正面贴图。

用法：
    .venv/bin/python scripts/make_back_texture.py [正面.png ...]
    （缺省 = assets/rugs/ 下所有 rug-*.png，跳过已有的 -back.png）

实现：CoreGraphics 读像素（RGBA 预乘）→ numpy 变换（预乘下"乘 RGB 系数" = 正确的
压暗，且不破坏边缘透明度）→ CoreGraphics 写回 PNG。无其他依赖。
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
from Foundation import NSData, NSURL
from Quartz import (
    CGImageSourceCreateWithURL, CGImageSourceCreateImageAtIndex,
    CGImageGetWidth, CGImageGetHeight, CGImageGetBytesPerRow, CGImageGetBitsPerPixel,
    CGImageGetDataProvider, CGDataProviderCopyData,
    CGImageCreate, CGColorSpaceCreateDeviceRGB, CGDataProviderCreateWithCFData,
    CGImageDestinationCreateWithURL, CGImageDestinationAddImage, CGImageDestinationFinalize,
    kCGImageAlphaPremultipliedLast, kCGBitmapByteOrderDefault, kCGRenderingIntentDefault)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DARKEN = 0.72      # 背面压暗系数（v5：0.55 太黑，顶视翻折处像"墨块"；真毯背面图案清晰、
                   # 只是略暗，观感见 §12.17）
BLUR_R = 0          # 盒式模糊半径（px；v6.6：2 → 0——用户反馈折痕处"毛糙模糊"，
                   # 真毯背面的图案是清晰的，只是颜色暗；模糊留给渲染的 AO 去做）


def _read_rgba(path: str):
    src = CGImageSourceCreateWithURL(NSURL.fileURLWithPath_(path), None)
    if src is None:
        raise SystemExit(f"读不到图片：{path}")
    img = CGImageSourceCreateImageAtIndex(src, 0, None)
    w, h = int(CGImageGetWidth(img)), int(CGImageGetHeight(img))
    bpr, bpp = int(CGImageGetBytesPerRow(img)), int(CGImageGetBitsPerPixel(img))
    if bpp != 32 or bpr < w * 4:
        raise SystemExit(f"{os.path.basename(path)} 不是 32bpp RGBA（bpp={bpp}），"
                         f"请先用 sips 转成带 alpha 的 PNG")
    buf = bytes(CGDataProviderCopyData(CGImageGetDataProvider(img)))
    arr = np.frombuffer(buf, dtype=np.uint8)[: bpr * h].reshape(h, bpr // 4, 4)[:, :w, :]
    return arr.copy(), (w, h)


def _box_blur(a: np.ndarray, r: int) -> np.ndarray:
    """可分离盒式模糊（cumsum 实现；线性运算 → 预乘 RGBA 可直接用）。"""
    if r <= 0:
        return a
    k = 2 * r + 1
    p = np.pad(a, ((r, r), (r, r), (0, 0)), mode="edge")
    c = np.cumsum(p, axis=0)
    c = np.concatenate([np.zeros_like(c[:1]), c], axis=0)
    p = (c[k:] - c[:-k]) / k
    c = np.cumsum(p, axis=1)
    c = np.concatenate([np.zeros_like(c[:, :1]), c], axis=1)
    p = (c[:, k:] - c[:, :-k]) / k
    return np.clip(p, 0, 255)


def make_back(src_path: str) -> str:
    arr, (w, h) = _read_rgba(src_path)
    a = arr.astype(np.float32)
    a = a[:, ::-1, :]                      # 水平镜像（翻到背面看图案是反的）
    a[:, :, :3] *= DARKEN                  # 压暗（预乘下合法：只缩 RGB、保留 alpha）
    a = _box_blur(a, BLUR_R)               # 织底纹理模糊
    out = a.astype(np.uint8)
    data = NSData.dataWithBytes_length_(out.tobytes(), out.nbytes)
    provider = CGDataProviderCreateWithCFData(data)
    cs = CGColorSpaceCreateDeviceRGB()
    img = CGImageCreate(w, h, 8, 32, w * 4, cs,
                        kCGImageAlphaPremultipliedLast | kCGBitmapByteOrderDefault,
                        provider, None, False, kCGRenderingIntentDefault)
    dst_path = src_path[:-4] + "-back.png"
    dest = CGImageDestinationCreateWithURL(NSURL.fileURLWithPath_(dst_path), "public.png", 1, None)
    CGImageDestinationAddImage(dest, img, None)
    if not CGImageDestinationFinalize(dest):
        raise SystemExit(f"写出失败：{dst_path}")
    return dst_path


def main(argv: list[str]) -> int:
    paths = argv[1:]
    if not paths:
        paths = sorted(p for p in glob.glob(os.path.join(ROOT, "assets", "rugs", "rug-*.png"))
                       if not p.endswith("-back.png"))
    if not paths:
        print("没有需要处理的正面贴图")
        return 1
    for p in paths:
        out = make_back(p)
        size = os.path.getsize(out)
        print(f"BACK-OK {p} -> {out} ({size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
