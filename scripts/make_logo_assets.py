#!/usr/bin/env python3
"""make_logo_assets.py — 从品牌 SVG 生成全部图标资产（应用图标 + 菜单栏模板图标）。

用法（仓库根执行）：
    .venv/bin/python scripts/make_logo_assets.py            # 用默认源 SVG 全量重建
    .venv/bin/python scripts/make_logo_assets.py --check    # 只渲染 + 终端 ASCII 预览，不落盘
    .venv/bin/python scripts/make_logo_assets.py --svg path/to/logo.svg

产出：
    icon.iconset/icon_*.png + icon.icns          应用图标（macOS 网格：1024 画布内 824 圆角主体）
    assets/logo/menubar.png / @2x / @3x          菜单栏模板图标（18pt，黑 + alpha，系统自动适配深浅色）
    assets/logo/logo-mark-1024.png               品牌标记（透明底，文档/README 用）
    icon_temp.png                                1024 预览图（历史文件名，保留）

为什么这么渲染：macOS 13+ 的 NSImage 能直接吃 SVG 并保持矢量，所以图标都是「按目标像素尺寸
现场渲染 + 抗锯齿圆角遮罩」，不是把一张大图缩下来——16/32px 的小尺寸才不糊。
菜单栏图标必须是 template（纯 alpha 蒙版 + 黑色），系统按当前菜单栏配色自动反色。
"""
from __future__ import annotations

import argparse
import io
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SVG = ROOT / "assets" / "logo" / "applogo.svg"
ICONSET_SPEC = [  # (文件名, 像素边) —— iconutil 只认这套标准名字
    ("icon_16x16.png", 16), ("icon_16x16@2x.png", 32),
    ("icon_32x32.png", 32), ("icon_32x32@2x.png", 64),
    ("icon_128x128.png", 128), ("icon_128x128@2x.png", 256),
    ("icon_256x256.png", 256), ("icon_256x256@2x.png", 512),
    ("icon_512x512.png", 512), ("icon_512x512@2x.png", 1024),
]
BODY_RATIO = 824.0 / 1024.0     # macOS 图标网格：1024 画布里的可见主体
CORNER_RATIO = 185.4 / 824.0    # 主体圆角半径（Big Sur 起的系统观感）
MENUBAR_PT = 18.0               # 菜单栏图标点尺寸（状态栏标准高度 24pt）
MENUBAR_INSET = 0.75            # 四周留白（pt），让图形不顶到菜单栏上下边
ASCII_CHARS = " .:-=+*#%@"


# --------------------------------------------------------------------------- 渲染
def _appkit():
    import objc  # noqa: F401
    from AppKit import NSBitmapImageRep, NSGraphicsContext, NSImage, NSImageInterpolationHigh
    return NSBitmapImageRep, NSGraphicsContext, NSImage, NSImageInterpolationHigh


def render_svg(svg: Path, w: int, h: int) -> Image.Image:
    """把 SVG 按 (w,h) 像素渲染成 RGBA 位图（保持矢量锐度，与屏幕缩放无关）。"""
    (NSBitmapImageRep, NSGraphicsContext, NSImage,
     NSImageInterpolationHigh) = _appkit()
    from AppKit import (NSCompositingOperationSourceOver, NSDeviceRGBColorSpace,
                        NSBitmapImageFileTypePNG, NSMakeRect, NSZeroRect)
    img = NSImage.alloc().initWithContentsOfFile_(str(svg))
    if img is None:
        raise SystemExit(f"✗ NSImage 读不了这个 SVG：{svg}（macOS 13+ 才支持 SVG）")
    rep = NSBitmapImageRep.alloc(
    ).initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_(
        None, w, h, 8, 4, True, False, NSDeviceRGBColorSpace, 0, 0)
    ctx = NSGraphicsContext.graphicsContextWithBitmapImageRep_(rep)
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.setCurrentContext_(ctx)
    ctx.setImageInterpolation_(NSImageInterpolationHigh)
    img.drawInRect_fromRect_operation_fraction_(NSMakeRect(0, 0, w, h), NSZeroRect,
                                                NSCompositingOperationSourceOver, 1.0)
    NSGraphicsContext.restoreGraphicsState()
    png = rep.representationUsingType_properties_(NSBitmapImageFileTypePNG, {})
    return Image.open(io.BytesIO(bytes(png))).convert("RGBA")


def strip_background_rect(text: str, canvas: int) -> tuple[str, bool]:
    """去掉铺满整张画布的底板 rect（图标底板由本脚本按 macOS 网格自己画）。"""
    dropped = False

    def repl(m: re.Match) -> str:
        nonlocal dropped
        tag = m.group(0)
        w = re.search(r'width="([\d.]+)"', tag)
        h = re.search(r'height="([\d.]+)"', tag)
        if not (w and h):
            return tag
        if float(w.group(1)) >= 0.9 * canvas and float(h.group(1)) >= 0.9 * canvas:
            dropped = True
            return ""
        return tag

    return re.sub(r"<rect\b[^>]*/?>", repl, text, flags=re.S), dropped


def mark_only(svg: Path, canvas: int = 1024) -> tuple[Image.Image, str]:
    """渲染「只有标记、没有底板」的透明底位图；返回 (图, 说明)。"""
    text = svg.read_text(encoding="utf-8")
    stripped, dropped = strip_background_rect(text, canvas)
    if dropped:
        with tempfile.NamedTemporaryFile("w", suffix=".svg", delete=False,
                                         encoding="utf-8") as fh:
            fh.write(stripped)
            tmp = Path(fh.name)
        try:
            img = render_svg(tmp, canvas, canvas)
        finally:
            tmp.unlink(missing_ok=True)
        return img, "已去掉底板 rect"
    return render_svg(svg, canvas, canvas), "无铺满底板 rect，直接用整图"


def alpha_bbox(img: Image.Image) -> tuple[int, int, int, int]:
    box = img.getchannel("A").getbbox()
    if box is None:
        raise SystemExit("✗ 渲染结果全透明：SVG 里没有可见图形？")
    return box


# --------------------------------------------------------------------------- 应用图标
def rounded_mask(size: int, radius: float) -> Image.Image:
    """抗锯齿圆角遮罩（PIL 画的是硬边，所以 4× 超采样再缩回来）。"""
    ss = 4
    m = Image.new("L", (size * ss, size * ss), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, size * ss - 1, size * ss - 1],
                                        radius=radius * ss, fill=255)
    return m.resize((size, size), Image.LANCZOS)


def app_icon(svg: Path, size: int) -> Image.Image:
    """单个尺寸的应用图标：按 macOS 网格缩到 824/1024 主体，再套抗锯齿圆角遮罩。"""
    body = max(1, round(size * BODY_RATIO))
    offset = (size - body) // 2
    content = render_svg(svg, body, body)
    content.putalpha(ImageChops.multiply(content.getchannel("A"),
                                         rounded_mask(body, CORNER_RATIO * body)))
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(content, (offset, offset))
    return canvas


# --------------------------------------------------------------------------- 菜单栏图标
def menubar_template(mark: Image.Image, px: int) -> Image.Image:
    """菜单栏模板图标：黑色 + alpha 蒙版（标记里的半透明层次在 alpha 里保留）。"""
    inset = round(px * MENUBAR_INSET / MENUBAR_PT)
    box = max(1, px - 2 * inset)
    keep = mark.crop(alpha_bbox(mark))
    scale = box / max(keep.size)
    target = (max(1, round(keep.width * scale)), max(1, round(keep.height * scale)))
    glyph = keep.resize(target, Image.LANCZOS)
    out = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    out.paste(Image.new("RGBA", target, (0, 0, 0, 255)), ((px - target[0]) // 2,
                                                          (px - target[1]) // 2),
              glyph.getchannel("A"))
    return out


# --------------------------------------------------------------------------- 预览
def ascii_preview(img: Image.Image, cols: int = 56, rows: int = 28,
                  use_alpha: bool = False) -> str:
    """把位图打成 ASCII 预览——本模型看不了图片，这是「肉眼」检查图标的唯一手段。

    use_alpha=True 用于 template 图标（黑 + alpha）：亮度恒为 0，只有 alpha 有意义。
    """
    small = img.convert("RGBA").resize((cols, rows * 2), Image.LANCZOS)
    lines = []
    for y in range(rows):
        row = ""
        for x in range(cols):
            r, g, b, a = small.getpixel((x, y * 2))
            if use_alpha:
                lum = a  # 深色菜单栏里 template 按 alpha 反白显示
            else:
                lum = (0.299 * r + 0.587 * g + 0.114 * b) * (a / 255.0)
            row += ASCII_CHARS[min(9, int(lum * 10 // 256))]
        lines.append(row)
    return "\n".join(lines)


def coverage(img: Image.Image) -> str:
    a = img.getchannel("A")
    hist = a.histogram()
    total = sum(hist)
    opaque = sum(hist[200:]) / total
    any_alpha = sum(hist[16:]) / total
    return f"不透明 {opaque * 100:.1f}% / 有任何内容 {any_alpha * 100:.1f}%"


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="从品牌 SVG 生成图标资产")
    ap.add_argument("--svg", default=str(DEFAULT_SVG), help="源 SVG（默认 assets/logo/applogo.svg）")
    ap.add_argument("--check", action="store_true", help="只渲染并打印 ASCII 预览，不写文件")
    args = ap.parse_args()

    svg = Path(args.svg)
    if not svg.is_file():
        raise SystemExit(f"✗ 找不到源 SVG：{svg}")
    mark, note = mark_only(svg)
    print(f"源 SVG：{svg}（{note}）")
    bbox = alpha_bbox(mark)
    print(f"标记外框：{bbox}（宽 {bbox[2] - bbox[0]} × 高 {bbox[3] - bbox[1]}）")

    if args.check:
        icon = app_icon(svg, 1024)
        print("\n=== 应用图标 1024（{0}） ===".format(coverage(icon)))
        print(ascii_preview(icon))
        glyph = menubar_template(mark, 72)
        print("\n=== 菜单栏图标 @4x 放大（{0}） ===".format(coverage(glyph)))
        print(ascii_preview(glyph, 36, 18, use_alpha=True))
        small = menubar_template(mark, 36)
        print("\n=== 菜单栏图标 @2x 原尺寸（36px，'#'=实心 '='=半透） ===")
        for y in range(36):
            print("".join("#" if small.getpixel((x, y))[3] > 200
                          else "=" if small.getpixel((x, y))[3] > 40 else "."
                          for x in range(36)))
        return 0

    # 应用图标：iconset + icns
    iconset = ROOT / "icon.iconset"
    iconset.mkdir(exist_ok=True)
    for stale in iconset.glob("*.png"):
        stale.unlink()
    master = app_icon(svg, 1024)
    for name, px in ICONSET_SPEC:
        app_icon(svg, px).save(iconset / name)
    master.save(ROOT / "icon_temp.png")
    subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(ROOT / "icon.icns")],
                   check=True)
    print(f"✓ 应用图标：{iconset.name}/（{len(ICONSET_SPEC)} 张）+ icon.icns（"
          f"{ICONSET_SPEC[-1][1]}px 主体 {round(ICONSET_SPEC[-1][1] * BODY_RATIO)}px 圆角）")

    # 菜单栏模板图标（1x/2x/3x 三份，运行时打包成一个 NSImage 的三档 rep）
    logo_dir = ROOT / "assets" / "logo"
    logo_dir.mkdir(parents=True, exist_ok=True)
    for suffix, factor in (("", 1), ("@2x", 2), ("@3x", 3)):
        px = round(MENUBAR_PT * factor)
        menubar_template(mark, px).save(logo_dir / f"menubar{suffix}.png")
    mark.crop(bbox).save(logo_dir / "logo-mark-1024.png")
    print(f"✓ 菜单栏图标：assets/logo/menubar.png（{MENUBAR_PT:.0f}pt @1x/@2x/@3x，template）")
    print(f"✓ 品牌标记：assets/logo/logo-mark-1024.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
