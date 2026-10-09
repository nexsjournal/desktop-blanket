"""菜单栏项可见性探针：创建一个（或两个）NSStatusItem，让外部用 AX 量它落在哪。

为什么需要它：macOS 26 上 status item 由系统（控制中心）渲染，进程内
`button.window().frame()` 读到的是 34x0 @ (0,0) 的假值，不能用来判断可见性；
只有 AX（System Events）能看到真实坐标——正 x 在屏内，负 x = 被系统挤到屏外。

用法：
    .venv/bin/python scripts/probe_status_item.py [秒数] [阻塞秒数]
默认建 2 个探针项（一个带 autosaveName、一个不带）并打进程内读数；另开终端用 AX 量坐标：
    osascript -e 'tell application "System Events" to repeat with p in (every process) ...
    读 menu bar 2 的 position/size'   # 正 x = 屏内可见，负 x = 被系统丢到屏外
（第 2 个参数 >0 时，建完项先把主线程阻塞 N 秒，用来验证「启动期忙」是否影响落位。）
"""
import sys
import time
from pathlib import Path

import objc  # noqa: F401
from AppKit import (NSApplication, NSApplicationActivationPolicyAccessory, NSImage,
                    NSMenu, NSMenuItem, NSStatusBar, NSVariableStatusItemLength,
                    NSImageScaleProportionallyDown, NSSize)

PROBE_LABEL_A = "RUGPROBE-A"   # 带 autosaveName
PROBE_LABEL_B = "RUGPROBE-B"   # 不带
PROBE_LABEL_C = "RUGPROBE-C"   # 用 icon.icns（复刻 rug.py 旧写法：非 template、1024px rep）
PROBE_LABEL_D = "RUGPROBE-D"   # 用 assets/logo/menubar@2x.png（新写法：template）
ICNS = "/Applications/桌面毛毯.app/Contents/Resources/AppIcon.icns"
MENUBAR_PNG = str(Path(__file__).resolve().parent.parent / "assets" / "logo" / "menubar@2x.png")


def make_icon(size=18.0):
    """临时画一个模板图标（实心方块），只为探针用；正式图标见 make_logo_assets.py。"""
    img = NSImage.alloc().initWithSize_(NSSize(size, size))
    img.lockFocus()
    from AppKit import NSColor, NSBezierPath, NSMakeRect
    NSColor.blackColor().set()
    NSBezierPath.fillRect_(NSMakeRect(2.0, 2.0, size - 4.0, size - 4.0))
    img.unlockFocus()
    img.setTemplate_(True)
    return img


def add_item(label, autosave_name=None, image=None, template=False):
    item = NSStatusBar.systemStatusBar().statusItemWithLength_(NSVariableStatusItemLength)
    if autosave_name:
        item.setAutosaveName_(autosave_name)
    img = image if image is not None else make_icon()
    if template:
        img.setTemplate_(True)
    item.button().setImage_(img)
    item.button().setImageScaling_(NSImageScaleProportionallyDown)
    item.button().setAccessibilityLabel_(label)
    item.button().setToolTip_(label)
    menu = NSMenu.alloc().init()
    menu.addItem_(NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("probe", None, ""))
    item.setMenu_(menu)
    return item


def load_icns(path):
    """复刻 rug.py 旧写法：整份 .icns（含 1024px rep）只把 size 设成 18pt。"""
    img = NSImage.alloc().initWithContentsOfFile_(path)
    if img is None:
        return None
    img.setSize_(NSSize(18.0, 18.0))
    return img


def load_template(path):
    """新写法：按 2x 像素图读入，size=18pt，标 template。"""
    rep = None
    from AppKit import NSBitmapImageRep
    data = Path(path).read_bytes()
    rep = NSBitmapImageRep.imageRepWithData_(data)
    if rep is None:
        return None
    img = NSImage.alloc().initWithSize_(NSSize(18.0, 18.0))
    img.addRepresentation_(rep)
    img.setSize_(NSSize(18.0, 18.0))
    img.setTemplate_(True)
    return img


def report(items):
    """进程内能读到的可见性信号——与外部 AX 实测对照，判断哪个可信。"""
    for label, item in items:
        line = f"{label}:"
        try:
            line += f" isVisible={bool(item.isVisible())}"
        except Exception as exc:
            line += f" isVisible=<{exc}>"
        try:
            f = item.button().window().frame()
            line += (f" window={f.origin.x:.0f},{f.origin.y:.0f}"
                     f" {f.size.width:.0f}x{f.size.height:.0f}")
        except Exception as exc:
            line += f" window=<{exc}>"
        line += f" image={item.button().image() is not None}"
        print(line, flush=True)


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0
    block = 0.0
    if len(sys.argv) > 2:
        block = float(sys.argv[2])       # 建完项后阻塞主线程 N 秒（模拟启动期重活）
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    items = [(PROBE_LABEL_A, add_item(PROBE_LABEL_A, "RugProbeA")),
             (PROBE_LABEL_B, add_item(PROBE_LABEL_B))]
    if block:
        print(f"blocking main thread {block:.0f}s after item creation...", flush=True)
        time.sleep(block)                # 不跑 runloop，主线程占住
    # 必须真跑 runloop：不跑的话 status item 不会注册进菜单栏，AX 里也看不到。
    from Foundation import NSTimer
    NSTimer.scheduledTimerWithTimeInterval_repeats_block_(2.0, False,
                                                          lambda _t: report(items))
    NSTimer.scheduledTimerWithTimeInterval_repeats_block_(seconds, False,
                                                          lambda _t: app.terminate_(None))
    print(f"probe running {seconds:.0f}s; items={len(items)} block={block:.0f}s", flush=True)
    app.run()


if __name__ == "__main__":
    main()
