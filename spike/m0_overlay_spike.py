#!/usr/bin/env python3
"""M0 spike：验证 Python + PyObjC 能否实现「桌面图标层之上」的 SceneKit 覆盖窗口。

验证点：
  1. NSPanel 能否放到 kCGDesktopIconWindowLevel+8（图标之上、普通窗口之下）
  2. SceneKit 自定义网格能否逐帧重建（numpy → SCNGeometry 的性能）
  3. 鼠标事件能否到达该层级窗口；hitTest 返回 nil 的区域是否让事件穿透
运行约 7 秒后自动退出，无任何残留。
"""
import os
import sys
import time
import traceback

import numpy as np
import objc
from Foundation import NSData, NSMakeRect, NSTimer
from AppKit import (NSApplication, NSApplicationActivationPolicyAccessory,
                    NSBackingStoreBuffered, NSBezierPath, NSColor, NSScreen,
                    NSPanel, NSView, NSWindowStyleMaskBorderless,
                    NSWindowCollectionBehaviorCanJoinAllSpaces,
                    NSWindowCollectionBehaviorStationary,
                    NSWindowCollectionBehaviorIgnoresCycle)
from Quartz import (CGWindowListCopyWindowInfo, kCGWindowListOptionAll,
                    kCGNullWindowID)

try:
    from Quartz import kCGDesktopIconWindowLevel as ICON_LEVEL_BASE
    LEVEL_SRC = "Quartz.kCGDesktopIconWindowLevel"
except ImportError:
    ICON_LEVEL_BASE = -2147483023
    LEVEL_SRC = "hardcoded"
LEVEL = int(ICON_LEVEL_BASE) + 8

import SceneKit

GRID = (90, 56)
stats = {"frames": 0, "total_ms": 0.0, "max_ms": 0.0, "events": 0, "errors": []}


def build_indices(w, h):
    idx = []
    for j in range(h - 1):
        for i in range(w - 1):
            a = j * w + i
            b, c, d = a + 1, a + w, a + w + 1
            idx += [a, b, d, a, d, c]
    return np.array(idx, dtype=np.uint32)


W, H = GRID
xs = np.linspace(-1.0, 1.0, W).astype(np.float32)
ys = np.linspace(-0.6, 0.6, H).astype(np.float32)
xx, yy = np.meshgrid(xs, ys)
verts = np.zeros((W * H, 3), dtype=np.float32)
verts[:, 0] = xx.reshape(-1)
verts[:, 1] = yy.reshape(-1)
indices = build_indices(W, H)


def make_vertex_source():
    data = NSData.dataWithBytes_length_(verts.tobytes(), verts.nbytes)
    return SceneKit.SCNGeometrySource.alloc().initWithData_semantic_colorSpace_vectorCount_floatComponents_componentsPerVector_bytesPerComponent_dataOffset_dataStride_(
        data, SceneKit.SCNGeometrySourceSemanticVertex, None, W * H, True, 3, 4, 0, 12)


try:
    idata = NSData.dataWithBytes_length_(indices.tobytes(), indices.nbytes)
    elem = SceneKit.SCNGeometryElement.alloc().initWithData_primitiveType_primitiveCount_indicesChannelCount_interleavedIndicesChannels_bytesPerIndex_(
        idata, SceneKit.SCNGeometryPrimitiveTypeTriangles, int(indices.size) // 3,
        1, False, 4)
    geo = SceneKit.SCNGeometry.geometryWithSources_elements_(
        [make_vertex_source()], [elem])
    mat = SceneKit.SCNMaterial.material()
    mat.diffuse().setContents_(NSColor.colorWithCalibratedRed_green_blue_alpha_(
        0.75, 0.10, 0.10, 1.0))
    mat.setLightingModelName_("SCNLightingModelConstant")
    geo.setMaterials_([mat])
except Exception:
    traceback.print_exc()
    print("SPIKE RESULT: SceneKit geometry setup FAILED", flush=True)
    sys.exit(1)


class ContainerView(NSView):
    def drawRect_(self, r):
        NSColor.colorWithCalibratedRed_green_blue_alpha_(
            0.75, 0.08, 0.08, 0.5).setFill()
        NSBezierPath.fillRect_(self.bounds())

    def hitTest_(self, p):
        # 左 60% 可交互（验证事件可达），右侧穿透（验证 hitTest nil 穿透机制）
        return self if p.x < self.bounds().size.width * 0.6 else None

    def mouseDown_(self, e):
        stats["events"] += 1
        print("EVENT mouseDown", e.locationInWindow(), flush=True)

    def mouseDragged_(self, e):
        stats["events"] += 1

    def mouseUp_(self, e):
        stats["events"] += 1
        print("EVENT mouseUp", flush=True)

    def tick_(self, timer):
        try:
            t = time.time()
            z = (np.sin(xx * 9 + t * 4) * 0.08
                 + np.cos(yy * 7 - t * 3) * 0.05) * 0.35
            verts[:, 2] = z.reshape(-1).astype(np.float32)
            t0 = time.time()
            geo2 = SceneKit.SCNGeometry.geometryWithSources_elements_(
                [make_vertex_source()], [elem])
            node.setGeometry_(geo2)
            dt = (time.time() - t0) * 1000.0
            stats["frames"] += 1
            stats["total_ms"] += dt
            stats["max_ms"] = max(stats["max_ms"], dt)
        except Exception:
            stats["errors"].append(traceback.format_exc())
            NSApplication.sharedApplication().terminate_(None)

    def finish_(self, timer):
        avg = stats["total_ms"] / stats["frames"] if stats["frames"] else -1
        print(f"SPIKE RESULT: frames={stats['frames']} avg_rebuild={avg:.2f}ms "
              f"max_rebuild={stats['max_ms']:.2f}ms mouse_events={stats['events']} "
              f"errors={len(stats['errors'])}", flush=True)
        for e in stats["errors"]:
            print(e, flush=True)
        os._exit(0)

    def probe_(self, timer):
        try:
            stats["probe_count"] = stats.get("probe_count", 0) + 1
            if stats["probe_count"] == 1:
                infos = CGWindowListCopyWindowInfo(kCGWindowListOptionAll,
                                                   kCGNullWindowID)
                for w in infos:
                    layer = w.get("kCGWindowLayer", None)
                    if layer is None or not (-2147483610 < layer < -2147483580):
                        continue
                    owner = str(w.get("kCGWindowOwnerName", ""))
                    print(f"WINDOW owner={owner!r} layer={layer}", flush=True)
            if stats["probe_count"] >= 8:
                self.finish_(None)
        except Exception:
            stats["errors"].append(traceback.format_exc())


app = NSApplication.sharedApplication()
app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

screen = NSScreen.mainScreen().frame()
panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
    screen, NSWindowStyleMaskBorderless, NSBackingStoreBuffered, False)
panel.setLevel_(LEVEL)
panel.setOpaque_(False)
panel.setBackgroundColor_(NSColor.clearColor())
panel.setHasShadow_(False)
panel.setCollectionBehavior_(
    NSWindowCollectionBehaviorCanJoinAllSpaces
    | NSWindowCollectionBehaviorStationary
    | NSWindowCollectionBehaviorIgnoresCycle)
panel.setIgnoresMouseEvents_(False)

container = ContainerView.alloc().initWithFrame_(screen)
panel.setContentView_(container)

scn_frame = NSMakeRect(0, 0, screen.size.width * 0.6, screen.size.height * 0.6)
scn = SceneKit.SCNView.alloc().initWithFrame_(scn_frame)
scn.setBackgroundColor_(NSColor.clearColor())
scn.setOpaque_(False)

scene = SceneKit.SCNScene.scene()
node = SceneKit.SCNNode.nodeWithGeometry_(geo)
scene.rootNode().addChildNode_(node)
cam_node = SceneKit.SCNNode.node()
cam_node.setCamera_(SceneKit.SCNCamera.camera())
cam_node.setPosition_((0.0, 0.0, 2.2))
scene.rootNode().addChildNode_(cam_node)
scn.setScene_(scene)
scn.setAutoresizingMask_(15)  # width+height
container.addSubview_(scn)

panel.orderFrontRegardless()
print(f"SPIKE start: screens={NSScreen.screens().__len__()} "
      f"level={LEVEL} (base={LEVEL_SRC}={ICON_LEVEL_BASE})", flush=True)

NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
    1.0, container, "probe:", None, True)
NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
    1.0 / 30.0, container, "tick:", None, True)

app.run()
