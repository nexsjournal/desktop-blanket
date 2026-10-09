"""overlay.py — RugOverlay：全屏覆盖窗口（NSPanel）+ SCNView 渲染 + 逐像素穿透。

实现 contracts/interfaces.py 的 RugOverlay；桥接姿势逐条遵循 contract.md §4 与
spike/m0_overlay_spike.py 的真机实录。

要点：
  - NSPanel level = kCGDesktopIconWindowLevel + 8（动态读取 Quartz 常量；本机
    pyobjc 的 CGWindowLevelForKey 返回值不可信，见本文件 _icon_level_base 注释）
  - 透明三件套：panel / view setOpaque_(False) + clearColor + setHasShadow_(False)
  - 正交相机（usesOrthographicProjection=True，1 场景单位 = 1pt，视野 = 屏宽高），
    相机俯视（沿 -y 看，屏幕"上"方向 = 场景 -z）；渲染映射（contract.md §4）：
    场景 x = sim x、场景 y = sim z（高度朝相机）、场景 z = sim y
    （sim y 向下恰好与"屏幕上方向 = -z"自洽，无需取负）
  - 每帧 sim.step(dt) + 顶点 NSData 重建 SCNGeometry（索引 buffer 复用，
    spike 实测 0.21ms/帧）；法线由三角网格实时计算 + 平行光/环境光 Blinn
  - **v6「厚度」**（用户报「像薄纸片」，§12.18）：
      1. **滚边**：沿网格边界环向外+向下卷出的圆角厚边（CLOTH_THICKNESS_PT=7pt，
         顶点色带「受光→背光」梯度）——顶视图下毯子四周有真实的「材料厚度带」，
         不再是一条一刀切的纸边；
      2. **接触阴影裙**：沿边界环外扩的顶点色渐隐条（贴边最暗 → 4 环淡出），
         抬起时按光源反方向偏移并柔化 → 毯子是真的「放在桌面上」而不是贴纸；
      3. **静止微浮雕**：网格空间的静态低频高度场（两档波长），贴地满幅、离地衰减
         → 平铺区域也有柔和的明暗起伏（厚织物的软塌感）；
      4. **灯光重配**：环境光 0.55→0.42、平行光 0.85→0.525（平铺面总亮不变），
         明暗对比范围翻倍 → 折面/滚边/浮雕都看得见。
  - 折层遮蔽：sim.ao()（xy 粗网格「局部顶高−自身高度」）按顶点色压暗折缝
  - 根 NSView.hitTest_：sim.nearest_distance ≤ 40pt 返回 self，否则 None
    （注意：view 级 hitTest 返回 None **不**参与窗口路由，只吞事件）
  - 点击穿透闸门（唯一真正生效的机制）：动态 setIgnoresMouseEvents_ —— 毯外
    （nearest_distance > 40pt）→ panel ignores=YES（点击真正落到桌面/其它 App 窗口），
    毯上 → NO（收鼠标）；驱动源 = show() 的 30Hz NSTimer + 全局鼠标移动监视器 +
    窗口事件；决策唯一入口 RugOverlay._update_click_through（纯函数 gate_wants_ignore）
  - mouseDown/Dragged/Up → sim.grab / drag_to / release(fling=True)；
    NSCursor 换 grab 手型（macOS 26 选择器已改名 openHandCursor/closedHandCursor，
    旧名 openHand/closedHand 在本机 pyobjc 未桥接）
  - 模块级只 import numpy / PyObjC / SceneKit / stdlib；sim 经 sim_factory 注入，
    本模块禁止依赖 cloth / icons / bumps（保证可独立 import 验证）
"""
from __future__ import annotations

import math
import os
import time

import numpy as np
from Foundation import NSData, NSMakeRect, NSMakeSize, NSThread, NSTimer
from AppKit import (NSBackingStoreBuffered, NSBitmapImageRep, NSColor, NSCursor, NSEvent,
                    NSEventMaskLeftMouseDown, NSEventMaskLeftMouseUp,
                    NSEventMaskMouseMoved, NSEventMaskOtherMouseDown,
                    NSEventMaskOtherMouseUp, NSEventMaskRightMouseDown,
                    NSEventMaskRightMouseUp, NSImage, NSPanel, NSView,
                    NSWindowCollectionBehaviorCanJoinAllSpaces,
                    NSWindowCollectionBehaviorIgnoresCycle,
                    NSWindowCollectionBehaviorStationary,
                    NSWindowStyleMaskBorderless,
                    NSWindowStyleMaskNonactivatingPanel)
import SceneKit

try:  # 动态读取（不写死）；Quartz 缺失时回退到 contract.md §4 的实测常量
    from Quartz import kCGDesktopIconWindowLevel as _ICON_LEVEL_BASE
    from Quartz import (CGImageCreateWithImageInRect as _cg_image_crop,
                        CGRectMake as _cg_rect_make)
except Exception:  # pragma: no cover - 无 Quartz 环境才触发
    _ICON_LEVEL_BASE = -2147483603
    _cg_image_crop = None
    _cg_rect_make = None

__all__ = ["RugOverlay", "gate_wants_ignore", "icon_level_base"]

# ---- 常量（契约/文档初值，M1 可调） ----
GRAB_RADIUS_PT = 40.0          # 抓取半径（contract：nearest_distance ≤ 40pt 收鼠标）
SHADOW_PLANE_Y = -16.0         # 接触阴影所在场景高度（低于滚边最低点，避免遮住布面自身）
CAM_HEIGHT = 1000.0            # 相机高度（正交投影下只影响 near/far 覆盖范围）
# ---- v6「厚度」：滚边 / 接触阴影 / 静止浮雕 / 灯光（用户报「像薄纸片」→ §12.18） ----
CLOTH_THICKNESS_PT = 7.0              # 毯子厚度（pt）：滚边剖面半径（≈1.2cm 的厚地毯）
# 剖面角（度）：0=贴在顶面边缘（必须为 0，否则布面与滚边间会留缝）、90=赤道（最外缘）
EDGE_ROLL_ANGLES = (0.0, 34.0, 64.0, 90.0)
EDGE_ROLL_DARK = (1.0, 0.95, 0.80, 0.58)  # 各环顶点色暗化 → 滚边自带「受光 → 背光」梯度
EDGE_SEAM_FRAC = 0.5                  # 接缝法线倾角 = 该系数 × 第一环角（顶面/滚边各让一半）
EDGE_BAND_DARK = 0.80                 # 顶面最外一圈的暗化（毯与桌面的接触暗缝）
EDGE_BAND_CELLS = 1.6                 # 边缘暗化带宽（格）
RELIEF_AMP_PT = (4.4, 1.8)            # 静止微浮雕振幅（pt；两档：大波长 + 小波长）
RELIEF_CELLS = (18.0, 8.0)            # 对应波长（格）——≥7 格（≥90pt），避免「揉纸」碎纹
RELIEF_FADE_PT = 8.0                  # 离地高度达它 → 浮雕衰减为 0（抬起/翻折处保持干净）
RELIEF_CURV_SOFT = 2.0                # 曲率软阈值（pt/格²）：曲率越大浮雕越弱
                                      # （1/(1+(lap/2)²)；波峰/折脊上不叠加浮雕）
RELIEF_MIN_CELL = 0.30                # 格距比下限：压缩区按格距比缩放浮雕振幅（保底 0.30）
AO_STRENGTH = 0.34                    # 折层遮蔽场对顶点色的最大暗化（布在自己身上留的影）
SHADOW_SKIRT_PT = (-2.0, 6.0, 18.0, 34.0)     # 接触阴影裙横向半径（pt；首环内缩藏到布下）
SHADOW_SKIRT_ALPHA = (0.48, 0.38, 0.15, 0.0)  # 各环顶点色 alpha（<0.5：不计入 coverage 判定）
SHADOW_LIFT_GAIN = 1.0                # 抬升高度的阴影偏移/柔化增益（×1/tan(仰角)）
# 灯光（**sRGB 值**：SceneKit 对 NSColor 做色彩管理、转线性后才进光照 → 0.55 实际是线性 0.263）。
# 实测标定：平铺面线性响应 ≈ amb_lin + sun_lin×sin(仰角)。旧值 (0.55,0.85) → 线性 (0.263,0.694)
# → 响应 0.863（与实测 0.875 一致）；新值 (0.43,1.0) → 线性 (0.153,1.0) → 响应 0.860
# （总亮不变）但明暗斜率 +44%：浅起伏/滚边/折面都有看得见的明暗差。
AMBIENT_WHITE = 0.43
SUN_WHITE = 1.0

SUN_TO_LIGHT = (-0.5, 0.7071, -0.5)   # 场景坐标下「指向光源」的单位向量：左上偏前 →
                                      # 屏幕观感＝光在左上、影子落右下（抬升处偏移也用同一方向）
NORMAL_FLIP_KEEP = 0.30               # 朝下法线保留比例（原为全翻正 → 折脊两侧对称、无体积）
FALLBACK_ICON_LEVEL = -2147483603  # contract.md §4 的回退常量
GATE_TICK_S = 1.0 / 30.0       # 穿透闸门轮询周期（毯子未入睡 / 活动期）
GATE_TICK_IDLE_S = 1.0 / 5.0   # 毯子入睡后的闸门轮询周期（省空闲 CPU；全局监视器仍是即时主驱动）
GATE_LOG_MIN_INTERVAL_S = 1.0  # 闸门 flip 日志限频（防边界抖动刷屏）
GATE_ERR_LOG_MIN_INTERVAL_S = 5.0  # setIgnoresMouseEvents_ 失败日志限频（防逐帧刷屏）
GATE_CURSOR_EPS_PT = 0.5       # 空闲早退：光标位移小于它视为「没动」
GATE_NEAR_PT = 120.0           # 上次距离小于它（光标在毯子附近）时不做空闲早退：
                               # 保证「外来按住解除」「毯上抓取」等状态变化 ≤1 拍内被纠正
# ---- 编辑模式（拖动四角缩放 / 旋转柄转角度） ----
HANDLE_SIZE_PT = 14.0          # 角柄边长（pt）
ROTATE_SIZE_PT = 16.0          # 旋转柄边长（pt）
HANDLE_HIT_R_PT = 26.0         # 手柄命中半径（pt）
ROTATE_OFFSET_PT = 42.0        # 旋转柄距顶边的外侧距离（pt）
HANDLE_Z = 2.0                 # 手柄场景高度（略高于布面，配合 renderingOrder 压在最上层）
HANDLE_MIN_PT = 120.0          # 缩放下限（pt，避免缩到退化的极小值）
HANDLE_MAX_FRAC = 1.6          # 缩放上限（相对屏幕尺寸的比例）
# 全局监视器订阅的鼠标事件（鼠标类事件无需辅助功能权限；含 up/down 以便即时纠正按/放状态）。
# 用 AppKit 导出的 NSEventMask* 常量按名 OR（OtherMouse* 的位号反直觉，不要手写位移）。
GATE_MONITOR_MASK = (NSEventMaskMouseMoved
                     | NSEventMaskLeftMouseDown | NSEventMaskLeftMouseUp
                     | NSEventMaskRightMouseDown | NSEventMaskRightMouseUp
                     | NSEventMaskOtherMouseDown | NSEventMaskOtherMouseUp)
# 验证专用环境变量（仅 scripts/verify_routing.py --live 用，生产不要设）：
#   RUG_CLICKTHROUGH_FORCE=ignore|receive → 钉死 panel 的 ignoresMouseEvents 状态
CLICKTHROUGH_FORCE_ENV = "RUG_CLICKTHROUGH_FORCE"


def _log(level: str, msg: str) -> None:
    """contract.md §5：stdout 单行日志，禁止逐帧打印。"""
    print(f"{level} {msg}", flush=True)


def gate_wants_ignore(dist_pt: float, currently_ignoring: bool, grabbed: bool,
                      foreign_press: bool) -> bool:
    """点击穿透闸门决策（纯函数：无 I/O、无 AppKit、无状态，供单测）。

    grabbed：我们正抓着毯子 → False（永远收鼠标，否则拖动/抛掷断线）
    foreign_press：有鼠标键按下但不是我们抓的 → 冻结为 currently_ignoring
        （不夺走别的窗口的拖拽）
    否则：dist_pt > GRAB_RADIUS_PT → True（毯外穿透）；≤ 40 收鼠标
        （阈值与 hitTest 完全一致，无死区）
    """
    if grabbed:
        return False
    if foreign_press:
        return bool(currently_ignoring)
    return bool(dist_pt > GRAB_RADIUS_PT)


def _shadow_basis_from(to_light) -> tuple[tuple[float, float], float]:
    """由「指向光源」的单位向量给出（阴影屏幕方向, lean）——纯函数，可离屏单测。

    阴影方向 = 光源水平来向的反方向（屏幕 x = 场景 x、屏幕 y = 场景 z）；
    lean = 水平分量/竖直分量 = 1/tan(仰角)：物体抬升 h 时影子外移 h·lean。
    """
    lx, ly, lz = (float(v) for v in to_light)
    h = math.hypot(lx, lz)
    if h < 1e-6:
        return (0.0, 0.0), 0.6
    return (-lx / h, -lz / h), float(min(max(h / max(abs(ly), 1e-3), 0.0), 3.0))


def orientation_for_to_light(to_light) -> tuple[float, float, float, float]:
    """四元数 (x,y,z,w)：把节点局部 -z（平行光的传播方向）旋到 -to_light。

    不碰欧拉角：SceneKit 的 eulerAngles 组件序/正负号与常见右手系不一致（2026-10-08 实测
    三个轴全部反号），用四元数 + 建灯后 convertVector:toNode: 自校验（见 _sun_shadow_basis）
    避免约定陷阱。
    """
    lx, ly, lz = (float(v) for v in to_light)
    nrm = math.sqrt(lx * lx + ly * ly + lz * lz)
    if nrm < 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    dx, dy, dz = -lx / nrm, -ly / nrm, -lz / nrm          # 目标传播方向
    ax = 0.0 * dz - (-1.0) * dy                            # cross((0,0,-1), d)
    ay = (-1.0) * dx - 0.0 * dz
    az = 0.0 * dy - 0.0 * dx
    s = math.sqrt(ax * ax + ay * ay + az * az)
    c = -dz                                                # dot((0,0,-1), d)
    if s < 1e-9:
        return (0.0, 0.0, 0.0, 1.0) if c > 0.0 else (1.0, 0.0, 0.0, 0.0)
    ang = math.atan2(s, c)
    k = math.sin(ang / 2.0) / s
    return (ax * k, ay * k, az * k, math.cos(ang / 2.0))


def icon_level_base() -> int:
    """桌面图标层基准 kCGDesktopIconWindowLevel。

    动态读取 Quartz 导出的常量（随 SDK 走，不写死）；本机 pyobjc 2.x 的
    CGWindowLevelForKey 返回值不可信（传大负数报 wrong magnitude、正常调用返回 0，
    contract.md §4 备忘），故不经过它。读不到或值异常时回退 -2147483603（真机实测值）。
    """
    try:
        base = int(_ICON_LEVEL_BASE)
    except Exception:
        return FALLBACK_ICON_LEVEL
    if base >= 0:
        return FALLBACK_ICON_LEVEL
    return base


def _src(arr: np.ndarray, semantic, comps: int):
    """C-contiguous float32 numpy → SCNGeometrySource（contract.md §4 桥接姿势）。"""
    data = NSData.dataWithBytes_length_(arr.tobytes(), arr.nbytes)
    return SceneKit.SCNGeometrySource.alloc().initWithData_semantic_colorSpace_vectorCount_floatComponents_componentsPerVector_bytesPerComponent_dataOffset_dataStride_(
        data, semantic, None, int(arr.shape[0]), True, comps, 4, 0, comps * 4)


# ---- 贴图边缘裁切（v5，§12.17）----------------------------------------------
# 用户报障「毯子边缘一圈白框」的真根因：贴图自带纯白/浅色边距（照片底、生成图留白），
# 布料把 uv 0..1 映射到整张贴图，边距就变成毯子四周的白边。这里在**加载期**检测并
# 裁掉「纯色（或全透明）边距」——对用户自备贴图同样生效，不必手工修图。
TEXTURE_MARGIN_MAX_FRAC = 0.06   # 单边最多裁掉的比例（防把画面内容误判成边距）
TEXTURE_MARGIN_LIGHT = 200       # 「留白」判定：该行/列 min(R,G,B) 均值高于它（2023 psd 软渐变）
TEXTURE_MARGIN_SAMPLES = 96      # 每条边扫描时的采样点数
TEXTURE_MARGIN_INSET = 4         # 检测到留白后的保险内缩（px）
TEXTURE_MARGIN_RESIDUAL = 10     # 内缩后仍偏亮的过渡带最多再收多少 px


def _rep_pixels(rep):
    """NSBitmapImageRep → (h, w, samples) uint8 视图（行按 bytesPerRow 对齐，去填充）。"""
    w = int(rep.pixelsWide())
    h = int(rep.pixelsHigh())
    spp = int(rep.samplesPerPixel())
    bpr = int(rep.bytesPerRow())
    if w <= 0 or h <= 0 or spp < 3 or bpr % w != 0:
        return None
    stride = bpr // w
    buf = np.frombuffer(rep.bitmapData(), dtype=np.uint8, count=bpr * h)
    return buf.reshape(h, w, stride)[:, :, :spp]   # 行内按像素连续排列，丢掉行尾填充


def _edge_line(px: np.ndarray, side: str, k: int) -> np.ndarray:
    """取某条边向里第 k 条线（k=0 = 最外一条）。"""
    h, w = px.shape[0], px.shape[1]
    if side == "top":
        return px[k]
    if side == "bottom":
        return px[h - 1 - k]
    if side == "left":
        return px[:, k]
    return px[:, w - 1 - k]


def _light_frac(line: np.ndarray, samples: int) -> float:
    step = max(int(line.shape[0] // samples), 1)
    return float((np.min(line[::step, :3], axis=1) > TEXTURE_MARGIN_LIGHT).mean())


def _line_is_margin(line: np.ndarray, samples: int) -> bool:
    """一行/列是否属于「边距」：**过半数像素偏亮**（白底），或整体近乎透明。

    用"过半偏亮"而不是"整行都纯色/都亮"：照片留白里常掺着毯角、流苏、渗色
    （rug-01.png 顶边前 12 行纯白占比 92%→73%→60%→29%），严格判据会在第 4-6 行
    就停下、白边裁不干净；毯子自身边框是深色（min 通道 ~40-90）→ 判定很稳。
    """
    step = max(int(line.shape[0] // samples), 1)
    px = line[::step]
    if px.shape[1] >= 4 and float(px[:, 3].mean()) < 12.0:
        return True
    return float((np.min(px[:, :3], axis=1) > TEXTURE_MARGIN_LIGHT).mean()) > 0.5


def _detect_texture_margin(rep):
    """检测四边可裁边距 (top, left, bottom, right)（px）；无 NSBitmapImageRep 时全 0。

    两段式：① 主体留白（过半偏亮 = 白底）→ ② 过渡带收尾（外侧仍 >18% 偏亮就继续收，
    最多再多 RESIDUAL px）。② 是补独立复核发现的漏：rug-01.png 右侧留白比左侧窄、
    判定早停在 8px，裁后右边缘仍留一条 5px 浅色带（贴图里 min>200 占 33%）——
    渲染层面近白(min>225)已只剩 0.0x%，但为稳妥把过渡带也收掉。
    """
    px = _rep_pixels(rep) if rep is not None else None
    if px is None:
        return (0, 0, 0, 0)
    h, w = px.shape[0], px.shape[1]
    cap = {"top": max(int(h * TEXTURE_MARGIN_MAX_FRAC), 1),
           "bottom": max(int(h * TEXTURE_MARGIN_MAX_FRAC), 1),
           "left": max(int(w * TEXTURE_MARGIN_MAX_FRAC), 1),
           "right": max(int(w * TEXTURE_MARGIN_MAX_FRAC), 1)}
    out = {}
    for side in ("top", "bottom", "left", "right"):
        k = 0
        while k < cap[side] and _line_is_margin(_edge_line(px, side, k), TEXTURE_MARGIN_SAMPLES):
            k += 1
        if k:
            k = min(k + TEXTURE_MARGIN_INSET, cap[side])          # 保险内缩
            lim = min(k + TEXTURE_MARGIN_RESIDUAL, cap[side])     # 过渡带收尾
            while k < lim and _light_frac(_edge_line(px, side, k), TEXTURE_MARGIN_SAMPLES) > 0.18:
                k += 1
        out[side] = k
    top, bottom, left, right = out["top"], out["bottom"], out["left"], out["right"]
    if top + bottom >= h - 8 or left + right >= w - 8:   # 整张都被判成边距 → 不裁
        return (0, 0, 0, 0)
    return (top, left, bottom, right)


def _load_rug_image(path, crop=(0, 0, 0, 0)):
    """读贴图；crop 全 0 时自动检测边距并裁切。返回 (NSImage | None, 实际裁切量)。"""
    img = None
    try:
        img = NSImage.alloc().initWithContentsOfFile_(str(path))
    except Exception:
        img = None
    if img is None:
        return None, (0, 0, 0, 0)
    data = NSData.dataWithContentsOfFile_(str(path))
    rep = NSBitmapImageRep.alloc().initWithData_(data) if data is not None else None
    if crop == (0, 0, 0, 0):
        crop = _detect_texture_margin(rep)
    t, l, b, r = crop
    if t == 0 and l == 0 and b == 0 and r == 0:
        return img, (0, 0, 0, 0)
    try:
        w, h = int(rep.pixelsWide()), int(rep.pixelsHigh())
        cg = rep.CGImage()
        if _cg_image_crop is None:
            return img, (0, 0, 0, 0)      # 无 Quartz：不裁（退化为旧行为）
        sub = _cg_image_crop(cg, _cg_rect_make(l, t, w - l - r, h - t - b))
        if sub is None:
            return img, (0, 0, 0, 0)
        out = NSImage.alloc().initWithCGImage_size_(
            sub, NSMakeSize(w - l - r, h - t - b))
        return (out if out is not None else img), crop
    except Exception as exc:
        _log("RUG-WARN", f"贴图边距裁切失败（用原图）：{exc}")
        return img, (0, 0, 0, 0)


def _cursor(kind: str):
    """取标准光标实例；失败返回 None（光标是装饰性功能，绝不因它中断交互）。"""
    try:
        if kind == "closed":
            return NSCursor.closedHandCursor()
        if kind == "open":
            return NSCursor.openHandCursor()
        if kind == "cross":
            c = NSCursor.crosshairCursor()
            if c is not None:
                return c
        c = NSCursor.arrowCursor()
        if c is not None:
            return c
    except Exception:
        pass
    try:
        return NSCursor.currentCursor()
    except Exception:
        return None


def _set_cursor(kind: str) -> None:
    try:
        c = _cursor(kind)
        if c is not None:
            c.set()
    except Exception:
        pass


class RugRootView(NSView):
    """根视图：hitTest 逐像素判定 + 鼠标事件 → sim 交互 + NSTimer 渲染/闸门 tick。

    _owner 指向 RugOverlay（Python 侧属性注入）。注意（真机路由实测）：hitTest
    返回 None **不会**把点击交给下层窗口（只吞事件），所以「毯外穿透」由
    RugOverlay 的 ignoresMouseEvents 闸门负责；本视图的 hitTest_ 只在窗口已收
    鼠标时做 40pt 精确判定。
    """

    def _sim_xy_from_window_point(self, p):
        """**窗口坐标**（bottom-left）点 → sim 坐标（主屏本地 top-left、y 向下）。

        先 convertPoint_fromView_(p, None) 转本视图坐标（其实同窗口坐标，本视图铺满窗口），
        再翻 y；**不要**在这里走 NSEvent 路径：hitTest_ 收到的点本来就没有
        locationInWindow()（旧实现把它当 NSEvent 用 → AttributeError 被吞 → 永远穿透）。
        """
        q = self.convertPoint_fromView_(p, None)
        return float(q.x), float(self.bounds().size.height - q.y)

    def _sim_point(self, e):
        """NSEvent 的 window 坐标 → 主屏本地屏幕点（top-left, y-down）。"""
        return self._sim_xy_from_window_point(e.locationInWindow())

    # ---- 穿透 ----
    def hitTest_(self, p):
        """窗口收到事件后的 40pt 精确判定：毯上返回 self（收鼠标），否则 None。

        注意：本返回值只在窗口已收鼠标时有效（「毯外点击穿透」必须靠
        RugOverlay 的 ignoresMouseEvents 闸门，view 级 hitTest 不参与窗口路由）。
        """
        owner = getattr(self, "_owner", None)
        sim = owner._sim if owner is not None else None
        if sim is None or not owner.is_shown():
            return None
        try:
            x, y = self._sim_xy_from_window_point(p)  # hitTest 的 p 已是窗口坐标
            d = sim.nearest_distance(x, y)
            if getattr(owner, "edit_mode", False):  # 编辑模式：手柄也要能收鼠标（旋转柄在毯外）
                d = min(d, owner._handle_distance(x, y))
        except Exception:
            return None  # 异常时宁可穿透，不许卡死鼠标
        return self if d <= GRAB_RADIUS_PT else None

    def acceptsFirstMouse_(self, event):
        """非激活态（accessory 面板不是 key 窗口）首击也照常收下，不被吞。

        否则「点一下毯子」只是激活窗口、要第二下才抓得住（P1）。
        """
        return True

    def menuForEvent_(self, event):
        """右键 / Ctrl+左键 → 弹出毯子上下文菜单（菜单由 RugApp 现场构建）。

        毯外（>GRAB_RADIUS_PT）返回 None = 不弹（AppKit 默认行为）；本来闸门也会把
        毯外的鼠标事件让给桌面，所以这里基本只在毯上被调用。
        """
        owner = getattr(self, "_owner", None)
        provider = getattr(owner, "menu_provider", None) if owner is not None else None
        if provider is None:
            return None
        try:
            sim = owner._sim
            if sim is None or not owner.is_shown():
                return None
            x, y = self._sim_point(event)
            d = sim.nearest_distance(x, y)
            if getattr(owner, "edit_mode", False):
                d = min(d, owner._handle_distance(x, y))
            if d > GRAB_RADIUS_PT:
                return None
        except Exception:
            return None
        try:
            return provider(x, y)
        except Exception as exc:
            _log("RUG-WARN", f"上下文菜单构建失败：{exc}")
            return None

    # ---- 鼠标 → sim ----
    def mouseDown_(self, e):
        owner = getattr(self, "_owner", None)
        sim = owner._sim if owner is not None else None
        if sim is None:
            return
        try:
            x, y = self._sim_point(e)
        except Exception:
            return
        # 编辑模式：优先命中手柄（拖角缩放 / 拖旋转柄转角度）
        if owner is not None and getattr(owner, "edit_mode", False):
            try:
                hi = owner._handle_at(x, y)
            except Exception:
                hi = None
            if hi is not None:
                owner._begin_handle(hi, x, y)
                self._grabbed = True          # 借用「active drag」标记：闸门保持收鼠标
                self._handle_grab = True
                _set_cursor("cross")
                owner._fire_grab()
                return
        try:
            ok = sim.grab(x, y, GRAB_RADIUS_PT)
        except Exception as exc:
            _log("RUG-ERROR", f"sim.grab 异常: {exc}")
            return
        if ok:
            self._grabbed = True
            try:
                sim.wake()
            except Exception:
                pass
            _set_cursor("closed")
            owner._fire_grab()

    def mouseDragged_(self, e):
        if not getattr(self, "_grabbed", False):
            return
        owner = getattr(self, "_owner", None)
        sim = owner._sim if owner is not None else None
        if sim is None:
            return
        try:
            x, y = self._sim_point(e)
            if getattr(self, "_handle_grab", False) and owner is not None:
                owner._drag_handle(x, y)
            else:
                sim.drag_to(x, y)
        except Exception as exc:
            _log("RUG-ERROR", f"拖动处理异常: {exc}")

    def mouseUp_(self, e):
        owner = getattr(self, "_owner", None)
        if getattr(self, "_handle_grab", False):
            # 手柄手势结束：毯子已是规整形状，唤醒收尾即可
            self._handle_grab = False
            self._grabbed = False
            if owner is not None:
                try:
                    owner._end_handle()
                except Exception:
                    pass
            _set_cursor("arrow")
            if owner is not None:
                try:
                    owner._update_click_through()
                except Exception:
                    pass
                owner._fire_release()
            return
        if not getattr(self, "_grabbed", False):
            # 非我们的抓取（外来按下在毯子上方解除）：只需即时刷新闸门
            if owner is not None:
                try:
                    owner._update_click_through()
                except Exception:
                    pass
            return
        self._grabbed = False
        sim = owner._sim if owner is not None else None
        over = False
        if sim is not None:
            try:
                x, y = self._sim_point(e)
                sim.release(True)  # fling=True（contract：松手带抛掷）
            except Exception as exc:
                _log("RUG-ERROR", f"sim.release 异常: {exc}")
            try:
                over = sim.nearest_distance(*self._sim_point(e)) <= GRAB_RADIUS_PT
            except Exception:
                over = False
        _set_cursor("open" if over else "arrow")
        if owner is not None:
            # 释放后再重算闸门（此时 _grabbed 已复位，结果反映松手后的真实状态）
            try:
                owner._update_click_through()
            except Exception:
                pass
            owner._fire_release()

    def mouseMoved_(self, e):
        """鼠标在窗口上移动：即时刷新穿透闸门 + 编辑模式下按手柄悬停换光标。"""
        owner = getattr(self, "_owner", None)
        if owner is None:
            return
        try:
            owner._update_click_through()
        except Exception:
            pass
        if getattr(owner, "edit_mode", False) and not getattr(self, "_grabbed", False):
            try:
                x, y = self._sim_point(e)
                _set_cursor("cross" if owner._handle_at(x, y) is not None else "arrow")
            except Exception:
                pass

    def cursorUpdate_(self, e):
        """悬停手型（窗口非 key 时可能不触发，仅尽力而为；抓取态以 mouseDown 为准）。"""
        owner = getattr(self, "_owner", None)
        if owner is not None:
            try:
                owner._update_click_through()
            except Exception:
                pass
        if getattr(self, "_grabbed", False):
            _set_cursor("closed")
            return
        sim = owner._sim if owner is not None else None
        over = False
        if sim is not None:
            try:
                over = sim.nearest_distance(*self._sim_point(e)) <= GRAB_RADIUS_PT
            except Exception:
                over = False
        _set_cursor("open" if over else "arrow")

    # ---- 渲染 tick / 闸门 tick ----
    def tick_(self, timer):
        owner = getattr(self, "_owner", None)
        if owner is not None:
            owner._on_tick()

    def gateTick_(self, timer):
        """穿透闸门轮询 tick：按入睡状态调整周期 + 消费脏标志 + 重算 ignores 状态。"""
        owner = getattr(self, "_owner", None)
        if owner is not None:
            try:
                owner._on_gate_tick()
            except Exception:
                pass


class RugOverlay:
    """全屏覆盖窗口（NSPanel）+ SCNView 渲染 + 点击穿透闸门（契约公开面）。

    「毯外穿透」由 _update_click_through 动态切 setIgnoresMouseEvents_ 实现
    （毯外 → ignores=YES 让点击真正落到桌面/下层窗口；毯上 → NO 收鼠标），
    view 的 hitTest_ 只在收到事件后做 40pt 判定。
    """

    def __init__(self, screen, texture_path: str, sim_factory,
                 on_grab=None, on_release=None, fps: int = 30,
                 menu_provider=None) -> None:
        self._screen = screen
        self._texture_path = str(texture_path)
        self._sim_factory = sim_factory
        self._on_grab_cb = on_grab
        self._on_release_cb = on_release
        self._menu_provider = menu_provider   # (x, y) -> NSMenu：毯上右键菜单（rug.py 注入）
        self._fps = max(30, min(60, int(fps)))  # 契约：30~60
        self._sim = None
        self._shown = False
        self._built_gui = False
        self._panel = None
        self._view = None
        self._scn = None
        self._scene = None
        self._cloth_node = None
        self._shadow_node = None
        self._rim_node = None        # v6 滚边节点（沿边界环卷出的厚边）
        self._cloth_mat = None       # 布面材质（_build_gui 建，_rebuild_geometry 逐帧挂到新几何上）
        self._rim_mat = None         # 滚边材质（正面材质副本 + 双面：避免环绕方向风险）
        self._shadow_mat = None      # 接触阴影材质（同上）
        self._timer = None
        self._elem = None            # SCNGeometryElement（索引 buffer 复用）
        self._elem2 = None
        self._rim_elem = None        # 滚边索引（拓扑静态）
        self._skirt_elem = None      # 阴影裙索引（拓扑静态）
        self._color_src = None       # 布面顶点色源（逐帧：静态边缘带 × 动态 AO）
        self._rim_color_src = None   # 滚边顶点色源（静态）
        self._rows = 0
        self._cols = 0
        self._dx = 1.0
        self._dy = 1.0
        self._w = 0.0
        self._h = 0.0
        self._vbuf = None            # 场景顶点 (N,3) f32（深度 = 物理位置）
        self._vrel = None            # 法线用顶点 (N,3)：vb + 微浮雕（只进法线，不进深度）
        self._nbuf = None            # 场景法线 (N,3) f32
        self._ubuf = None            # 场景 UV (N,2) f32（v 已取 1-v）
        self._cbuf = None            # 布面顶点色 (N,4) f32
        self._eband = None           # 静态边缘暗化系数 (N,) f32
        self._w_relief = None        # 上一帧的浮雕权重 (N,) f32（诊断/测试用）
        self._relief = None          # 静态微浮雕高度 (N,) f32
        self._bloop = None           # 边界环顶点索引（有序、闭合）
        self._bloop_next = None
        self._bloop_prev = None
        self._rim_v = None           # 滚边顶点 (L*rings,3)
        self._rim_n = None
        self._rim_uv = None
        self._rim_c = None
        self._rim_uv_src = None      # 滚边 UV/顶点色源（拓扑静态，建一次复用）
        self._sbuf = None            # 阴影裙顶点 (L*rings,3)
        self._scbuf = None           # 阴影裙顶点色 (L*rings,4)
        self._shadow_dir = (0.0, 0.0)  # 光源反方向的屏幕 xy（建灯时按灯光实际朝向算）
        self._shadow_lean = 0.6      # 1/tan(仰角)：抬升 z → 阴影偏移/柔化系数
        self._last_t = None
        self._err_count = 0
        # ---- 点击穿透闸门状态（见 _update_click_through） ----
        self._ignoring = False        # 当前 panel 的 ignoresMouseEvents 状态
        self._gate_timer = None       # 闸门轮询 timer（show 装 / hide 撤；周期随入睡自适应）
        self._gate_period = None      # 当前闸门 timer 周期（None=未装）
        self._monitor_token = None    # 全局鼠标监视器令牌
        self._force_ignore = None     # None/True/False：验证专用覆盖（env 注入）
        self._last_cursor = None      # 上次闸门计算用的 sim 坐标（空闲早退用）
        self._last_dist = None        # 上次算出的「光标→毯子」距离（空闲早退的近距豁免用）
        self._gate_dirty = False      # 监视器非主线程回调置脏，由 timer 消费
        self._last_flip_log = 0.0     # flip 日志限频时间戳
        self._last_gate_err_log = 0.0 # 闸门写失败日志限频时间戳
        # ---- 编辑模式（四角缩放柄 + 旋转柄；见 set_edit_mode） ----
        self._edit_mode = False
        self._placement = None        # (cx, cy, w, h, angle_rad) 编辑/摆放状态
        self._handle = None           # 手势中：(handle_index, start_placement, start_xy)
        self._handle_nodes = []       # 手柄场景节点（含 4 角 + 1 旋转柄）
        self._last_edit_angle = 0.0   # 编辑手势中的上一帧角度（速度旋转用）

    # ---- 契约公开面 ----
    def show(self) -> None:
        """铺上窗口（幂等）：建/复用 panel+view，orderFrontRegardless，启动渲染 timer。"""
        if self._shown:
            return
        frame = self._screen.frame()
        self._sim = self._sim_factory(float(frame.size.width), float(frame.size.height))
        if self._sim is None:
            _log("RUG-ERROR", "sim_factory 返回 None，show 失败")
            return
        if not self._built_gui:
            self._build_gui(frame)
        self._prepare_geometry()
        self._rebuild_geometry()  # 首帧即有画面，不等第一个 tick
        # 铺上前先算一次初始闸门状态：毯外 → ignores=YES，点击直接落到桌面，
        # 避免 orderFront 后第一帧短暂吞掉桌面点击（_shown 先置位，闸门才生效）
        self._shown = True
        self._read_force_env()
        self._update_click_through(force=True)
        self._panel.orderFrontRegardless()
        self._last_t = None
        self._err_count = 0
        self._start_timer()
        self._start_gate()
        try:
            self._sim.wake()  # 防"出生即休眠"卡住后续 step
        except Exception:
            pass
        _log("RUG-INFO", f"overlay show（fps={self._fps}, level={int(self._panel.level())}, "
                         f"ignoring={self._ignoring}）")

    def hide(self) -> None:
        """掀开窗口（幂等）：停 timer、撤闸门驱动、orderOut、丢弃 sim（下次 show 重建）。"""
        self._stop_timer()
        self._stop_gate()
        # P1 修复：用户按住毯子时经快捷键/菜单触发掀开、款式切换或退出，mouseUp 可能不再
        # 回投到本视图 → view._grabbed 会残留 True → 复铺后闸门恒 receive（毯外点击被吞）。
        # 这里统一复位（sim 侧由 rug.lift/_cleanup 负责 release）。
        if self._view is not None:
            try:
                self._view._grabbed = False
                self._view._handle_grab = False
            except Exception:
                pass
            _set_cursor("arrow")
        self._handle = None
        self._update_handles()
        if self._panel is not None:
            self._panel.orderOut_(None)
        self._sim = None
        self._shown = False
        self._elem = None
        self._elem2 = None
        self._rim_elem = None
        self._skirt_elem = None
        self._color_src = None
        self._rim_uv_src = None
        self._rim_color_src = None
        self._vbuf = self._nbuf = self._ubuf = self._sbuf = self._cbuf = None
        self._vrel = None
        self._rim_v = self._rim_n = self._rim_uv = self._rim_c = None
        self._scbuf = None
        self._bloop = self._bloop_next = self._bloop_prev = None
        self._eband = self._relief = None
        self._last_cursor = None
        self._last_dist = None
        self._gate_dirty = False
        _log("RUG-INFO", "overlay hide")

    def is_shown(self) -> bool:
        return bool(self._shown)

    @property
    def menu_provider(self):
        """毯上右键菜单提供者：(x, y) 主屏本地点 → NSMenu；None = 不弹菜单。"""
        return self._menu_provider

    def set_edit_mode(self, on: bool) -> None:
        """进入/退出「编辑毯子」模式：显示四角缩放柄 + 旋转柄。

        进入时把毯子规整化成「当前摆放」的矩形（自由拖动后的姿态会被拉正），
        这样手柄与毯子边角一一对应；之后：拖角柄 = 等比缩放（中心不动），
        拖旋转柄 = 绕中心转角度，拖毯身 = 平移（原有拖动）。
        """
        on = bool(on)
        if on and not self._shown:
            return
        self._edit_mode = on
        self._handle = None
        sim = self._sim
        if sim is None:
            return
        if on:
            if self._placement is None:
                cx, cy, w, h = sim.placement()
                self._placement = (cx, cy, w, h, 0.0)
            cx, cy, w, h, ang = self._placement
            try:
                sim.set_placement((cx, cy), w, h, ang, zero_velocity=True)
            except Exception as exc:
                _log("RUG-WARN", f"进入编辑模式时规整化失败：{exc}")
            self._build_handles()
        self._update_handles()
        _log("RUG-INFO", f"编辑模式 {'开' if on else '关'}"
                         + (f"（毯子 {self._placement[2]:.0f}x{self._placement[3]:.0f} "
                            f"{math.degrees(self._placement[4]):.1f}°）" if on else ""))

    @property
    def edit_mode(self) -> bool:
        return bool(self._edit_mode)

    @property
    def placement(self):
        """当前摆放 (cx, cy, w, h, angle_rad)；未铺上时为 None。"""
        if self._sim is not None:
            cx, cy, w, h = self._sim.placement()
            ang = self._placement[4] if self._placement is not None else 0.0
            return (cx, cy, w, h, float(ang))
        return self._placement

    def apply_placement(self, placement, to_sim: bool = True) -> None:
        """把毯子摆到指定 (cx, cy, w, h, angle)（「摆放回来」/ 重置 / 换贴图用）。

        to_sim=True 时同时把 sim 刚性摆到该位置（并清零速度 → 直接落定）；
        to_sim=False 只更新摆放状态（例如紧接飞入动画的 lay_down，不能提前把毯子钉住）。
        """
        cx, cy, w, h, ang = placement
        self._placement = (float(cx), float(cy), float(max(w, HANDLE_MIN_PT)),
                           float(max(h, HANDLE_MIN_PT)), float(ang))
        if to_sim and self._sim is not None and self._shown:
            try:
                self._sim.set_placement((cx, cy), self._placement[2], self._placement[3],
                                        self._placement[4], zero_velocity=True)
            except Exception as exc:
                _log("RUG-WARN", f"apply_placement 失败：{exc}")
        self._update_handles()

    def set_shadow_visible(self, on: bool) -> None:
        """接触阴影开关（设置窗用）：直接切阴影节点（贴边投影裙）可见性。"""
        if self._shadow_node is not None:
            try:
                self._shadow_node.setHidden_(not bool(on))
            except Exception:
                pass

    # ---- 编辑模式内部：手柄手势（由 RugRootView 的鼠标事件驱动） ----
    def _begin_handle(self, index: int, x: float, y: float) -> None:
        """开始手柄手势：记住起始摆放与起始指针位置。"""
        if self._placement is None:
            if self._sim is None:
                return
            cx, cy, w, h = self._sim.placement()
            self._placement = (cx, cy, w, h, 0.0)
        self._handle = (int(index), tuple(self._placement), (float(x), float(y)))
        self._last_edit_angle = float(self._placement[4])

    def _drag_handle(self, x: float, y: float) -> None:
        """手柄拖动：0..3 角柄 = 绕中心等比缩放；4 旋转柄 = 绕中心转角度。"""
        if self._handle is None or self._sim is None or self._placement is None:
            return
        index, start, (sx, sy) = self._handle
        cx, cy, w0, h0, ang0 = start
        if index == 4:                                    # 旋转
            a0 = math.atan2(sy - cy, sx - cx)
            a1 = math.atan2(y - cy, x - cx)
            ang = ang0 + (a1 - a0)
            self._placement = (cx, cy, w0, h0, ang)
            self._sim.set_placement((cx, cy), w0, h0, ang,
                                    prev_angle_rad=self._last_edit_angle)
            self._last_edit_angle = ang
        else:                                             # 等比缩放（中心不动）
            r0 = math.hypot(sx - cx, sy - cy)
            if r0 < 8.0:
                return
            scale = math.hypot(x - cx, y - cy) / r0
            nw = min(max(w0 * scale, HANDLE_MIN_PT), max(self._w, 1.0) * HANDLE_MAX_FRAC)
            nh = min(max(h0 * scale, HANDLE_MIN_PT), max(self._h, 1.0) * HANDLE_MAX_FRAC)
            self._placement = (cx, cy, nw, nh, ang0)
            self._sim.set_placement((cx, cy), nw, nh, ang0,
                                    prev_angle_rad=self._last_edit_angle)
            self._last_edit_angle = ang0
        self._update_handles()

    def _end_handle(self) -> None:
        """手柄手势结束：唤醒布料做一次收尾（形状已是规整矩形）。"""
        self._handle = None
        if self._sim is not None:
            self._sim.wake()

    def _refresh_placement_from_sim(self) -> None:
        """编辑模式下毯身被拖动后，让摆放中心跟随（尺寸/角度沿用已记录值）。"""
        if self._sim is None or self._placement is None or self._handle is not None:
            return
        cx, cy, w, h = self._sim.placement()
        self._placement = (cx, cy, w, h, self._placement[4])

    # ---- 编辑模式内部：手柄几何/命中 ----
    def _handle_points(self):
        """当前四个角柄 + 一个旋转柄的位置（sim 坐标 top-left, y-down）。"""
        if self._placement is None:
            return []
        cx, cy, w, h, ang = self._placement
        ca, sa = math.cos(ang), math.sin(ang)
        hw, hh = w * 0.5, h * 0.5
        pts = []
        for sx_, sy_ in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            lx, ly = sx_ * hw, sy_ * hh
            pts.append((cx + ca * lx - sa * ly, cy + sa * lx + ca * ly))
        lx, ly = 0.0, -hh - ROTATE_OFFSET_PT        # 顶边外上方
        pts.append((cx + ca * lx - sa * ly, cy + sa * lx + ca * ly))
        return pts

    def _handle_distance(self, x: float, y: float) -> float:
        """(x, y) 到最近手柄的距离（无手柄时给一个很大的值）。"""
        best = 1e9
        for px, py in self._handle_points():
            d = math.hypot(px - x, py - y)
            if d < best:
                best = d
        return best

    def _handle_at(self, x: float, y: float):
        """命中哪个手柄：返回 0..3（四角）或 4（旋转柄）；未命中 None。"""
        pts = self._handle_points()
        best = None
        best_d = HANDLE_HIT_R_PT
        for i, (px, py) in enumerate(pts):
            d = math.hypot(px - x, py - y)
            if d <= best_d:
                best_d = d
                best = i
        return best

    def _build_handles(self) -> None:
        """建手柄节点（一次）：角柄=白底黑框方块，旋转柄=蓝底白框方块 + 一根连线。"""
        if self._handle_nodes or self._scene is None:
            return
        import SceneKit as _SK

        def quad(size, fill, border, z=HANDLE_Z):
            node = _SK.SCNNode.node()
            inner = _SK.SCNMaterial.material()
            inner.diffuse().setContents_(fill)
            inner.setLightingModelName_(_SK.SCNLightingModelConstant)
            outer = _SK.SCNMaterial.material()
            outer.diffuse().setContents_(border)
            outer.setLightingModelName_(_SK.SCNLightingModelConstant)
            geo_in = _SK.SCNPlane.planeWithWidth_height_(size, size)
            geo_in.setMaterials_([inner])
            n_in = _SK.SCNNode.nodeWithGeometry_(geo_in)
            geo_out = _SK.SCNPlane.planeWithWidth_height_(size + 4.0, size + 4.0)
            geo_out.setMaterials_([outer])
            n_out = _SK.SCNNode.nodeWithGeometry_(geo_out)
            n_out.setPosition_((0.0, 0.0, -0.2))
            node.addChildNode_(n_out)
            node.addChildNode_(n_in)
            # SCNPlane 默认在 x-y 平面朝 +z：本场景相机沿 -y 俯视 → 绕 x 转 90° 立起来
            node.setEulerAngles_((-math.pi / 2.0, 0.0, 0.0))
            node.setRenderingOrder_(100)
            return node

        white = NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 1.0, 1.0, 0.97)
        dark = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.10, 0.10, 0.10, 0.85)
        blue = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.16, 0.42, 0.94, 0.97)
        for _ in range(4):
            self._handle_nodes.append(quad(HANDLE_SIZE_PT, white, dark))
        self._handle_nodes.append(quad(ROTATE_SIZE_PT, blue, white))
        for n in self._handle_nodes:
            self._scene.rootNode().addChildNode_(n)

    def _update_handles(self) -> None:
        """按当前摆放摆放/显示手柄（编辑模式外全部隐藏）。"""
        if not self._handle_nodes:
            return
        if not self._edit_mode or not self._shown or self._placement is None:
            for n in self._handle_nodes:
                n.setHidden_(True)
            return
        pts = self._handle_points()
        if len(pts) < 5:
            for n in self._handle_nodes:
                n.setHidden_(True)
            return
        for i in range(5):
            px, py = pts[i]
            self._handle_nodes[i].setPosition_((px, HANDLE_Z, py))
            self._handle_nodes[i].setHidden_(False)

    @property
    def sim(self):
        return self._sim

    # ---- 内部：GUI 构建（一次） ----
    def _build_gui(self, frame) -> None:
        w, h = float(frame.size.width), float(frame.size.height)
        self._w, self._h = w, h
        panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            frame,
            NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel,
            NSBackingStoreBuffered, False)
        panel.setLevel_(icon_level_base() + 8)
        # 透明三件套（contract.md §4）
        panel.setOpaque_(False)
        panel.setBackgroundColor_(NSColor.clearColor())
        panel.setHasShadow_(False)
        panel.setHidesOnDeactivate_(False)      # NSPanel 防失活自动隐藏
        panel.setBecomesKeyOnlyIfNeeded_(True)  # accessory 不抢焦点
        panel.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorStationary
            | NSWindowCollectionBehaviorIgnoresCycle)
        panel.setIgnoresMouseEvents_(False)     # 初始收鼠标；穿透由闸门动态切换
        panel.setAcceptsMouseMovedEvents_(True)

        view = RugRootView.alloc().initWithFrame_(frame)
        view._owner = self
        view._grabbed = False
        view._handle_grab = False
        panel.setContentView_(view)

        scn = SceneKit.SCNView.alloc().initWithFrame_(NSMakeRect(0.0, 0.0, w, h))
        scn.setBackgroundColor_(NSColor.clearColor())
        scn.setOpaque_(False)
        scn.setAutoresizingMask_(15)  # width|height，跟随根视图

        scene = SceneKit.SCNScene.scene()

        # ---- 布面材质：顶/背两张贴图，按面朝向分渲染（cullMode）----
        # 环绕方向（_prepare_geometry 的 index 表）：毯面朝上（平铺）时，三角形从相机
        # 看是「背向面」→ 正面材质 cullMode=Front（只画背向面 = 看到毯面）；
        # 翻折露底时三角面朝向反转 → 背面材质 cullMode=Back（只画面向面 = 看到毯底）。
        # 背面贴图默认用同目录 -back.png（scripts/make_back_texture.py 生成）；
        # 不存在则回退正面贴图（翻折处看到镜像图案，不至于缺图）。
        tex, crop = _load_rug_image(self._texture_path)
        if tex is None:
            _log("RUG-WARN", f"贴图加载失败，回退纯色：{self._texture_path}")
        _log("RUG-INFO", f"贴图 {os.path.basename(self._texture_path)} 边距裁切 "
                         f"(上,左,下,右)={crop}（0 = 无留白；见 §12.17 白边根因）")
        back_path = self._texture_path[:-4] + "-back.png" \
            if self._texture_path.lower().endswith(".png") else self._texture_path
        if back_path != self._texture_path:
            tex_back, _ = _load_rug_image(back_path, crop=crop)   # 背图与正图同尺寸同裁切
        else:
            tex_back = None
        if tex_back is None:
            tex_back = tex
        self._back_texture_path = back_path if tex_back is not tex else self._texture_path

        def _cloth_material(image, cull_mode):
            m = SceneKit.SCNMaterial.material()
            if image is not None:
                m.diffuse().setContents_(image)
            else:
                m.diffuse().setContents_(
                    NSColor.colorWithCalibratedRed_green_blue_alpha_(0.72, 0.12, 0.10, 1.0))
            m.setLightingModelName_(SceneKit.SCNLightingModelBlinn)
            m.setDoubleSided_(False)      # 双面已由两张单面材质代替（否则会互相打架）
            m.setCullMode_(cull_mode)
            m.specular().setContents_(NSColor.blackColor())  # 布料不要高光
            return m

        self._cloth_mat = _cloth_material(tex, SceneKit.SCNCullModeFront)       # 正面（兼容旧断言名）
        self._cloth_mat_back = _cloth_material(tex_back, SceneKit.SCNCullModeBack)
        # 滚边材质：正面材质副本 + 双面（滚边是薄条，双面可免去环绕方向的脆弱依赖）
        rim_mat = _cloth_material(tex, SceneKit.SCNCullModeFront)
        rim_mat.setDoubleSided_(True)   # SceneKit 的双面语义 = 不剔除背面（本机无 CullModeNone 常量）
        self._rim_mat = rim_mat
        _log("RUG-INFO", f"布面材质：正面={os.path.basename(self._texture_path)} "
                         f"背面={os.path.basename(self._back_texture_path)}")

        cloth_node = SceneKit.SCNNode.node()
        cloth_node.setCastsShadow_(False)
        scene.rootNode().addChildNode_(cloth_node)

        # v6 滚边节点（几何逐帧重建；材质=正面材质副本，顶点色带滚边暗化梯度）
        rim_node = SceneKit.SCNNode.node()
        rim_node.setCastsShadow_(False)
        rim_node.setRenderingOrder_(1)      # 滚边压在布面之后画（贴边处遮挡关系更稳）
        scene.rootNode().addChildNode_(rim_node)
        self._rim_node = rim_node

        # ---- 接触阴影材质：暗色 Constant，透明度交给顶点色 ----
        shadow_mat = SceneKit.SCNMaterial.material()
        shadow_mat.diffuse().setContents_(
            NSColor.colorWithCalibratedRed_green_blue_alpha_(0.0, 0.0, 0.0, 1.0))
        shadow_mat.setLightingModelName_(SceneKit.SCNLightingModelConstant)
        shadow_mat.setDoubleSided_(True)
        self._shadow_mat = shadow_mat
        shadow_node = SceneKit.SCNNode.node()
        shadow_node.setCastsShadow_(False)
        shadow_node.setRenderingOrder_(-1)
        scene.rootNode().addChildNode_(shadow_node)

        # ---- 正交相机：俯视，1 场景单位 = 1pt，视野 = 屏宽高 ----
        cam = SceneKit.SCNCamera.camera()
        cam.setUsesOrthographicProjection_(True)  # usesOrthographicProjection（contract.md §4）
        cam.setProjectionDirection_(SceneKit.SCNCameraProjectionDirectionVertical)
        cam.setOrthographicScale_(h / 2.0)        # 垂直半高 → 竖向 [0,h]，横向恰为 [0,w]
        cam.setZNear_(1.0)
        cam.setZFar_(4000.0)
        cam_node = SceneKit.SCNNode.node()
        cam_node.setCamera_(cam)
        cam_node.setPosition_((w / 2.0, CAM_HEIGHT, h / 2.0))
        # 绕 x 轴 -90°：沿 -y 俯视；相机局部 +y（屏幕上方向）= 场景 -z。
        # ⚠️ 真机踩坑（2026-10-08）：SCNNode.rotation 是 SCNVector4 **(x, y, z, 角度弧度)**，
        #    角度在**第四位**。旧写法 (-π/2, 0, 0, 1) 被当成「轴=(-π/2,0,0)、角=1 rad」→
        #    相机实际只俯了 57.3°，画面被斜视压扁（cos32.7°≈0.84）并整体下移——渲染出的毯子
        #    与交互用的 sim 坐标不在同一坐标系（用户报障「移动效果很抽象」的真根因）。
        #    正确写法：轴=(1,0,0)、角=-π/2。tests/test_overlay.py 有 projectPoint 回归锁。
        cam_node.setRotation_((1.0, 0.0, 0.0, -math.pi / 2.0))
        scene.rootNode().addChildNode_(cam_node)

        # ---- 灯光：环境光 + 平行光（Blinn 出立体感） ----
        # v6：环境光占比下调、平行光补足（平铺面总亮不变）→ 明暗对比范围翻倍。
        # 平铺面：AMBIENT_WHITE + SUN_WHITE×sin(仰角) ≈ 0.42 + 0.525×0.866 ≈ 0.875（与旧版一致）。
        amb = SceneKit.SCNLight.light()
        amb.setType_(SceneKit.SCNLightTypeAmbient)
        amb.setColor_(NSColor.colorWithCalibratedWhite_alpha_(AMBIENT_WHITE, 1.0))
        amb_node = SceneKit.SCNNode.node()
        amb_node.setLight_(amb)
        scene.rootNode().addChildNode_(amb_node)

        sun = SceneKit.SCNLight.light()
        sun.setType_(SceneKit.SCNLightTypeDirectional)
        sun.setColor_(NSColor.colorWithCalibratedWhite_alpha_(SUN_WHITE, 1.0))
        sun_node = SceneKit.SCNNode.node()
        sun_node.setLight_(sun)
        # 斜射方向：用四元数把平行光旋到 SUN_TO_LIGHT（不碰欧拉角约定）；建灯后立刻用
        # SceneKit 自己的 convertVector:toNode: 实测核对（约定不符就翻转并记日志）。
        sun_node.setOrientation_(orientation_for_to_light(SUN_TO_LIGHT))
        scene.rootNode().addChildNode_(sun_node)
        dir_m, lean_m = self._sun_shadow_basis(sun_node, scene.rootNode())
        if dir_m[0] * SUN_TO_LIGHT[0] + dir_m[1] * SUN_TO_LIGHT[2] > 0.0:
            # 实测传播方向与目标同侧（= 差 180°）→ 四元数共轭重设
            q = orientation_for_to_light(SUN_TO_LIGHT)
            sun_node.setOrientation_((q[0], -q[1], -q[2], -q[3]))
            dir_m, lean_m = self._sun_shadow_basis(sun_node, scene.rootNode())
            _log("RUG-WARN", f"平行光朝向共轭修正后 dir=({dir_m[0]:.2f},{dir_m[1]:.2f})")
        self._shadow_dir, self._shadow_lean = dir_m, lean_m
        dir_t, lean_t = _shadow_basis_from(SUN_TO_LIGHT)
        _log("RUG-INFO", f"灯光：环境 {AMBIENT_WHITE:.2f} + 平行 {SUN_WHITE:.2f}；"
                         f"阴影方向=({dir_m[0]:.2f},{dir_m[1]:.2f}) lean={lean_m:.2f}"
                         f"（纯函数期望 ({dir_t[0]:.2f},{dir_t[1]:.2f}) lean={lean_t:.2f}）")

        scn.setScene_(scene)
        view.addSubview_(scn)

        self._panel = panel
        self._view = view
        self._scn = scn
        self._scene = scene
        self._cloth_node = cloth_node
        self._shadow_node = shadow_node
        self._built_gui = True
        _log("RUG-INFO", f"overlay GUI 构建：{w:.0f}x{h:.0f}pt "
                         f"level={icon_level_base() + 8}（base={icon_level_base()}）")

    # ---- 内部：几何缓冲（每次 show 重建） ----
    def _prepare_geometry(self) -> None:
        sim = self._sim
        V = sim.vertices()
        n = int(V.shape[0])
        uv = np.asarray(sim.uv(), dtype=np.float64)
        # 行主序拓扑推断：v 在行内恒定、跨行递增 → rows = 1 + diff(v)>0 的次数
        rows = 1 + int(np.count_nonzero(np.diff(uv[:, 1]) > 0.0))
        cols = max(2, n // rows)
        if rows * cols != n or rows < 2:
            _log("RUG-WARN", f"uv 行列推断异常（n={n}, rows={rows}），回退 56 行")
            rows, cols = 56, max(2, n // 56)
        self._rows, self._cols = rows, cols
        # 法线梯度用的网格间距：优先取 sim 实际尺寸（设计变更 A：毯子小于屏幕，
        # 按屏幕宽高算会让法线明暗失真）；getattr 兜底保持 overlay 不依赖 cloth 模块
        try:
            pw = float(getattr(sim, "width_pt", self._w))
            ph = float(getattr(sim, "height_pt", self._h))
        except Exception:
            pw, ph = self._w, self._h
        if not (pw > 0.0 and ph > 0.0):  # 含 NaN
            pw, ph = self._w, self._h
        self._dx = pw / float(cols - 1)
        self._dy = ph / float(rows - 1)

        idx = np.ascontiguousarray(sim.indices(), dtype=np.uint32)
        idata = NSData.dataWithBytes_length_(idx.tobytes(), idx.nbytes)
        self._elem = SceneKit.SCNGeometryElement.alloc().initWithData_primitiveType_primitiveCount_indicesChannelCount_interleavedIndicesChannels_bytesPerIndex_(
            idata, SceneKit.SCNGeometryPrimitiveTypeTriangles, int(idx.size) // 3,
            1, False, 4)
        # 第二个 element：同索引数据（顶/背两张材质各自带一套 element，避免共享实例）
        self._elem2 = SceneKit.SCNGeometryElement.alloc().initWithData_primitiveType_primitiveCount_indicesChannelCount_interleavedIndicesChannels_bytesPerIndex_(
            idata, SceneKit.SCNGeometryPrimitiveTypeTriangles, int(idx.size) // 3,
            1, False, 4)
        self._tri = idx.reshape(-1, 3).astype(np.int64)   # 法线计算用三角索引表

        self._vbuf = np.empty((n, 3), np.float32)
        self._vrel = np.empty((n, 3), np.float32)      # 法线用顶点（= vb + 微浮雕）
        self._nbuf = np.empty((n, 3), np.float32)
        self._fnbuf = np.empty((self._tri.shape[0], 3), np.float32)  # 面法线工作区
        self._ubuf = np.empty((n, 2), np.float32)
        self._ubuf[:, 0] = uv[:, 0]
        self._ubuf[:, 1] = 1.0 - uv[:, 1]  # SceneKit v 轴朝上，喂贴图前 1-v（contract.md §4）

        # ---- v6：静态顶点色（边缘暗化带）与静止微浮雕 ----
        rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        cells = np.minimum(np.minimum(rr, rows - 1 - rr),
                           np.minimum(cc, cols - 1 - cc)).astype(np.float32)
        band = np.clip(cells / EDGE_BAND_CELLS, 0.0, 1.0)
        self._eband = (EDGE_BAND_DARK + (1.0 - EDGE_BAND_DARK) * band).ravel()
        self._cbuf = np.empty((n, 4), np.float32)
        self._cbuf[:, 3] = 1.0
        # 微浮雕：网格空间静态低频场（两档波长）。每个分量 = 行正弦 × 列正弦，两个不同取向的
        # 波叠加 → 无格点感；波长 ≥ 7 格（≥90pt，远大于格距）→ 大而柔的起伏，不是碎纹。
        # 振幅按「倾角」标定：RMS 倾角约 4°，实测 relief_dlum≈3（look_demo 厚度指标）。
        fld = np.zeros_like(rr, dtype=np.float64)
        for amp, wl in zip(RELIEF_AMP_PT, RELIEF_CELLS):
            k = 2.0 * np.pi / float(wl)
            fld += amp * (np.sin(k * (rr + 0.35 * cc) + 0.7)
                          * np.cos(k * (cc - 0.42 * rr) + 2.1))
        self._relief = fld.ravel().astype(np.float32)

        # ---- v6：边界环（滚边 + 接触阴影裙共用；拓扑静态） ----
        # 环序：顶行左→右、右列上→下、底行右→左、左列下→上（顺时针，长度 L=2(cols+rows)-4）
        top = np.arange(cols, dtype=np.int64)
        right = np.arange(1, rows, dtype=np.int64) * cols + (cols - 1)
        bottom = np.arange(cols - 2, -1, -1, dtype=np.int64) + (rows - 1) * cols
        left = np.arange(rows - 2, 0, -1, dtype=np.int64) * cols
        loop = np.concatenate([top, right, bottom, left])
        self._bloop = loop
        self._bloop_next = np.roll(loop, -1)
        self._bloop_prev = np.roll(loop, 1)
        L = int(loop.size)
        rings = len(EDGE_ROLL_ANGLES)
        sk = len(SHADOW_SKIRT_PT)
        self._rim_rings = rings
        self._skirt_rings = sk
        # 布局：index = k*rings + j（k=环上第 k 点、j=第 j 环）→ `buf[j::rings]` 即第 j 环，
        # 逐环整体向量化写入（滚边与阴影裙同布局）。
        self._rim_v = np.empty((L * rings, 3), np.float32)
        self._rim_n = np.empty((L * rings, 3), np.float32)
        self._rim_uv = np.repeat(self._ubuf[loop], rings, axis=0)     # UV = 边界 UV（贴图边缘）
        self._rim_c = np.empty((L * rings, 4), np.float32)
        for j, dark in enumerate(EDGE_ROLL_DARK):
            self._rim_c[j::rings, 0] = dark
            self._rim_c[j::rings, 1] = dark
            self._rim_c[j::rings, 2] = dark
            self._rim_c[j::rings, 3] = 1.0

        def _strip_indices(n_rings: int) -> np.ndarray:
            """环带三角索引（拓扑静态）：每段 × 每环带 2 三角形。"""
            k = np.arange(L, dtype=np.int64)
            kn = (k + 1) % L
            kk = np.repeat(k, n_rings - 1)
            knn = np.repeat(kn, n_rings - 1)
            jj = np.tile(np.arange(n_rings - 1, dtype=np.int64), L)
            a0 = kk * n_rings + jj
            a1 = a0 + 1
            b0 = knn * n_rings + jj
            b1 = b0 + 1
            tri = np.empty((a0.size * 2, 3), dtype=np.uint32)
            tri[0::2, 0], tri[0::2, 1], tri[0::2, 2] = a0, b0, a1
            tri[1::2, 0], tri[1::2, 1], tri[1::2, 2] = b0, b1, a1
            return tri.ravel()

        ridx = _strip_indices(rings)
        self._rim_elem = SceneKit.SCNGeometryElement.alloc().initWithData_primitiveType_primitiveCount_indicesChannelCount_interleavedIndicesChannels_bytesPerIndex_(
            NSData.dataWithBytes_length_(ridx.tobytes(), ridx.nbytes),
            SceneKit.SCNGeometryPrimitiveTypeTriangles, int(ridx.size) // 3, 1, False, 4)
        sidx = _strip_indices(sk)
        self._skirt_elem = SceneKit.SCNGeometryElement.alloc().initWithData_primitiveType_primitiveCount_indicesChannelCount_interleavedIndicesChannels_bytesPerIndex_(
            NSData.dataWithBytes_length_(sidx.tobytes(), sidx.nbytes),
            SceneKit.SCNGeometryPrimitiveTypeTriangles, int(sidx.size) // 3, 1, False, 4)
        self._sbuf = np.empty((L * sk, 3), np.float32)
        self._scbuf = np.empty((L * sk, 4), np.float32)
        self._scbuf[:, :3] = 0.0
        self._rim_color_src = _src(self._rim_c, SceneKit.SCNGeometrySourceSemanticColor, 4)
        self._rim_uv_src = _src(self._rim_uv, SceneKit.SCNGeometrySourceSemanticTexcoord, 2)
        _log("RUG-INFO", f"v6 厚度几何：边界环 {L} 点、滚边 {rings} 环（厚 "
                         f"{CLOTH_THICKNESS_PT:.0f}pt）、阴影裙 {sk} 环；"
                         f"浮雕振幅 {RELIEF_AMP_PT} pt / 波长 {RELIEF_CELLS} 格")

    # ---- 内部：逐帧几何重建 ----
    def _rebuild_geometry(self) -> None:
        sim = self._sim
        V = sim.vertices()  # (N,3) f32：x, y_sim, z_height
        n = V.shape[0]
        vb = self._vbuf
        # v6：静止微浮雕 = **只扰法线**的 bump 场（不动顶点深度）。权重在网格空间算：
        #   a) 离地高度（抬起/翻折/图标隆起上不加）；
        #   b) 曲率（波峰/折脊上不加——那里加浮雕等于「锐化」，法线被成倍放大）；
        #   c) 按格距比缩放振幅（压缩区同一材料场被挤短，等比缩放后「倾角」与位姿无关）。
        # 只扰法线的原因（实测）：浮雕曾直接加在渲染高度上，拖动场景里压缩区在屏幕上
        # 自重叠，±6pt 的深度差会改变「哪一层通过深度测试」→ 出现碎斑。改成法线扰动后
        # 深度仍由物理决定，观感回到干净的表面起伏。
        fl = self._sim_array(sim, "floor_heights", n, 0.0)
        rows, cols = self._rows, self._cols
        zz = V[:, 2].reshape(rows, cols)
        w_relief = np.clip(1.0 - (zz - fl.reshape(rows, cols)) / RELIEF_FADE_PT, 0.0, 1.0)
        lap = np.zeros_like(zz)
        lap[1:-1, 1:-1] = (zz[:-2, 1:-1] + zz[2:, 1:-1] + zz[1:-1, :-2] + zz[1:-1, 2:]
                           - 4.0 * zz[1:-1, 1:-1])
        w_relief *= 1.0 / (1.0 + (np.abs(lap) / RELIEF_CURV_SOFT) ** 2)
        px = V[:, 0].reshape(rows, cols)
        py = V[:, 1].reshape(rows, cols)
        rxc = np.ones_like(px)
        ryc = np.ones_like(py)
        rxc[:, 1:-1] = 0.5 * (np.abs(px[:, 2:] - px[:, 1:-1])
                              + np.abs(px[:, 1:-1] - px[:, :-2])) / max(self._dx, 1e-6)
        ryc[1:-1, :] = 0.5 * (np.abs(py[2:, :] - py[1:-1, :])
                              + np.abs(py[1:-1, :] - py[:-2, :])) / max(self._dy, 1e-6)
        # 边界行列沿用相邻内圈值（否则角点/边缘恒为 1.0 → 均匀压缩时它们漏过缩放）
        rxc[:, 0] = rxc[:, 1]
        rxc[:, -1] = rxc[:, -2]
        ryc[0, :] = ryc[1, :]
        ryc[-1, :] = ryc[-2, :]
        # 压缩区按格距比缩放振幅：涟漪的**倾角**（=振幅/波长，二者同比例缩短）与位姿无关
        # → 浮雕的明暗贡献恒定（实测拖动场景曾达静置态的 8 倍并出现 p99≈86 的碎斑，
        # 按格距比缩放后回到同一量级）。下限保底防抖。
        w_relief *= np.clip(np.minimum(rxc, ryc), RELIEF_MIN_CELL, 1.0)
        w_relief = w_relief.ravel()
        self._w_relief = w_relief       # 供回归测试读（密集折区/压缩区的浮雕权重）
        # v6.1：深度偏置修复折叠层级问题——sim_z 越高（折起），scene_z 越小（深度测试中更近）。
        # v6.5：系数 0.15 → **0.90**。为什么可以这么大：正交俯视 + 毯子平铺，**只有同一屏幕
        # 位置的不同层会互相遮挡**（不同 (x,y) 的粒子投影到不同像素，深度顺序无关），所以
        # 高度差近乎 1:1 映射到深度是安全的，而且更稳——折层只要高 1pt 就必定画在上层。
        # 0.15 时两层贴到一起（互穿残余）深度差≈0 → 逐像素锯齿"撕口"（用户截图里的穿模）。
        # 场景坐标：scene_x=sim_x, scene_y=sim_z（朝相机）, scene_z=sim_y+bias（深度测试轴）。
        depth_bias = -0.90 * V[:, 2]   # sim_z 越高，bias 越负 → scene_z 越小 → 越近
        np.copyto(vb, np.column_stack((V[:, 0], V[:, 2], V[:, 1] + depth_bias)))
        vr = self._vrel
        np.copyto(vr, vb)
        vr[:, 1] += self._relief * w_relief                          # 仅法线用的起伏

        # 边界环切向/外法线（滚边、阴影裙、边界法线过渡共用）
        loop = self._bloop
        p = vb[loop]
        tvec = vb[self._bloop_next] - vb[self._bloop_prev]
        th = tvec[:, [0, 2]]
        ln = np.sqrt((th * th).sum(axis=1))
        np.maximum(ln, 1e-9, out=ln)
        th /= ln[:, None]
        out = np.empty_like(th)
        out[:, 0] = th[:, 1]          # 外法线 = 切向转 -90°（sim y 向下；环序为顺时针，
        out[:, 1] = -th[:, 0]         # 该符号把法线指向毯外——test_overlay 有平铺期回归锁）

        # 法线：从三角网格算（真 3D——翻折区需要真实朝向，不能假设 z 单值）
        # v6：朝下法线只保留 NORMAL_FLIP_KEEP 比例（原来全翻正 → 折脊两侧被照得一样亮、
        # 折面没有「迎光/背光」差 → 看起来像折纸；保留比例后远端折面自然变暗）。
        tri = self._tri
        fn = self._fnbuf
        e1 = vr[tri[:, 1]] - vr[tri[:, 0]]
        e2 = vr[tri[:, 2]] - vr[tri[:, 0]]
        fn[:, 0] = e1[:, 1] * e2[:, 2] - e1[:, 2] * e2[:, 1]
        fn[:, 1] = e1[:, 2] * e2[:, 0] - e1[:, 0] * e2[:, 2]
        fn[:, 2] = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        nb = self._nbuf
        nb.fill(0.0)
        tri_flat = tri.ravel()
        for c in range(3):
            nb[:, c] = np.bincount(tri_flat, weights=np.repeat(fn[:, c], 3),
                                   minlength=nb.shape[0])
        neg = nb[:, 1] < 0.0
        if neg.any():
            nb[neg, 1] *= -NORMAL_FLIP_KEEP
        # 边界环法线朝滚边倾斜「接缝角」→ 顶面到滚边是圆滑过渡（与滚边首环法线一致）
        seam = EDGE_SEAM_FRAC * EDGE_ROLL_ANGLES[1]
        s1 = math.sin(math.radians(seam))
        c1 = math.cos(math.radians(seam))
        nb[loop, 0] += out[:, 0] * s1
        nb[loop, 1] += c1 - 1.0
        nb[loop, 2] += out[:, 1] * s1
        length = np.sqrt(nb[:, 0] ** 2 + nb[:, 1] ** 2 + nb[:, 2] ** 2)
        bad = length < 1e-9
        np.maximum(length, 1e-9, out=length)
        nb /= length[:, None]
        if bad.any():
            nb[bad] = (0.0, 1.0, 0.0)                  # 退化三角（折叠贴合处）给恒定向上的法线

        # 顶点色 = 静态边缘暗化带 × 折层遮蔽（sim.ao()）——厚织物贴边与折缝的接触阴影
        ao = self._sim_array(sim, "ao", n, 0.0)
        dark = self._eband * (1.0 - AO_STRENGTH * np.clip(ao, 0.0, 1.0))
        cb = self._cbuf
        cb[:, 0] = dark
        cb[:, 1] = dark
        cb[:, 2] = dark
        cb[:, 3] = 1.0
        self._color_src = _src(cb, SceneKit.SCNGeometrySourceSemanticColor, 4)

        geo = SceneKit.SCNGeometry.geometryWithSources_elements_(
            [_src(vb, SceneKit.SCNGeometrySourceSemanticVertex, 3),
             _src(nb, SceneKit.SCNGeometrySourceSemanticNormal, 3),
             _src(self._ubuf, SceneKit.SCNGeometrySourceSemanticTexcoord, 2),
             self._color_src],
            [self._elem, self._elem2])
        # 关键（真机踩坑 2026-10-08）：geometryWithSources 造出的几何**自带空材质列表**，
        # macOS 26 上无材质的几何**完全不绘制**（不报错、不警告，画面全透明）。
        # 材质在 _build_gui 里建好，这里必须逐帧挂上；elements[i] ↔ materials[i]：
        # [0] 背面材质 cullMode=Back（翻折露底时可见）、[1] 正面材质 cullMode=Front。
        if self._cloth_mat is not None:
            mats = [self._cloth_mat_back, self._cloth_mat] \
                if getattr(self, "_cloth_mat_back", None) is not None else [self._cloth_mat]
            geo.setMaterials_(mats)
        self._cloth_node.setGeometry_(geo)

        # ---- v6：滚边（沿边界环向外+向下卷出的圆角厚边） ----
        # 剖面：横向 R·sinφ、下坠 R(1-cosφ)；φ=90° 即「赤道」（最外缘，与桌面接触处）。
        # 顶视图下这条带子有真实宽度（=R）且自带受光→背光梯度 → 材料厚度肉眼可见。
        rv, rn = self._rim_v, self._rim_n
        R = CLOTH_THICKNESS_PT
        for j, ang in enumerate(EDGE_ROLL_ANGLES):
            s = math.sin(math.radians(ang))
            c = math.cos(math.radians(ang))
            dv = rv[j::self._rim_rings]
            dv[:, 0] = p[:, 0] + out[:, 0] * (R * s)
            dv[:, 1] = p[:, 1] - R * (1.0 - c)
            dv[:, 2] = p[:, 2] + out[:, 1] * (R * s)
            # 首环（角 0，几何贴在布面边缘、用来封住缝）法线取接缝角，与布面边缘法线一致
            na = ang if ang > 0.0 else EDGE_SEAM_FRAC * EDGE_ROLL_ANGLES[1]
            sa = math.sin(math.radians(na))
            ca = math.cos(math.radians(na))
            dn = rn[j::self._rim_rings]
            dn[:, 0] = out[:, 0] * sa
            dn[:, 1] = ca
            dn[:, 2] = out[:, 1] * sa
        rgeo = SceneKit.SCNGeometry.geometryWithSources_elements_(
            [_src(rv, SceneKit.SCNGeometrySourceSemanticVertex, 3),
             _src(rn, SceneKit.SCNGeometrySourceSemanticNormal, 3),
             self._rim_uv_src, self._rim_color_src],
            [self._rim_elem])
        if self._rim_mat is not None:
            rgeo.setMaterials_([self._rim_mat])
        self._rim_node.setGeometry_(rgeo)

        # ---- v6：接触阴影裙（贴边最暗 → 外圈淡出；抬起时按光源反方向偏移并柔化） ----
        sv, sc = self._sbuf, self._scbuf
        lift_loop = np.clip(V[loop, 2] - fl[loop], 0.0, None)
        soft = 1.0 / (1.0 + lift_loop / 26.0)
        lat_max = max(abs(v) for v in SHADOW_SKIRT_PT)
        for j, (lat, al) in enumerate(zip(SHADOW_SKIRT_PT, SHADOW_SKIRT_ALPHA)):
            adv = self._shadow_lean * lift_loop * (abs(lat) / lat_max) * SHADOW_LIFT_GAIN
            dv = sv[j::self._skirt_rings]
            dv[:, 0] = p[:, 0] + out[:, 0] * lat + self._shadow_dir[0] * adv
            dv[:, 1] = SHADOW_PLANE_Y
            dv[:, 2] = p[:, 2] + out[:, 1] * lat + self._shadow_dir[1] * adv
            sc[j::self._skirt_rings, 3] = al * soft
        sgeo = SceneKit.SCNGeometry.geometryWithSources_elements_(
            [_src(sv, SceneKit.SCNGeometrySourceSemanticVertex, 3),
             _src(sc, SceneKit.SCNGeometrySourceSemanticColor, 4)],
            [self._skirt_elem])
        if self._shadow_mat is not None:
            sgeo.setMaterials_([self._shadow_mat])
        self._shadow_node.setGeometry_(sgeo)

    # ---- 内部：sim 的可选渲染支持数组（缺失时回退常量，保证模块可独立验证） ----
    @staticmethod
    def _sim_array(sim, name: str, n: int, default: float) -> np.ndarray:
        fn = getattr(sim, name, None)
        if callable(fn):
            try:
                a = np.asarray(fn(), dtype=np.float32)
                if a.shape == (n,):
                    return a
            except Exception:
                pass
        return np.full(n, default, dtype=np.float32)

    # ---- 内部：灯光基向 → 阴影偏移方向与抬升柔化系数 ----
    def _sun_shadow_basis(self, sun_node, root) -> tuple[tuple[float, float], float]:
        """用 SceneKit 自己的坐标转换取平行光传播方向（不猜欧拉/矩阵约定）。

        平行光沿节点自身 -z 传播 → 让 SceneKit 把它换算到场景根节点空间，
        得到权威的传播方向 d。阴影方向 = d 的水平分量（屏幕 x = 场景 x、屏幕 y = 场景 z），
        lean = |水平|/|竖直| = 1/tan(仰角)：物体抬升 h 时影子外移 h·lean。
        """
        try:
            d = sun_node.convertVector_toNode_((0.0, 0.0, -1.0), root)
            dx, dy, dz = float(d[0]), float(d[1]), float(d[2])
            h = math.hypot(dx, dz)
            if h < 1e-6:
                return (0.0, 0.0), 0.6
            return (dx / h, dz / h), float(min(max(h / max(abs(dy), 1e-3), 0.0), 3.0))
        except Exception:
            return (0.0, 0.0), 0.6

    # ---- 内部：timer 与 tick ----
    def _start_timer(self) -> None:
        self._stop_timer()
        self._timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            1.0 / float(self._fps), self._view, "tick:", None, True)

    def _stop_timer(self) -> None:
        if self._timer is not None:
            self._timer.invalidate()
            self._timer = None

    # ---- 内部：点击穿透闸门（ignoresMouseEvents 动态闸门） ----
    def _read_force_env(self) -> None:
        """读一次验证专用环境变量并立即生效（仅 scripts/verify_routing.py --live 用，
        生产不要设；未设时 _force_ignore=None，闸门正常工作）。"""
        raw = ""
        try:
            raw = str(os.environ.get(CLICKTHROUGH_FORCE_ENV, "") or "").strip().lower()
        except Exception:
            raw = ""
        if raw in ("ignore", "receive"):
            self._force_ignore = (raw == "ignore")
            _log("RUG-INFO", f"click-through forced={raw}"
                             f"（仅 scripts/verify_routing.py --live 验证用，生产不要设）")
        else:
            self._force_ignore = None

    def _cursor_sim_xy(self):
        """当前鼠标位置 → sim 坐标（主屏本地 top-left、y 向下）。

        NSEvent.mouseLocation() 是 AppKit 全局 bottom-left；frame 取主屏 frame()。
        """
        loc = NSEvent.mouseLocation()
        f = self._screen.frame()
        x = float(loc.x) - float(f.origin.x)
        y = (float(f.origin.y) + float(f.size.height)) - float(loc.y)
        return x, y

    def _sim_asleep(self) -> bool:
        try:
            return bool(self._sim.is_asleep())
        except Exception:
            return False

    def _update_click_through(self, force: bool = False) -> None:
        """穿透闸门的唯一决策点：按光标位置/抓取/外来按下算 ignoresMouseEvents 目标值。

        驱动源（全部走这里，逻辑只此一份）：show() 的 30Hz NSTimer、全局鼠标移动
        监视器、以及窗口事件 mouseMoved_/cursorUpdate_/mouseUp_。
        毯外（dist > 40pt）→ ignores=YES（点击真正落到桌面/下层窗口）；毯上 → NO。
        """
        if self._sim is None or self._panel is None or not self._shown:
            return
        if self._force_ignore is not None:  # 验证专用覆盖：钉死状态、不再计算
            self._apply_ignore(bool(self._force_ignore), None)
            return
        try:
            cx, cy = self._cursor_sim_xy()
        except Exception:
            return
        last = self._last_cursor
        same = (last is not None
                and abs(cx - last[0]) < GATE_CURSOR_EPS_PT
                and abs(cy - last[1]) < GATE_CURSOR_EPS_PT)
        # 空闲早退：光标没动 + 毯子睡着 + 无脏标志 → 跳过重算（CPU≈0 的保证）。
        # 近距豁免（P2）：上次距离 ≤ GATE_NEAR_PT（光标在毯子附近）时不早退——
        # 否则「外来按住解除」「毯上直接点击」这类不移动光标的转变会被漏掉，
        # 状态可能冻结在 ignore 而导致毯上点击穿透到下层 App。
        near = self._last_dist is not None and self._last_dist <= GATE_NEAR_PT
        if not force and same and not self._gate_dirty and not near and self._sim_asleep():
            return
        self._last_cursor = (cx, cy)
        self._gate_dirty = False
        try:
            dist = float(self._sim.nearest_distance(cx, cy))
            if self._edit_mode:  # 编辑模式：手柄所在的区域也算「毯上」（旋转柄在毯外）
                dist = min(dist, self._handle_distance(cx, cy))
        except Exception:
            return
        self._last_dist = dist
        grabbed = bool(getattr(self._view, "_grabbed", False))
        try:
            foreign_press = (int(NSEvent.pressedMouseButtons()) != 0) and not grabbed
        except Exception:
            foreign_press = False
        self._apply_ignore(
            gate_wants_ignore(dist, self._ignoring, grabbed, foreign_press), dist)

    def _apply_ignore(self, want: bool, dist) -> None:
        """写入 panel 的 ignoresMouseEvents；仅状态翻转时写 +（限频）打 RUG-INFO。"""
        want = bool(want)
        if want == self._ignoring:
            return
        try:
            self._panel.setIgnoresMouseEvents_(want)
        except Exception as exc:
            # P2：失败不能逐帧刷日志（闸门是唯一穿透机制，会持续重试；不更新状态以便下拍重试）
            now = time.monotonic()
            if now - self._last_gate_err_log >= GATE_ERR_LOG_MIN_INTERVAL_S:
                self._last_gate_err_log = now
                _log("RUG-ERROR",
                     f"setIgnoresMouseEvents_ 失败（闸门未生效，将持续重试）：{exc}")
            return
        self._ignoring = want
        now = time.monotonic()
        if now - self._last_flip_log >= GATE_LOG_MIN_INTERVAL_S:  # 防边界抖动刷屏
            self._last_flip_log = now
            state = "ignore" if want else "receive"
            extra = f" d={float(dist):.1f}" if dist is not None else ""
            _log("RUG-INFO", f"click-through -> {state}{extra}")

    def _start_gate(self) -> None:
        """装闸门驱动源：NSTimer（活动期 30Hz / 入睡后 5Hz）+ 全局鼠标监视器（即时，尽力而为）。"""
        self._stop_gate()
        self._last_cursor = None
        self._last_dist = None
        self._gate_dirty = False
        self._gate_period = None
        self._retune_gate_timer()
        try:
            # 鼠标类事件的全局监视器不需要辅助功能权限（键盘事件才需要）；
            # mask 含 up/down：按住状态的变化也要即时进闸门（见 GATE_MONITOR_MASK）
            self._monitor_token = NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
                GATE_MONITOR_MASK, self._on_global_mouse_moved)
        except Exception as exc:
            _log("RUG-WARN",
                 f"全局鼠标监视器不可用（{exc}），闸门改由轮询 timer + 窗口事件驱动")

    def _gate_tick_period(self) -> float:
        """闸门轮询周期：毯子睡着 → 慢档（省空闲 CPU；全局监视器仍即时驱动），否则 30Hz。"""
        try:
            return GATE_TICK_IDLE_S if self._sim.is_asleep() else GATE_TICK_S
        except Exception:
            return GATE_TICK_S

    def _retune_gate_timer(self) -> None:
        """按当前状态重建闸门 timer（周期变化时才动）。"""
        want = self._gate_tick_period()
        if self._gate_period is not None and abs(want - self._gate_period) < 1e-6:
            return
        t, self._gate_timer = self._gate_timer, None
        if t is not None:
            try:
                t.invalidate()
            except Exception:
                pass
        try:
            self._gate_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                want, self._view, "gateTick:", None, True)
            self._gate_period = want
        except Exception as exc:
            self._gate_period = None
            _log("RUG-WARN", f"click-through 闸门 timer 启动/重建失败（仅靠鼠标事件驱动）：{exc}")

    def _on_gate_tick(self) -> None:
        """闸门 tick 入口：先按入睡状态调周期，再做一次决策。"""
        if not self._shown:
            return
        self._retune_gate_timer()
        self._update_click_through()

    def _on_global_mouse_moved(self, event) -> None:
        """全局鼠标回调（移动/按下/松开）：主线程直接刷新；其它线程只置脏（由 timer 消费）。

        订阅 mask 含 up/down（见 GATE_MONITOR_MASK），保证「外来按住解除」等
        不移动光标的转变也能即时纠正，而不是等下一次轮询。
        """
        try:
            if NSThread.isMainThread():
                self._update_click_through()
            else:
                self._gate_dirty = True
        except Exception:
            pass

    def _stop_gate(self) -> None:
        """撤闸门驱动源（hide 调用）：停 timer + 移除全局监视器。"""
        if self._gate_timer is not None:
            try:
                self._gate_timer.invalidate()
            except Exception:
                pass
            self._gate_timer = None
        self._gate_period = None
        token, self._monitor_token = self._monitor_token, None
        if token is not None:
            try:
                NSEvent.removeMonitor_(token)
            except Exception as exc:
                _log("RUG-WARN", f"移除全局鼠标监视器失败：{exc}")

    def _on_tick(self) -> None:
        sim = self._sim
        if sim is None or not self._shown:
            return
        now = time.monotonic()
        if self._last_t is None:
            self._last_t = now
        dt = now - self._last_t
        self._last_t = now
        dt = min(max(dt, 1.0 / 240.0), 0.05)
        try:
            if not sim.is_asleep():
                sim.step(dt)
                self._rebuild_geometry()
            if self._edit_mode:
                # 编辑模式：毯身被拖动后手柄要跟着走（尺寸/角度沿用已记录值）
                self._refresh_placement_from_sim()
                self._update_handles()
            self._err_count = 0
        except Exception as exc:
            self._err_count += 1
            if self._err_count == 1:
                _log("RUG-ERROR", f"渲染 tick 异常：{exc}")
            if self._err_count >= 90:  # 连续约 3s 失败 → 停表，避免逐帧刷日志
                _log("RUG-ERROR", "渲染 tick 连续失败，停止渲染 timer")
                self._stop_timer()

    # ---- 回调桥 ----
    def _fire_grab(self) -> None:
        if self._on_grab_cb is not None:
            try:
                self._on_grab_cb()
            except Exception as exc:
                _log("RUG-ERROR", f"on_grab 回调异常：{exc}")

    def _fire_release(self) -> None:
        if self._on_release_cb is not None:
            try:
                self._on_release_cb()
            except Exception as exc:
                _log("RUG-ERROR", f"on_release 回调异常：{exc}")
