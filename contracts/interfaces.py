"""contracts/interfaces.py — 桌面毛毯 v1 接口契约（architect 冻结）。

纯逻辑契约：只 import numpy / stdlib / dataclasses；**禁止 import AppKit / Quartz / SceneKit**
（保持纯逻辑可单测）。各模块在自己的文件中实现本契约；签名与语义以此为准，
要改动必须回报 architect（契约变更请求），不许自行发明字段/方法。

== 全局约定（所有签名共用） ==
坐标 / 单位：
  - 除注明外一律「主屏本地屏幕点 pt」：原点 = 主屏（NSScreen.mainScreen）左上角，
    x 向右、y 向下、z 向上（凸出屏幕、朝观察者）。1 场景单位 = 1 pt（正交投影）。
  - 例外：IconInfo.x_pt / y_pt 是 Finder 桌面视图坐标（桌面区域左上为原点、y 向下、pt），
    未经标定。转主屏本地屏幕点：screen = finder + (dx, dy)，(dx, dy) 来自
    IconSensor.calibration()。
  - AppKit（bottom-left origin）与 SceneKit 场景坐标的换算由 overlay/rug 实现层负责；
    契约层流动的数据一律 top-left、y 向下。
线程模型：主线程单线程。AppKit 事件 + NSTimer 驱动 sim.step()；所有回调保证在主线程；
禁止创建额外线程或锁（IconSensor 降级轮询的调度方式见其 docstring）。
安全红线（开发文档 §4，写进实现而非自觉）：~/Desktop 只读；不碰任何壁纸 API /
desktoppicture.db / defaults write / killall Finder；零网络；退出即无残留。
日志：stdout 单行 `RUG-INFO|WARN|ERROR <msg>`；禁止逐帧打印；selftest 探针行
`SELFTEST-LEVEL <int>` / `SELFTEST-OK`（无 RUG- 前缀，供脚本 grep）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

__all__ = [
    "IconInfo",
    "BumpField",
    "ClothSim",
    "IconSensorError",
    "IconSensor",
    "RugOverlay",
    "RugApp",
    "parse_args",
    "main",
]


# --------------------------------------------------------------------------
# icons.py / bumps.py 共用的数据类型
# --------------------------------------------------------------------------
@dataclass
class IconInfo:
    """桌面上一个图标的只读快照。

    Attributes:
        name: 文件名（含扩展名、不含路径），如 "notes.txt"。
        x_pt: Finder 桌面视图坐标 x（pt，桌面左上为原点、向右）。
        y_pt: Finder 桌面视图坐标 y（pt，向下）。未经标定；
              转主屏本地屏幕点 = (x_pt + dx, y_pt + dy)，(dx, dy) 见 IconSensor.calibration()。
        size_bytes: 文件大小（os.stat().st_size，只读访问）；stat 失败时为 0。
    """

    name: str
    x_pt: float
    y_pt: float
    size_bytes: int


# --------------------------------------------------------------------------
# bumps.py — 图标 → 高度场
# --------------------------------------------------------------------------
class BumpField:
    """图标位置 → 静态目标高度场（"taller stacks, bigger bumps"）。

    语义（向量化参考实现，M2 可调参数不调结构）::

        A_i  = base_height_pt * (1 + min(log2(1 + size_bytes / 2**20), 2.0) * size_gain)
        c_i  = (x_pt + dx, y_pt + dy)        # calibration 偏移在本类内应用
        z(p) = min(max_height_pt, Σ_i A_i * exp(-|p - c_i|² / (2 * sigma_pt²)))

    文件越大 A 越大（对应文档 §7.3）；同格多图标的高斯直接相加再截断
    （对应 §7.2 "每档堆叠 +8pt、上限 28pt" 的堆叠效应）。icons 为空 → 恒为 0。
    纯逻辑：无任何 I/O；同一输入重复调用结果确定。
    """

    def __init__(
        self,
        icons: list[IconInfo],
        width_pt: float,
        height_pt: float,
        calibration: tuple[float, float] = (0.0, 0.0),
        sigma_pt: float = 45.0,
        base_height_pt: float = 10.0,
        size_gain: float = 0.6,
        max_height_pt: float = 28.0,
    ) -> None:
        """构建高度场。

        Args:
            icons: 图标快照（Finder 桌面视图坐标，未标定）。
            width_pt / height_pt: 主屏尺寸（pt），用于裁剪屏外图标。
            calibration: (dx, dy)，IconSensor.calibration() 的结果；
                screen_point = finder_point + calibration。
            sigma_pt: 高斯宽度（文档初值 ≈45pt）。
            base_height_pt: 隆起基础高度（文档初值 10pt）。
            size_gain: 文件大小增益系数（文档初值 0.6）。
            max_height_pt: 叠加截断上限（文档初值 28pt）。
        """
        raise NotImplementedError

    def heights(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """查询高度场（ClothSim 每次重建 z 目标时批量调用）。

        Args:
            xs / ys: 同形状的 numpy 数组（任意形状），主屏本地屏幕点 pt（top-left, y-down）。

        Returns:
            与输入同形状的 np.ndarray：z 高度 pt（≥0，向上）。不修改输入数组。
        """
        raise NotImplementedError


# --------------------------------------------------------------------------
# cloth.py — 2.5D 布料模拟
# --------------------------------------------------------------------------
class ClothSim:
    """布料模拟：**v4 起为真 3D 位置 PBD**（2.5D 高度场已退役——同一 (x,y) 只能一个
    高度、数学上无法对折；决策与调参记录见开发文档 §12.16）。

    模型（实现语义）：3D 位置/速度；重力 -z、地板 z ≥ 图标隆起高度；距离约束
    拉伸全刚度 / **压缩软**（织物不可推：压缩转成拱起/折痕/真翻折，而不是被强制摊平）+
    结构边拉伸上限；压缩屈曲按「压缩量 → 高度」注入并以局部高度饱和（拖动中生长、
    停手即止）；自碰撞保证翻折两层间距（同一 (x,y) 可出现多层——真 3D 语义）；
    抓取 = 抓点钉 xy + 近强远弱跟手弹簧；休眠 = 3D 动能 < ε 且未抓取。

    网格：cols×rows 粒子，间距 = width_pt/(cols-1) × height_pt/(rows-1)。
    顶点顺序（vertices()/uv() 共用，行主序）：index = row * cols + col；row 0 = 顶边（y 最小）。
    公开签名（width_pt, height_pt, cols, rows, bumps）不变；M1 的调优 keyword 组
    已随模型更替（v4 新增 gravity/buckling/collision_* 等，含义见 §12.16），实现另提供
    flatten_folds()/placement()/set_placement()（菜单「摊平」与编辑模式用）。
    纯逻辑（numpy）：无 I/O、无 AppKit、step() 内不读时钟（时间信息只来自 dt），
    由外部 NSTimer 以 30~60fps 驱动。
    """

    def __init__(
        self,
        width_pt: float,
        height_pt: float,
        cols: int = 90,
        rows: int = 56,
        bumps: BumpField | None = None,
    ) -> None:
        """创建布料，静止平铺在主屏本地坐标 (0,0)-(width_pt,height_pt)。

        bumps 可为 None（无隆起模式）；后续可随时用 set_bumps() 替换。
        """
        raise NotImplementedError

    # ---- 交互（由 RugOverlay 的鼠标事件驱动，主线程） ----
    def grab(self, x: float, y: float, radius: float = 40.0) -> bool:
        """尝试在 (x, y)（主屏本地 pt）抓取布料。

        最近粒子与 (x, y) 的 xy 平面距离 ≤ radius 即命中：记录抓点、wake()、返回 True；
        未命中返回 False 且不改变任何状态。已抓取时再次调用视为重抓（更新抓点）。
        """
        raise NotImplementedError

    def drag_to(self, x: float, y: float) -> None:
        """把抓点目标移动到 (x, y)（主屏本地 pt）。未抓取时为 no-op。

        拖动使材料跟手/滞后产生压缩，屈曲把压缩转成拱起 → 波浪/折痕/真翻折（v4 真 3D）。
        """
        raise NotImplementedError

    def release(self, fling: bool = True) -> None:
        """释放抓点。重复调用幂等。

        fling=True 且最近若干帧平均速度超阈值时，给全体粒子注入平移初速度 +
        角速度（抛掷飞行，视频 5–7s 效果）；释放后布料应能在数秒内 is_asleep()。
        """
        raise NotImplementedError

    def throw_in(
        self,
        target_center: tuple[float, float],
        from_corner: str = "top_left",
    ) -> None:
        """「铺上」动画入口：毯子从屏幕一角飞入并落在 target_center。

        Args:
            target_center: 落点中心（主屏本地 pt）。
            from_corner: "top_left" | "top_right" | "bottom_left" | "bottom_right"；
                非法值抛 ValueError。
        行为：把整块布瞬移到对应角落（屏外/收拢）并注入朝 target_center 的初速度 +
        旋转；之后由调用方 step() 驱动，落定后 is_asleep()。
        """
        raise NotImplementedError

    # ---- 步进与休眠 ----
    def step(self, dt: float) -> None:
        """推进一个 tick；dt 单位秒（典型 1/30~1/60），内部 clamp 到 [1/240, 0.05]。

        一个 tick 完成：阻尼 → 跟手/牵引 → 重力 → 积分 → 压缩屈曲 → 距离约束
        （运动 3 轮 / 准静态自适应收尾 + 修正限速）→ 拉伸上限 → 地板（图标隆起）→
        自碰撞 → 速度回写 + 摩擦/接触非弹性。
        注（2026-10-08 v4，注释性说明，不改公开语义）：准静态期实现会做「收尾迭代 +
        修正限速 + 限速记速」以加速入睡（否则褶皱堆里「修正→速度→运动→再修正」
        会自激、毯子永不入睡，见 §12.16）；任何真实运动立即恢复常规路径。
        is_asleep() 后外部应停止调用 step()（CPU≈0% 验收项）。
        """
        raise NotImplementedError

    def is_asleep(self) -> bool:
        """kinetic_energy() < ε 且未抓取 → True。休眠中遇任何事件须被 wake() 拉起。"""
        raise NotImplementedError

    def wake(self) -> None:
        """强制唤醒，直到再次收敛才允许休眠（抓取/图标变化/铺上后调用）。"""
        raise NotImplementedError

    def set_bumps(self, field: BumpField | None) -> None:
        """替换 BumpField（None = 无隆起/降级模式）；下次 step() 生效。配合 wake() 使用。"""
        raise NotImplementedError

    # ---- 几何输出（RugOverlay 每帧读取，重建 SCNGeometry） ----
    def vertices(self) -> np.ndarray:
        """返回 (N, 3) float32 C-contiguous 数组：x, y（主屏本地 pt，top-left, y-down）
        与 z（高度 pt，≥ 地板高度（图标隆起）向上；**v4 真 3D**：翻折时同一 (x,y) 可
        出现多层，z 不再单值）。N = cols*rows = 5040。行主序见类 docstring。
        """
        raise NotImplementedError

    def uv(self) -> np.ndarray:
        """返回 (N, 2) 数组，与 vertices() 行序一致：u = col/(cols-1) 从左到右，
        v = row/(rows-1) 从上到下（v=0 对应贴图顶边）。

        注意：SceneKit 贴图 v 轴朝上，渲染层喂 SCNGeometrySource 前需 1-v（见 contract.md §5）。
        """
        raise NotImplementedError

    def indices(self) -> np.ndarray:
        """返回 (M,) uint32 三角列表；环绕顺序与 spike build_indices 相同
        （对每个格子 a=j*cols+i 输出 [a, a+1, a+cols+1, a, a+cols+1, a+cols]）。

        必须复用同一 buffer（每次调用返回同一数组），勿逐帧重建（0.21ms/帧 的前提之一）。
        """
        raise NotImplementedError

    # ---- 查询 ----
    def nearest_distance(self, x: float, y: float) -> float:
        """(x, y)（主屏本地 pt）到最近粒子的 xy 平面距离（pt，不含 z）。

        供 RugOverlay.hitTest_ 使用：≤ 40pt 返回 self（收鼠标）。>40pt 时返回 None 只让本
        视图不吃这一击（**不会**把点击转交下层窗口，见 RugOverlay 的穿透说明与开发文档 §7.1）。
        """
        raise NotImplementedError

    def kinetic_energy(self) -> float:
        """Σ|v|²（内部单位，仅用于休眠阈值比较与单测收敛断言）。"""
        raise NotImplementedError


# --------------------------------------------------------------------------
# icons.py — 图标感知（只读）
# --------------------------------------------------------------------------
class IconSensorError(RuntimeError):
    """图标感知失败：权限被拒（AppleEvent -1743）/ osascript 超时 / 输出解析失败 /
    标定失败。RugApp 捕获后进入降级模式（无隆起、物理照常），且不得二次弹权限。
    """


class IconSensor:
    """只读感知 ~/Desktop：图标位置/大小 + 变化通知 + Finder→屏幕坐标标定。

    安全：对 ~/Desktop 与 Finder 只读（名字、position、stat）；唯一的权限依赖是
    「自动化 → Finder」（S5）。禁止移动/重命名/隐藏任何文件，禁止读 .DS_Store。
    """

    def __init__(self, desktop_path: str | None = None) -> None:
        """desktop_path 默认 ~/Desktop；仅用于只读 stat 与变化监听。"""
        raise NotImplementedError

    def query(self) -> list[IconInfo]:
        """返回当前桌面项快照 list[IconInfo]（顺序不限）。

        实现：osascript 子进程跑只读 AppleScript（tell application "Finder" 取
        desktop items 的 name 与 position，超时 ≈5s），再 os.stat 本地补 size_bytes
        （文件已消失 → 0）。失败 → raise IconSensorError（message 含原因）。
        主线程调用（子进程阻塞最多数秒）。
        """
        raise NotImplementedError

    def start(self, on_change: Callable[[], None]) -> None:
        """订阅桌面变化通知（G4：新建/删除文件后 2s 内触发重建隆包）。

        on_change: 无参回调，**保证主线程调用**，须快速返回（只置脏标志）。
        优先 FSEvents 只读监听 desktop_path；`import FSEvents` 失败
        （requirements 未含 pyobjc-framework-FSEvents）或流启动失败时，
        **降级为 1~2s 轮询**（本契约允许的降级，实现用 NSTimer 保持主线程）。
        幂等：重复 start 先停掉旧流。
        """
        raise NotImplementedError

    def stop(self) -> None:
        """停止通知。幂等；之后可再次 start()。"""
        raise NotImplementedError

    def calibration(self) -> tuple[float, float]:
        """一次性标定，返回 (dx, dy)：screen_point = finder_point + (dx, dy)。

        建议实现：AppleScript 读 Finder 桌面窗口 bounds（同一 Automation 权限、只读）
        推导偏移；结果可在实例内缓存。失败 → raise IconSensorError
        （调用方 RugApp 回退近似值 (0.0, 24.0) 菜单栏高度并打 RUG-WARN）。
        """
        raise NotImplementedError


# --------------------------------------------------------------------------
# overlay.py — 覆盖窗口 + 渲染 + 点击穿透闸门（AppKit 层，契约只定义公开面）
# --------------------------------------------------------------------------
class RugOverlay:
    """全屏覆盖窗口（NSPanel）+ SCNView 渲染 + 40pt 判定 + 点击穿透闸门。

    本类涉及 AppKit/SceneKit，**实现细节见 spike/m0_overlay_spike.py 与 contract.md §5**；
    契约只固定公开方法与回调。实现要点（开发文档 §7.1/§7.4）：
      - NSPanel：level = kCGDesktopIconWindowLevel + 8（动态读取，不写死）、borderless、
        透明、无阴影、CanJoinAllSpaces|Stationary|IgnoresCycle、铺满 screen、orderFrontRegardless；
      - SCNView 透明 + 正交相机（usesOrthographicProjection，1 场景单位 = 1 pt）；
      - NSTimer(fps) 驱动 sim.step() + 逐帧重建几何（索引 buffer 复用）；
        法线由三角网格实时计算（朝下翻正，v4）→ Blinn 光照（平行光+环境光）；
      - 布面材质（v4）：**顶/背两张贴图 + 面朝向分渲染**——同索引双 element，
        正面材质 cullMode=Front（毯面朝上时可见）、背面材质 cullMode=Back（翻折露底时
        可见，贴图取同目录 <名字>-back.png，缺失回退正面）；
      - 伪阴影：顶点同源的暗色剪影 mesh（偏移 6–10pt、z 压平、边缘 alpha 渐隐）；
      - 根 NSView.hitTest_：指针位置喂 sim.nearest_distance(x, y)，≤ 40pt 返回 self，
        否则 None。**注意（真机路由实测 2026-10-08）**：hitTest 返回 None 只会吞掉
        事件，**不会**把点击交给下层窗口——「毯外点击真正穿透」必须由 ignoresMouseEvents
        动态闸门实现（毯外 → setIgnoresMouseEvents_(True)，点击直达桌面/下层 App；毯上 →
        恢复接收，再交本 hitTest 做 40pt 精确判定），见 开发文档 §7.1；
      - mouseDown / mouseDragged / mouseUp → sim.grab / drag_to / release(fling=True)。
    """

    def __init__(
        self,
        screen: Any,
        texture_path: str,
        sim_factory: Callable[[float, float], ClothSim],
        on_grab: Callable[[], None] | None = None,
        on_release: Callable[[], None] | None = None,
        fps: int = 30,
    ) -> None:
        """Args:
        screen: NSScreen（v1 固定主屏 NSScreen.mainScreen()）。
        texture_path: 贴图绝对路径（assets/rugs/rug-*.png，本地文件、零网络 S4）。
        sim_factory: 工厂回调 (screen_w_pt, screen_h_pt) -> ClothSim；
            **每次 show() 都调用工厂新建 sim**（rug.py 侧负责闭包内注入 BumpField）。
        on_grab / on_release: 主线程回调，供 RugApp 状态机（resting↔dragging）驱动。
        fps: 30~60。
        """
        raise NotImplementedError

    def show(self) -> None:
        """铺上窗口（幂等）：建/复用 panel+view，orderFrontRegardless，启动渲染 timer。
        首次调用后 sim 属性可用（经 sim_factory 新建）。
        """
        raise NotImplementedError

    def hide(self) -> None:
        """掀开窗口（幂等）：停 timer、orderOut、丢弃 sim（下次 show 重建）。"""
        raise NotImplementedError

    def is_shown(self) -> bool:
        """窗口当前是否可见。"""
        raise NotImplementedError

    @property
    def sim(self) -> ClothSim | None:
        """当前活动的 ClothSim；尚未 show() 或已 hide() 时为 None。

        RugApp 经此访问 sim（set_bumps / wake / throw_in / is_asleep）。
        """
        raise NotImplementedError


# --------------------------------------------------------------------------
# rug.py — 入口与总控（AppKit 层，契约只定义公开面）
# --------------------------------------------------------------------------
class RugApp:
    """accessory 菜单栏应用 + 总控状态机。

    职责（AppKit 细节见 spike / contract.md §5）：
      - NSApplication(ActivationPolicyAccessory) + NSStatusItem 菜单：
        「铺上 / 掀开 ⌘\\」「地毯款式」子菜单（glob assets/rugs/rug-*.png；切换 =
        hide 旧 RugOverlay、以新贴图重建）、「退出」。
      - 状态机（state 取下列 STATE_* 常量）：
        idle --lay_down--> throwing --落定(is_asleep 或超时≈4s)--> resting
        --on_grab--> dragging --on_release(经 release/settle)--> resting
        --lift--> lifting --完成/超时--> idle；throwing/dragging 中允许直接 lift。
      - IconSensor 生命周期（主线程）：query() 成功 → calibration() → 构建 BumpField
        → overlay.sim.set_bumps(field) + wake()；sensor.on_change → 重新 query + 重建
        BumpField → set_bumps + wake()（G4：2s 内更新，轮询/FSEvents 均满足）。
      - 权限降级（S5）：query/calibration 抛 IconSensorError → degraded=True：
        不建 BumpField（物理照常好玩），菜单标注「图标隆起：不可用（需自动化权限）」，
        **不发起第二次权限请求**。
      - 退出必须彻底（S6）：stop sensor、hide overlay、terminate，最后 os._exit(0) 兜底
        （app.run() 不可靠返回，见 spike）。
      - selftest 模式（--selftest）：启动并 lay_down；t≈2s 用 CGWindowListCopyWindowInfo
        探针找到本 app 的面板，打印 `SELFTEST-LEVEL <level 数字>`；t≈5s 打印
        `SELFTEST-OK` 后 os._exit(0)。找不到面板 → 打印 `SELFTEST-LEVEL -1`，
        t≈5s 打印 `SELFTEST-FAIL` 后 os._exit(1)。
    """

    STATE_IDLE = "idle"
    STATE_THROWING = "throwing"
    STATE_RESTING = "resting"
    STATE_DRAGGING = "dragging"
    STATE_LIFTING = "lifting"

    def __init__(self, selftest: bool = False, assets_dir: str | None = None) -> None:
        """Args:
        selftest: True 走 selftest 分支（见类 docstring）。
        assets_dir: 地毯贴图目录；默认 <rug.py 所在目录>/assets/rugs。
        """
        raise NotImplementedError

    def run(self) -> int:
        """进入主循环（正常路径不返回）。Ctrl-C 与菜单退出都必须走到 os._exit 兜底。"""
        raise NotImplementedError

    def lay_down(self) -> None:
        """「铺上」：仅 idle 态有效（其他态幂等忽略）→ throwing；show overlay 并
        sim.throw_in(target_center=主屏中心, from_corner=top_left)。
        """
        raise NotImplementedError

    def lift(self) -> None:
        """「掀开」：非 idle 态 → lifting：抓毯角 drag 至屏外后 release；
        落定/超时 → idle 并 hide overlay。
        """
        raise NotImplementedError

    def quit(self) -> None:
        """彻底退出（S6）：清理 sensor/overlay 后 terminate + os._exit(0) 兜底。"""
        raise NotImplementedError

    @property
    def state(self) -> str:
        """当前状态（STATE_* 之一）。"""
        raise NotImplementedError

    @property
    def degraded(self) -> bool:
        """True = 图标隆起降级模式（权限被拒），菜单需标注。"""
        raise NotImplementedError


def parse_args(argv: list[str] | None = None) -> Any:
    """解析命令行。目前仅 --selftest（argparse.Namespace.selftest: bool）。"""
    raise NotImplementedError


def main(argv: list[str] | None = None) -> int:
    """rug.py 入口：parse_args → RugApp(selftest=...).run()。
    KeyboardInterrupt → 清理后返回 0（或直接 os._exit(0)）。
    """
    raise NotImplementedError
