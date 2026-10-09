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
desktoppicture.db / defaults write / killall Finder；零网络；退出即无系统状态残留
（v6.2 起有意的持久副产物只有用户 Library 下三件套：~/Library/Logs/DesktopRug.log、
~/Library/Application Support/DesktopRug/{settings.json,instance.lock}，均可删，
见开发文档 §12.19）。
日志：stdout 单行 `RUG-INFO|WARN|ERROR <msg>`；禁止逐帧打印；selftest 探针行
`SELFTEST-LEVEL <int>` / `SELFTEST-OK`（无 RUG- 前缀，供脚本 grep）。
v6.2 日志落盘：stdin 与 stdout 都不是终端（Finder 双击 / `open` 启动）时，
stdout/stderr 追加到 ~/Library/Logs/DesktopRug.log（>1MB 轮转为 .1）；终端运行
输出不受影响（有一端是终端即原样输出）；RUG_NO_FILE_LOG=1 强制关闭落盘。
落盘不改变上面任何行格式（构建门禁按原格式 grep SELFTEST-OK）。
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
# cloth.py — 3D 布料模拟
# --------------------------------------------------------------------------
class ClothSim:
    """布料模拟：**v4 起为真 3D 位置 PBD**（2.5D 高度场已退役——同一 (x,y) 只能一个
    高度、数学上无法对折；决策与调参记录见开发文档 §12.16）。

    模型（实现语义）：3D 位置/速度；重力 -z、地板 z ≥ 图标隆起高度；距离约束
    拉伸全刚度 / **压缩软**（织物不可推：压缩转成拱起/折痕/真翻折，而不是被强制摊平）+
    结构边拉伸上限（**始终按 rest0 原始材料长度**算——`_strain_rest0`，不随材料记忆
    缩短；否则折痕记进 rest 后被上限锁死，用户想把折拖平都拖不动）；压缩屈曲按
    「压缩量 → 高度」注入并以局部高度饱和（拖动中生长、停手即止）；自碰撞保证翻折
    两层间距（同一 (x,y) 可出现多层——真 3D 语义）；抓取 = 抓点钉 xy（v6.3：快拖时抓点
    抬离桌面 → 毯子被拎起垂挂）+ 跟手助力（v5 起只驱动质心跟随手、不参与形状；v6.4 起
    形状由**材料记忆 + 「放下就记住」的保持**共同维持）；休眠 = 3D 动能 < ε 且未抓取
    （保持态也能入睡，见下）。

    网格：cols×rows 粒子，间距 = width_pt/(cols-1) × height_pt/(rows-1)。
    顶点顺序（vertices()/uv() 共用，行主序）：index = row * cols + col；row 0 = 顶边（y 最小）。
    公开签名（width_pt, height_pt, cols, rows, bumps）不变；M1 的调优 keyword 组
    已随模型更替（v4 新增 gravity/buckling/collision_* 等，含义见 §12.16；v6.3 新增
    fling_lift / fling_lift_max / throw_arc_z / grab_lift_v0 / grab_lift_gain /
    grab_lift_max / grab_lift_rate 共 7 个；v6.4 新增 plastic_rate / plastic_thresh /
    plastic_min_frac / pose_k / pose_delay_ticks / pose_hold_ticks / pose_gain 共 7 个，
    均为 keyword-only、默认值即出厂手感，默认与语义见 __init__ 注记），实现另提供
    flatten_folds()/placement()/set_placement()（菜单「摊平」与编辑模式用）。

    注（2026-10-09 v6.4「材料记忆 / 放下就记住」，注释性说明，不改公开语义）：
    - **材料记忆**（塑性 rest，修「折起来老是自己恢复平整」）：边缘 rest 就地朝当前
      距离改写。压缩方向：距离约束边（结构横/竖 + 两条剪切对角）在 |边| <
      plastic_thresh×rest0 时按压缩深度加权蠕变（速率 clip 到 [0, 2]×plastic_rate，
      每 3 tick 写一次）；**折痕对**（轴向与对角 (i,i±2)，触发比 ≈ crease_ratio）进
      圆角区时**瞬时**记入（真毯子一折就留痕）；所有记忆下限 plastic_min_frac×rest0。
      拉伸方向**双向**更新（|边| > rest×1.03，内部常量）：把折拖平 = 材料重新屈服成
      平面，不会「记住折痕后拉不平」、也不会越拖越松垮；上限 rest0。
      只在抓取中或真被揉动（运动门控 > ε）时写入——落定后形状就地冻结。
      「抚平折痕 / 重置摆放 / 改尺寸」走 _rebuild_constraints() 重建 rest → 记忆清零。
      圆角下限（_crease_min）恒按 rest0 算，不随记忆缩短。
    - **「放下就记住」保持**：松手（且未甩出）后延迟捕获整片形状——捕获前先把退化互穿
      清一次（_separate_degenerate_overlaps：自碰撞只推抬起粒子、近邻按「材料是否直连」
      豁免，斜折两层落到地板高度时正好落在两条规则的缝里，留下 ~0.05pt 贴合互穿 →
      渲染上是逐像素锯齿「撕口」）；清理按「实际距离 vs 网格距离」判层（≤0.55× 网格
      距离 = 中间折了、把上层沿 z 推到碰撞间距），向量化、单次 ~ms 级，**运行中每
      30 tick 还会周期性再清一次**（仅抓取中/保持中/运动中，摊薄 ≈0.16ms/帧，见 step
      的 7c2）。随后把**所有**记忆组的 rest 钳在 [plastic_min_frac, 1.0]×rest0 写入
      当前距离，并记下抬起粒子的高度作为保持目标；此后由**位置投影**（见硬约束一）
      把抬起粒子按回捕获高度。
      清空路径（以代码为准）：grab、set_placement（flatten_folds / 重置 / 改尺寸同路）、
      release 的甩出分支；pose_hold_ticks > 0 时窗口到点也会结束保持（形状交给 rest
      记忆）。throw_in 不额外清空保持状态（实现只在新建/换图后的整布上调用）。
    - **对角折痕圆角下限（v6.4 补）**：只约束轴向时「斜着折」会出现 0 半径锐折（实测
      对角粒子被折到 0.05pt，既是纸片锐折、又落进自碰撞的近邻豁免 → 互穿）；实现补
      两条对角 (i,i±2 格) 组（在枚举里排在轴向组之后，`_diag_group_from` 即对角组起
      始下标），用**同一** crease_ratio 的圆角下限（rest0 基准）。对角组带**全局门控**：
      整块布完全平铺（无任何粒子高于地板 + 6pt）时才整组跳过；不能用「组内最小距离」
      这种局部门控（会在跳过↔放松间闪烁 → 与形状保持叠成极限环，实测 ke 卡在 192
      永不入睡 + 两个回归测试变红）。
    - 实现硬约束一（**形状保持必须是位置投影，不能用速度弹簧**）：速度弹簧会与
      「接触排除」交替触发留下极限环 → ke 反复冲到 2e3、永不入睡（实测）；位置投影
      净位移 ≈ 抵消重力 → 速度回写看到 ~0、可正常入睡，同时把姿态稳稳压住
      （实测下陷 <0.2pt）。据此 pose_k（弹簧刚度）已废弃，见 __init__。
    - 实现硬约束二（**折痕/弯曲约束的分组着色必须沿 (i,i+2) 链交替**）：链上相邻两对
      共享端点，同组会让 fancy-index 原地更新互相覆盖、圆角下限形同虚设（实现按
      c//2、r//2 的奇偶分组，对角组按 (r+c)//4、(r−c)//4；2026-10-09 实测）。

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

        注（2026-10-09 v6.3「大折 / 腾空」，注释性说明，不改公开语义）：实现另带一组
        keyword-only 调优参数（全部有默认值、默认即可用；v4 组见 §12.16）。v6.3 新增
        7 个——对齐参考效果「毯子被甩起来时是腾空垂挂出大折，而不是贴地滑」：
        - fling_lift=0.45：抛掷竖直初速度 = 水平速度 × 该系数（0 = 恢复旧行为「只贴地滑」）；
        - fling_lift_max=850.0：抛掷竖直初速度上限（pt/s；拱高 ≈ v²/2g ≈ 300pt）；
        - throw_arc_z=480.0：「铺上」飞入的竖直初速度（pt/s；拱高 ≈ v²/2g ≈ 96pt）；
        - grab_lift_v0=700.0：抓点抬升的起始拖动速度（pt/s）——慢拖仍贴地滑；
        - grab_lift_gain=0.16：抓点目标高度 =（拖动速度 − v0）× 该系数；
        - grab_lift_max=170.0：抓点抬升上限（pt）；
        - grab_lift_rate=6.0：抬升/回落平滑速率（1/s）。
        另有默认值调整：fold_cap_z（屈曲饱和高度）默认 40 → 240——实测桌面平摊折只
        拱到 ~22pt，抬高上限是为极端堆叠兜底。

        注（2026-10-09 v6.4「材料记忆」，注释性说明，不改公开语义）：另带 7 个
        keyword-only 参数（全部有默认值、默认即可用；整体语义见类 docstring 的
        v6.4 注）——对齐用户报障「折起来之后老是自己恢复/变得平整」：
        - plastic_rate=0.03：蠕变速率（/tick，每 3 tick 写一次）——受压边 rest 朝当前
          距离靠近的速度；实现取 max(值, 0)，0 = 关闭材料记忆（等价 v6.3 理想弹性）；
        - plastic_thresh=0.80：触发阈值——|边| < 0.80×rest0 才写入记忆（实现 clamp 到
          [0.05, 1.0]）；
        - plastic_min_frac=0.55：记忆下限——rest 最短可缩到 0.55×rest0（防越折越小、
          堆成一团；实现 clamp 到 [0.05, 1.0]）；
        - pose_k=300.0：**v6.4 起已废弃**——形状保持最初用速度弹簧（刚度单位 /s²），
          实测弹簧与「接触排除」交替触发留下极限环、永不入睡（ke 反复冲到 2e3），已
          改为**位置投影**（pose_gain，见类 docstring 硬约束一与 step 的 7d）；实现只
          保存该值（max(值, 0)）、不再参与任何动力学，改它没有任何效果，保留参数只为
          签名兼容；
        - pose_delay_ticks=12：松手（未甩出）后延迟多少 tick 再捕获形状（≈0.4s @30fps；
          先让自碰撞把折层推开、落定一点再冻结）；实现取 max(int(值), 0)；
        - pose_hold_ticks=0：形状保持窗口（tick）；0 或负数 = 一直保持到被抓住（当前
          默认，即「放下就记住」）；
        - pose_gain=0.35：形状保持**位置投影**的每帧增益（实现 clamp 到 [0.05, 1.0]，
          应用时再乘 0.5 软投影系数 → 有效 0.175×高度误差；与重力的数值平衡下陷实测
          <0.2pt）。
        """
        raise NotImplementedError

    # ---- 交互（由 RugOverlay 的鼠标事件驱动，主线程） ----
    def grab(self, x: float, y: float, radius: float = 40.0) -> bool:
        """尝试在 (x, y)（主屏本地 pt）抓取布料。

        最近粒子与 (x, y) 的 xy 平面距离 ≤ radius 即命中：记录抓点、wake()、返回 True；
        未命中返回 False 且不改变任何状态。已抓取时再次调用视为重抓（更新抓点）。
        v6.3：命中时抓点抬升状态复位为 0（每次抓取都从贴地开始）；此后是否被抬离
        桌面由拖动速度决定，见 drag_to。
        v6.4：命中即**清空形状保持**（捕获目标与延迟倒计时一并复位）——手一碰就完全
        自由、可随意拖平/重折；松手（未甩出）后会按新形状重新捕获，见 release。
        """
        raise NotImplementedError

    def drag_to(self, x: float, y: float) -> None:
        """把抓点目标移动到 (x, y)（主屏本地 pt）。未抓取时为 no-op。

        拖动使材料跟手/滞后产生压缩，屈曲把压缩转成拱起 → 波浪/折痕/真翻折（v4 真 3D）。
        v6.3「快拖抬升」：最近几帧平均拖动速度 > grab_lift_v0 时抓点被抬离桌面——
        目标高度 =（拖动速度 − grab_lift_v0）× grab_lift_gain、上限 grab_lift_max，
        按 grab_lift_rate 指数平滑跟进/回落 → 毯子被拎起垂挂出大折；慢拖（≤ v0）
        保持贴地滑的旧手感。
        """
        raise NotImplementedError

    def release(self, fling: bool = True) -> None:
        """释放抓点。重复调用幂等。

        fling=True 且最近若干帧平均速度超阈值时，给全体粒子注入平移初速度 +
        角速度 + **竖直初速度**（抛掷飞行，视频 5–7s 效果）。竖直分量 v6.3 新增：
        min(水平速度 × fling_lift, fling_lift_max)，且抓点附近抬得最高、最远端衰减
        到 ≈0.45× → 腾空垂挂出大折，落地靠非弹性接触停住；fling_lift=0 恢复旧行为
        （纯贴地滑）。
        v6.4「放下就记住」：**未甩出**（含 fling=False，即慢放/轻放）时置延迟捕获
        倒计时（pose_delay_ticks ≈12 tick ≈0.4s，先让自碰撞把折层推开）；到点把整片
        形状写进 rest + 记下高度目标，此后由**位置投影**保持（见 step 的 v6.4 注）。
        **甩出分支不捕获**（空中姿态不留）并清空既有保持；其余清空路径：grab、
        set_placement（摊平/重置/改尺寸同路）。释放后布料应能在数秒内 is_asleep()
        （保持态也满足，见 step）。
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
        旋转；之后由调用方 step() 驱动，落定后 is_asleep()。v6.3：另注入竖直初速度
        throw_arc_z（飞入带弧线，拱高 ≈ throw_arc_z²/2g ≈ 96pt，落地由非弹性接触
        接管、不弹跳；throw_arc_z=0 恢复旧行为——z 始终贴地）。
        """
        raise NotImplementedError

    # ---- 步进与休眠 ----
    def step(self, dt: float) -> None:
        """推进一个 tick；dt 单位秒（典型 1/30~1/60），内部 clamp 到 [1/240, 0.05]。

        一个 tick 完成：阻尼 → 跟手/牵引 → 重力 → 运动门控 + 积分 → 延迟捕获/保持
        窗口倒计时（v6.4）→ 抓点钉住（v6.3：含快拖抬升）→ 压缩屈曲 → 距离约束
        （运动 3 轮 / 准静态自适应收尾 + 修正限速）→ 拉伸上限 → 塑性蠕变（v6.4）→
        周期性退化清理（v6.4，每 30 tick）→ 形状保持位置投影（v6.4）→ 地板（图标隆起）
        → 自碰撞 → 速度回写 + 摩擦/接触非弹性 + 保持态竖直速度冻结（v6.4）。
        注（2026-10-08 v4，注释性说明，不改公开语义）：准静态期实现会做「收尾迭代 +
        修正限速 + 限速记速」以加速入睡（否则褶皱堆里「修正→速度→运动→再修正」
        会自激、毯子永不入睡，见 §12.16）；任何真实运动立即恢复常规路径。
        注（2026-10-09 v6.3，注释性说明，不改公开语义）：运动门控（区分「运动中」与
        准静态收尾；实现把 xy 平均速度 / 80pt/s、竖直平均速度 / 200pt/s 归一化取大
        并 clamp 到 [0,1]，≤0.01 视为静止）中竖直**必须**计入：否则「腾空抛掷/垂挂」
        这类 xy 静止、纯 z 运动的姿态会被判成准静态、被准静态收尾的每帧修正位移上限
        （settle_move_cap）钳住（实测注入 vz=500 只升到 18pt 就被按回，飞不起来）。
        竖直项还须减掉本帧刚加的重力增量（接触粒子在上一帧末尾回写时已清零，剩的才是
        真实竖直运动）：不减则每颗粒子带 -g·dt≈-40pt/s 的假运动、门控被钉在 0.195、
        永不进入收尾（实测入睡从 2.9s 拖到 17~36s）。门控必须保持在积分**前**——挪到
        帧尾改用回写速度会形成「修正→速度→门控」自激，破坏收敛。
        注（2026-10-09 v6.4，注释性说明，不改公开语义）：材料记忆/形状保持的时序与约束：
        - 延迟捕获与保持窗口倒计时在积分**前**：未抓取时 pose_pending 递减、到 0 调
          _capture_pose()；pose_hold_ticks > 0 时窗口倒计时到点关保持（形状交给 rest
          记忆）；
        - 塑性蠕变在约束求解**后**（用解完的几何，每 3 tick 一次；仅抓取中或真被揉动
          motion_gate > ε 时写入）；
        - 周期性退化清理（7c2）在蠕变之后：每 30 tick 一次、仅抓取中/保持中/运动中跑
          （_separate_degenerate_overlaps，向量化、摊薄 ≈0.16ms/帧）——自碰撞的两条
          规则漏洞（只推抬起粒子 + 近邻豁免）会留下贴合互穿，渲染上是逐像素锯齿撕口；
        - 形状保持 = **位置投影**（不是速度弹簧，见类 docstring 硬约束一）：只对捕获时
          高于地板 +1pt、且**上一帧**未被地板/下层托住的粒子（`_support_prev` 接触优先）
          把 z 误差按 pose_gain（实现再 ×0.5 软投影）投影回去，本身不注入速度；地板与
          自碰撞在其后应用、永远优先（否则会把自碰撞刚推开的粒子又拽回去）；
        - 帧尾对保持中的粒子把竖直速度冻结（×0.05）：投影不注入速度，但重力每帧增量与
          「接触排除」交替会让残余微动长期存在（实测 ke 卡 ~192 / 95s 不入睡）；冻结只
          降能量、不影响姿态（姿态由投影保证）。休眠阈值本身**不放宽**（sleep_eps 照旧），
          靠投影 + 冻结让 ke 正常落到阈值以下（保持态实测 ~2ms/帧 × 30fps，开销可接受）。
        is_asleep() 后外部应停止调用 step()（CPU≈0% 验收项）。
        """
        raise NotImplementedError

    def is_asleep(self) -> bool:
        """kinetic_energy() < ε 且未抓取 → True。休眠中遇任何事件须被 wake() 拉起。"""
        raise NotImplementedError

    def wake(self) -> None:
        """强制唤醒，直到再次收敛才允许休眠（抓取/图标变化/铺上后调用）。

        v6.3：唤醒当帧先把运动门控置 1（先按「运动中」处理），下一 tick 内立即按
        真实速度更新（见 step 的运动门控注）。
        """
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

    # ---- 渲染支持查询（v6「厚度」轮新增；2026-10-08，注释性扩展，不改既有公开语义） ----
    def floor_heights(self) -> np.ndarray:
        """返回 (N,) float32：本 tick 每个粒子下方的地板高度（= 图标隆起；无隆起为 0）。

        渲染层用它算「离地高度」——静止微浮雕贴地满幅、抬起/翻折处衰减为 0。只读。
        """
        raise NotImplementedError

    def ao(self) -> np.ndarray:
        """返回 (N,) float32 ∈ [0,1]：折层间遮蔽近似（被上层布料压住的粒子接近 1）。

        实现 = xy 粗网格「局部顶高 − 自身高度」，随 step() 更新。渲染层按顶点色压暗折缝
        （翻折两层之间的接触阴影）。**纯渲染数据**：不读时钟、不改位置/速度/休眠。
        """
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
      - mouseDown / mouseDragged / mouseUp → sim.grab / drag_to / release(fling=True)；
      - 右键 / Ctrl+左键 → 根视图（RugRootView）menuForEvent_ 弹「毯上上下文菜单」
        （v6.3，菜单本体由 rug.py 注入的 menu_provider 现场构建）——判定与返回约定见
        本类 __init__ 的 v6.3 menu_provider 注记。
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

        注（2026-10-09 v6.3「毯上右键菜单」，注释性说明；契约签名保持向后兼容、
        公开面不变）：实现另带末位 keyword 参数 `menu_provider=None`（rug.py 的
        RugApp._ensure_overlay 注入 `self._make_context_menu`）：
        - menu_provider: 可选回调 `(x, y) -> NSMenu | None`（契约层不 import AppKit，
          返回类型只用文字描述）。x, y = 主屏本地屏幕点 pt（top-left, y-down），即
          右键/Ctrl+左键的点击点；返回 None = 不弹菜单。None（默认）= 本实例无右键菜单。
        - 实现另提供只读属性 `menu_provider`：返回注入的 provider（未注入为 None）。
        - 根视图 RugRootView.menuForEvent_(event) 的契约语义（v6.3）：
          触发 = AppKit 对右键 / Ctrl+左键调 menuForEvent_（毯外一般已被穿透闸门让给
          桌面，故基本只在毯上被调用）。毯上判定同 hitTest_：x, y 由 event 转主屏本地
          点，d = sim.nearest_distance(x, y)；`overlay.edit_mode` 为真时并入手柄距离
          d = min(d, _handle_distance(x, y))；d ≤ GRAB_RADIUS_PT（40pt）= 毯上。
          毯上 → 调 `provider(x, y)`，返回其构建的 NSMenu（每次现建，见 RugApp
          `_make_context_menu` 约定）；毯外 / provider 为 None / sim 缺失 / 未 show →
          返回 None（不弹，走 AppKit 默认行为）；provider 抛异常 → 打一行
          `RUG-WARN 上下文菜单构建失败：...` 后仍返回 None。**任何路径都绝不向
          AppKit 抛异常**。
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
    """常驻状态栏应用（Accessory；仅设置窗期间临时 Regular）+ 总控状态机。

    职责（AppKit 细节见 spike / contract.md §5）：
      - 激活策略（v6.4「Clash 式常驻状态栏」）：平时 ActivationPolicyAccessory
        （**不占 Dock**，只有菜单栏图标）；**设置窗打开期间临时切 Regular**（Dock 图标
        出现、主菜单/⌘Q 可用），关窗回 Accessory。统一切换点 = `_set_dock_visible
        (visible, reason)`（幂等：策略已是目标就不动；切换时打一行 RUG-INFO）；
        `menuSettings_` 打开设置窗前调 `_set_dock_visible(True, reason="设置窗")`，
        SettingsController 在关窗（「关闭」按钮或红点）回调 `_settings_window_closed()`
        → `_set_dock_visible(False, ...)`。理由（2026-10-09 真机教训）：纯菜单栏项在
        多屏 / 菜单栏拥挤（刘海屏）时可能被系统收进**溢出区**（窗口 frame 变 0 高、
        isVisible=False；`_log_status_item_geometry` 会打 RUG-WARN）→ 用户「看不到
        图标、也关不掉」；设置窗期间的 Dock 图标 + ⌘Q 是确定的路。
      - 主菜单（v6.4，`_build_main_menu`）：Dock 图标出现时菜单栏会切成它——应用菜单
        「设置… ⌘,」/「退出桌面毛毯 ⌘Q」+「毯子」子菜单（铺上／掀开 ⌘\、
        调整毯子（大小/角度）⌘E、抚平折痕 ⌘F、重置位置/大小/角度 ⌘⇧R）。纯 Accessory
        时代不需要主菜单；现在必须给出可发现的退出路径，否则用户还是只能去找小图标。
      - NSStatusItem 菜单：
        「铺上 / 掀开 ⌘\」「地毯款式」子菜单（glob assets/rugs/rug-*.png；切换 =
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
      - 启动行为（v6.2 起，打包 .app 后调整；CLI 见 parse_args/main）：
        默认「启动即铺上」（等价旧 --demo）——进主循环前直接 lay_down，双击 .app
        屏幕上立刻出现毯子；旧默认（只驻留菜单栏、屏幕零反馈）被用户判定为
        「打开应用没反应」。`--start-hidden` 恢复旧行为；`--demo` 保留为兼容别名。
        主菜单开关项随之有第三态：掀开后变「摆放回来」（记住的摆放原样放回）。
      - 状态栏项（v6.2）：优先显示 app 图标（打包后取包内 AppIcon.icns，开发时取
        项目 icon.icns）；图标缺失回退文字「Rug」（状态栏项本身必须存在，否则整个
        应用在屏幕上没有任何入口）；带图标时 tooltip「桌面毛毯」。
      - 设置持久化落位（v6.2，实现于 settings_ui.settings_path）：源码运行 = 项目内
        settings.json（开发习惯不变）；打包运行（源码路径含 ".app/Contents/"）=
        ~/Library/Application Support/DesktopRug/settings.json，首次从包内默认播种
        （写 app 包内文件会在升级/重装时丢设置）。
      - 单实例锁（main 层，**非** selftest 模式）：flock 非阻塞占
        ~/Library/Application Support/DesktopRug/instance.lock；拿不到 → 一行 RUG-INFO
        后返回 0（避免出现两个菜单栏项/两张毯子）。selftest 是诊断模式，不加锁。
      - 毯上右键菜单（v6.3）：毯子上右键 / Ctrl+左键弹上下文菜单（overlay 侧判定与
        注入姿势见 RugOverlay 的 menu_provider 注记）。菜单项与 action 约定（所有项的
        target = 本 RugApp 实例、keyEquivalent 均为 ""）：
        1) 第一项 = 状态相关开关（标题由 `_context_toggle_title()` 每次现算），
           action `menuToggleRug_`：非 idle → lift()；idle 且 _saved_placement 非空 →
           place_back()（原样放回）；否则 lay_down()；
        2) 「调整尺寸」子菜单：3 个预设（action `menuSizePreset:`，representedObject =
           "small"/"medium"/"large"，勾选 = 当前 settings.rug_w_frac 与预设宽之差
           < 0.005）+ 分隔符 +「拖拽调整（四角缩放 / 旋转）…」（action `menuEditRug:`，
           未铺上（idle）时 setEnabled_(False)；= 切换编辑模式）；
        3) 「抚平折痕（摊平毯子）」`menuFlattenRug_`（sim.flatten_folds()，位置/大小/
           角度不变）与「重置位置/大小/角度」`menuResetRug_`（回默认摆放 = 屏中心 +
           settings 当前默认尺寸/角度，并清 _saved_placement）；
        4) 分隔符 +「设置…」`menuSettings_`（设置窗）+「退出桌面毛毯」`menuQuit_`（quit()）。

        注（2026-10-09 v6.3「毯上右键菜单」，注释性说明，不改既有公开语义）——
        实现新增（用法见上）：
        - 模块级 `_SIZE_PRESETS`（rug.py）：{"small": (0.34, 0.42),
          "medium": (0.52, 0.62), "large": (0.70, 0.80)}，值 = (宽占屏比, 高占屏比)；
          "medium" 即 settings_ui.DEFAULTS 出厂值（rug_w_frac=0.52 / rug_h_frac=0.62）；
        - `_context_toggle_title(self) -> str`：右键菜单第一项标题，随当前状态现算：
          state ≠ idle → 「掀开毯子（收起）」；idle 且 _saved_placement 非空（掀开过、
          有记住的摆放）→ 「摆放回来」；否则 → 「铺上毯子」；
        - `_make_context_menu(self, x: float, y: float)`：provider 本体，返回 NSMenu
          （契约层不 import AppKit，返回类型只用文字描述）。**每次右键现建**菜单，
          标题/勾选/可用性按当下状态计算，不维护第二份菜单状态；setAutoenablesItems_
          (False)；构建时打一行 `RUG-INFO 毯上右键菜单：... 状态=...`；(x, y) = 主屏
          本地点击点 pt，当前版本不按点位做内容差异（留作后续扩展）——即 provider
          只保证收到毯上点击点，实现方不得假设其它语义；
        - `menuSizePreset_(self, sender) -> None`（action）：读 `sender.
          representedObject()` 取 "small"/"medium"/"large"（非法值回退 "medium"）→ 用
          `_SIZE_PRESETS` 改写 settings 的 rug_w_frac/rug_h_frac **并落盘**
          （Settings.set 即持久化）→ 调 `apply_settings_default_size()`：仅当已铺上
          （state ≠ idle）且不在编辑模式时立刻应用（只改尺寸，位置/角度不打扰），
          否则只改下次铺上的默认值；随后打一行 `RUG-INFO 尺寸预设 ...`。
        其余 action（menuToggleRug_ / menuEditRug_ / menuFlattenRug_ / menuResetRug_ /
        menuSettings_ / menuQuit_）沿用既有行为，无新契约。

        注（2026-10-09 v6.4「常驻状态栏 / 材料记忆」，注释性说明，不改既有公开语义）：
        - 激活策略切换见上（`_set_dock_visible` / `_settings_window_closed` 为实现内部
          方法）；`applicationShouldHandleReopen_hasVisibleWindows_(sender,
          has_visible_windows) -> bool`（点 Dock 图标 / 应用重开）：设置窗面板可见 →
          提到前面；否则调 `menuToggleRug_(None)` 切换毯子；**任何路径都返回 True**、
          内部异常只打 RUG-WARN、绝不向 AppKit 抛出。正常态（Accessory）没有 Dock 图标，
          所以这条基本只在设置窗打开过（Dock 图标在）时触发。
        - **可靠入口** = 「毯上右键菜单」（v6.3）与「设置窗期间的 Dock 图标」：菜单栏项
          可能被 macOS 收进溢出区（见上），实现与文档不得把菜单栏项当作唯一入口。
        - sim 侧「材料记忆 / 放下就记住」的语义见 ClothSim 类 docstring 的 v6.4 注。
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

        注（v6.2）：实现另带第三参 `demo: bool = False`（True = 启动即铺上；CLI 由
        `--start-hidden` 取反传入，故默认 True）——契约签名保持向后兼容、公开面不变。
        """
        raise NotImplementedError

    def run(self) -> int:
        """进入主循环（正常路径不返回）。Ctrl-C 与菜单退出都必须走到 os._exit 兜底。

        v6.2：demo 标志为真时（CLI 默认路径）在 app.run() 前先 lay_down；selftest 走
        探针分支（跳过 IconSensor）。主循环本身不返回，出口一律 os._exit(0)。
        v6.4：启动即 setActivationPolicy_(Accessory)（常驻菜单栏、不占 Dock；打一行
        RUG-INFO「激活策略=Accessory…」）；Dock 图标只在设置窗期间由
        `_set_dock_visible(True, ...)` 临时出现、关窗收回（见类 docstring）。
        """
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
    """解析命令行（v6.2 起三项，返回 argparse.Namespace，字段均为 bool）：

    - `--selftest` → `selftest`：无人值守自检（见 RugApp 类 docstring）；跳过
      IconSensor，且 main 不为它取单实例锁（诊断模式可与常驻实例并存）。
    - `--start-hidden` → `start_hidden`：启动不铺上，只驻留菜单栏（v6.2 前的旧默认）。
    - `--demo` → `demo`：历史选项，保留兼容——v6.2 起「启动即铺上」已是默认行为，
      该开关与不传参等价（不再是唯一入口）。
    """
    raise NotImplementedError


def main(argv: list[str] | None = None) -> int:
    """rug.py 入口（v6.2 顺序）：文件日志落盘（非 TTY 启动，见模块 docstring）→
    parse_args → 非 selftest 时取单实例锁（拿不到 → 一行 RUG-INFO 后返回 0）→
    RugApp(selftest=..., demo=not start_hidden).run()。
    KeyboardInterrupt → 清理后返回 0（或直接 os._exit(0)）。
    """
    raise NotImplementedError
