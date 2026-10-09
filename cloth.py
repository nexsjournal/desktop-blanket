"""cloth.py — 3D 布料模拟（ClothSim）· v4「真翻折」。

纯逻辑模块：只依赖 numpy；无 I/O、无 AppKit、不读时钟（时间信息只来自
step(dt) 传入的 dt），由外部 NSTimer 以 30~60fps 驱动。语义唯一来源：
contracts/interfaces.py 的 ClothSim docstring 与 开发文档 §7.2（v4 注记见 §12.16）。

模型（真 3D 位置 PBD/verlet；v3 的 2.5D 高度场已退役——「每个 (x,y) 只能有一个
高度」在数学上无法对折，见 §12.16 决策记录）：
  1. **3D 位置/速度** (N,3)：z 是离桌面的高度（≥ 地板高度）。重力沿 -z；
     面内 (x,y) 由地面摩擦约束（桌面玩具：毯子不会滑出屏幕）。
  2. **距离约束**：结构 + 剪切，按 (row,col) 棋盘分色拆 8 组全向量化求解；
     外加**结构边硬拉伸上限**（max_stretch×rest，拉力沿织物近似 1:1 传播——
     「抓住一点拖整块毯子」整块跟手的关键，v3 起沿用）。
  3. **地板 = 图标高度场**：z ≥ floor(x,y)，floor = BumpField.heights（无隆起时 0）
     ——图标把毯子顶起、「堆越高包越大」；接触粒子做额外竖直速度吸收（防弹跳）。
  4. **压缩屈曲（折痕/翻折的种子）**：结构边被压短时，把多余长度按系数转成
     **向上的位置偏移**注入两端粒子 → 压缩带自动拱起成褶皱；拱起本身消耗多余
     长度（自限，无需 v3 的塑性场门控）。持续横向拖动会把拱带推过布身 → 真翻折。
  5. **自碰撞**：xy 均匀哈希 + 3D 距离检查，只对抬起粒子（z > 地板+阈值）启用 →
     翻折压到自己身上时两层保持 ≥ collision_sep 间距、不互穿。平面区域零开销。
  6. **抓取**：抓点钉住（只钉 xy——手在桌面上拖，不锁高度）+「整块跟手」助力
     （v5：**只驱动质心**跟随手，无形状记忆、不参与形状——形状完全由材料自身在
     张力/摩擦下的响应决定，这是"拖动出大波浪而不是揉成一团"的关键，见 §12.17）。
  7. **抛掷/铺入**：xy 平移初速度 + 旋转（观感与 v3 一致；z 交给重力自然落回）。
  8. **休眠**：3D 动能 < ε 且未抓取 → is_asleep()（CPU≈0%）；准静态（未抓取、
     无 throw 拉动、平均速度低于阈值）时多用 settle_iterations 轮约束快速安定。
     任何事件（抓取/throw_in/wake）重新拉起。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # 仅类型标注；运行时零依赖 contracts，模块可独立导入
    from contracts.interfaces import BumpField

__all__ = ["ClothSim"]

_CORNERS = ("top_left", "top_right", "bottom_left", "bottom_right")
_DT_MIN = 1.0 / 240.0
_DT_MAX = 0.05


class ClothSim:
    """3D 布料模拟：真 3D 位置 PBD（含自碰撞与翻折）+ 图标隆起 + 抛掷 + 休眠。

    网格 cols×rows 粒子，间距 = width_pt/(cols-1) × height_pt/(rows-1)。
    顶点顺序（vertices()/uv() 共用，行主序）：index = row * cols + col；row 0 = 顶边。
    """

    def __init__(
        self,
        width_pt: float,
        height_pt: float,
        cols: int = 90,
        rows: int = 56,
        bumps: BumpField | None = None,
        *,
        iterations: int = 3,
        grab_iters: int = 3,              # 抓取中每帧的约束轮数（v6.7：拖动的钉子每帧最多移动
                                          # 几十 pt，3 轮追不上 → 钉子甩出尖刺；抓取时多跑几轮
                                          # 让钉点邻域当帧收敛，代价只落在拖动态）
        damping: float = 0.972,
        gravity: float = 1200.0,          # z 向重力（pt/s²）
        friction: float = 6500.0,         # 面内 (x,y) 地面摩擦（pt/s 速度削减；"厚毯"：毯子不轻易
                                          # 滑走 → 手上的位移只能靠材料折叠消化，折痕才会出现；
                                          # v6.1: 4000→5000 增加重量感）
        buckling: float = 0.7,            # 压缩屈曲：多余长度 → 向上位置偏移的系数
                                          # （必须 > 重力每帧 1.33pt 的量级，否则拱起被重力压回）
        buckling_step_cap: float = 2.5,   # 单帧屈曲注入上限（pt；防猛冲打炸）
        buckling_dead_zone: float = 0.4,  # 压缩死区（pt）：滤掉欠收敛的微量压缩
        buckling_min_ratio: float = 0.10, # 压缩超过 rest×该比例才注入（普通拖动只起浅纹）
        buckling_blur_cells: int = 4,     # 屈曲种子的网格空间低通半径（格；σ≈4 格≈50pt）——
                                          # 拱起波长必须远大于格距，否则顶视是"揉纸"碎皱
        fold_cap_z: float = 240.0,         # 屈曲饱和高度（pt）：拱到这么高就不再注入（防能量泵）
        bend_passes: int = 2,             # 每轮距离约束后做几遍弯曲/折痕约束。**不能只做 1 遍**：
                                          # 多对 (i,i+2) 同时塌陷时，一遍里的分组更新会互相抵消
                                          # （独立复核实测：密集折区 ~2% 对仍低于圆角下限、
                                          # 最深只有下限的一半），2 遍把折区贴合度拉回来，代价可忽略
        bend_soft: float = 0.60,          # 弯曲刚度（柔性区双向 (i,i+2) 约束）：波浪圆润、
                                          # 落定后保留（0 = 折纸感且落定就摊平，实测见 §12.16）
        crease_ratio: float = 0.90,       # 折痕圆角下限（= |p_{i+2}-p_i| / (2×间距) 的最小值）。
                                          # 0.90 ↔ 曲率半径 ≈ 1.2 格 ≈ 15pt：**禁止剪纸式锐折**，
                                          # 材料被迫拱起成有体积的圆角折脊（v5 观感关键，见 §12.17）。
                                          # 1.0 = 只允许完全平直（不可用）；0 = 关闭（回到 v4 锐折）
        collision_sep: float = 15.0,      # 自碰撞最小间距（两层翻折的”布厚”；v6.1: 9→12 防穿透）
        collision_iters: int = 5,         # v6.1: 3→5 加强碰撞求解
        collision_relax: float = 0.6,     # 每个碰撞对的推出松弛系数（<1 防多点过冲振荡）
        collision_damp: float = 0.5,      # 碰撞接触粒子的每帧 xy 摩擦缩放（接触非弹性）
        v_max: float = 4000.0,            # 速度护栏（pt/s）
        max_z: float = 400.0,             # 高度护栏（pt）
        sleep_eps: float = 1.0,
        fling_threshold: float = 600.0,
        fling_max: float = 2200.0,
        throw_flight_time: float = 1.1,
        throw_max_speed: float = 2600.0,
        throw_spin: float = 0.0,          # 落定不带斜角
        pull_k: float = 30.0,
        pull_c: float = 11.0,
        pull_timeout: float = 3.0,
        settle_iterations: int = 24,
        settle_move_cap: float = 1.0,     # 静止期每帧约束修正位移上限（pt，防"修正→速度"自激）
        settle_vel_credit: float = 0.3,   # 静止期修正量转成速度的比例（其余视作投影、不计速度）
        max_stretch: float = 1.12,
        strain_passes: int = 5,
        grab_strain_passes: int = 15,     # 抓取中的拉伸上限遍数（v6.7）：拉伸上限是**逐格传播**的
                                          # （每遍把拉力传一格），拖动时每帧只跑 5 遍追不上手速
                                          # → 钉点邻域被拉开成"针"、材料跟不上、折叠手势被削弱
                                          # （实测翻底三角 509→72）。抓取时跑 15 遍（~270pt 传播）
        compression_soft: float = 0.40,   # 压缩方向距离约束刚度比例（"推"要能传出去：手的推挤
                                          # 在毯身上形成**一片**压缩区 → 低频屈曲长成大波浪；
                                          # 0.1 时推挤不外传、0.9 时把折痕推开，实测两端都差）
        grab_friction_scale: float = 0.50,
        body_k: float = 24.0,             # 「整块跟手」助力（只驱动**质心**跟随手移动；越大毯子越
                                          # "飘"（不易出折痕），越小越沉，v6.1: 30→18 增加重量感）
        body_c: float = 14.0,             # 助力阻尼（≈临界阻尼）
        body_hold_speed: float = 60.0,    # 手速门控（pt/s）：**瞬时**手速达到它就给满助力，
                                          # 停手（手速≈0）即归零。
                                          # v6.10 关键修复，见 step 第 2 步注释：手停住时
                                          # `lag` 长期剩 200~340pt，助力成了永不消失的常力 →
                                          # 整块以 ~60pt/s 持续蠕动、折痕反复被揉 = 「一直抖」。
        body_k0_frac: float = 0.35,       # 起手助力比例（v6.6「先聚拢再拖走」：起手只给 22%
                                          # 的助力 → 拉边角先就地起皱聚拢；见 step 第 2 步）
        body_ramp_pt: float = 700.0,      # 助力斜坡：累计手位移达到该值后助力满值（整块跟随）

        # ---- v6.3「大折 / 腾空」（对齐参考视频：毯子被甩起来时是垂挂着折的，不是贴地滑）----
        fling_lift: float = 0.18,         # 抛掷竖直初速度 = 水平速度 × 该系数（0 = 旧行为：只贴地滑）
        fling_lift_max: float = 420.0,    # 抛掷竖直初速度上限（pt/s；≈0.85s 滞空、拱高 ~300pt）
        throw_arc_z: float = 300.0,       # 「铺上」飞入的竖直初速度（pt/s；拱高 ≈ v²/2g ≈ 96pt）
        grab_lift_v0: float = 1400.0,      # 抓点抬升的起始拖动速度（pt/s）：慢拖仍贴地滑
        grab_lift_gain: float = 0.16,     # 抓点目标高度 =（拖动速度 − v0）× 该系数
        grab_lift_max: float = 90.0,     # 抓点抬升上限（pt）
        grab_lift_rate: float = 6.0,      # 抬升/回落平滑速率（1/s）
        pin_leash_pt: float = 220.0,       # 牵引皮带（v6.7）：钉点最多领先局部材料多少 pt
        pin_edge_k: float = 4.0,          # 钉点**相连边**的应变上限（×格距）：用户截图的"尖尖"
                                          # 就是一条被拉到 6 倍的边（实测 6.3×）。正确四邻居
                                          # 拓扑下 3.5 格足以让角部跨过毯身，同时禁止针状长边；
                                          # 折起来是压缩方向，不受此限

        # ---- v6.4「材料记忆」（塑性 rest 长度）：折起来就不会自己摊平 ----
        plastic_rate: float = 0.03,       # 蠕变速率（/tick）：受压边 rest 朝当前距离靠近的速度
        plastic_thresh: float = 0.70,     # 触发阈值：距离 < 0.70×rest0 的边才写入记忆
                                          # （v6.6：0.80 太松——普通拖动也写记忆 → 材料积累永久余量、
                                          #  越拖越皱「揉接在一起」；收紧到只认真折）
                                          # （v6.6：0.80 太松——普通拖动也写记忆 → 材料积累永久余量、
                                          #  越拖越皱「融合在一起」；收紧到只认真折）
        plastic_min_frac: float = 0.85,   # 记忆下限：rest 最短只到 0.85×rest0（v6.6：0.55 留出的永久余量太大，
                                          #  多折几次材料就摊不平了——用户报的「揉接/融合在一起」）
        pose_k: float = 300.0,            # 形状保持弹簧刚度（/s²）：g/k ≈ 4pt 平衡下陷，压不塌折
        pose_delay_ticks: int = 12,       # 松手后延迟多少 tick 再冻结形状（先让自碰撞把层推开）
        pose_hold_ticks: int = 0,         # 形状保持的窗口（tick；0 或负数 = 一直保持到被抓住）
        pose_gain: float = 0.28,          # 形状保持位置投影的每帧增益（0.35 → 平衡下陷≈g·dt/gain≈4pt）

        # ---- v6.9「折痕不再撕成锯齿」（2026-10-09；用户截图里折脊那圈亮锯齿的真根因）----
        sep_max_dz: float = 3.0,          # 退化重叠清理只处理 |Δz| < 该值的对（pt）。
                                          # 为什么：只有 |Δz| 小到高度偏置也分不开时，两层才会
                                          # **逐像素争夺深度** → 顶视看就是锯齿"撕口"。z 差已有
                                          # 十几 pt 的正常折层不需要推（实测旧实现推了 203 对、
                                          # 仍残留 184 对，净效果为零）。
        sep_min_depth: float = 3.0,       # 共面两层的最小深度分离（pt）：修锯齿只需要几 pt 的
                                          # 高度差（正交俯视 + depth_bias=0.90 → 3pt 就给出 2.7
                                          # 个深度单位，远大于 z-fight 阈值）。旧实现一律抬到
                                          # `sep+1=16pt`，实测把普通拖动平白抬高 ~10pt（29→39.6，
                                          # 击穿「普通拖不拎起」守卫）；5.0 仍会到 21.0，故取 3.0。
        clean_interval: int = 8,          # 退化清理的 tick 间隔（原来 30，且只在动作期跑）
        clean_idle_ticks: int = 60,       # 入睡后再维持清理多少帧：互穿是**冻在保持/入睡态**里的
                                          # （实测：入睡那一刻仍有 10 对 Δz<3pt 的贴合被冻住）。
                                          # 落地即等于"保持态每帧清理"，之后彻底静默。
        clean_max_iters: int = 400,       # 单次清理最多迭代几轮（每轮重算候选，直到无贴合对）
    ) -> None:
        """创建布料，静止平铺在主屏本地坐标 (0,0)-(width_pt,height_pt) 的桌面平面上。

        bumps 可为 None（无隆起模式）；后续可随时用 set_bumps() 替换。
        其余 keyword 参数为调优项（含义见模块 docstring）。
        """
        width_pt = float(width_pt)
        height_pt = float(height_pt)
        cols = int(cols)
        rows = int(rows)
        if width_pt <= 0.0 or height_pt <= 0.0:
            raise ValueError("width_pt/height_pt 必须为正")
        if cols < 2 or rows < 2:
            raise ValueError("cols/rows 必须 ≥ 2")

        self.width_pt = width_pt
        self.height_pt = height_pt
        self.cols = cols
        self.rows = rows
        self._n = cols * rows
        sx = width_pt / (cols - 1)
        sy = height_pt / (rows - 1)

        # ---- 调优参数 ----
        self._iterations = max(int(iterations), 1)
        self._grab_iters = max(int(grab_iters), self._iterations)
        self._damping = float(damping)
        self._gravity = float(gravity)
        self._friction = float(friction)
        self._buckling = float(buckling)
        self._buckling_cap = float(buckling_step_cap)
        self._buckling_dead = float(buckling_dead_zone)
        self._buckling_min_ratio = float(buckling_min_ratio)
        self._buckling_blur_cells = max(int(buckling_blur_cells), 0)
        self._fold_cap_z = float(fold_cap_z)
        self._bend_soft = float(np.clip(bend_soft, 0.0, 1.0))
        self._crease_ratio = float(np.clip(crease_ratio, 0.0, 0.999))
        self._bend_passes = max(int(bend_passes), 1)
        self._collision_sep = float(collision_sep)
        self._collision_iters = max(int(collision_iters), 0)
        self._collision_relax = float(np.clip(collision_relax, 0.05, 1.0))
        self._collision_damp = float(np.clip(collision_damp, 0.0, 1.0))
        self._v_max = float(v_max)
        self._max_z = float(max_z)
        self._sleep_eps = float(sleep_eps)
        self._fling_threshold = float(fling_threshold)
        self._fling_max = float(fling_max)
        self._throw_flight_time = float(throw_flight_time)
        self._throw_max_speed = float(throw_max_speed)
        self._throw_spin = float(throw_spin)
        self._pull_k = float(pull_k)
        self._pull_c = float(pull_c)
        self._pull_timeout = float(pull_timeout)
        self._settle_iterations = max(int(settle_iterations), 1)
        self._settle_move_cap = float(settle_move_cap)
        self._settle_vel_credit = float(np.clip(settle_vel_credit, 0.0, 1.0))
        self._max_stretch = float(max_stretch)              # 结构边硬拉伸上限（×rest；≤1 关闭）
        self._strain_passes = max(int(strain_passes), 0)
        self._grab_strain_passes = max(int(grab_strain_passes), self._strain_passes)
        self._compression_soft = float(np.clip(compression_soft, 0.0, 1.0))
        self._grab_friction_scale = float(grab_friction_scale)
        self._body_k = float(body_k)
        self._body_c = float(body_c)
        self._body_hold_speed = max(float(body_hold_speed), 1.0)
        self._hand_speed = 0.0             # 本帧瞬时手速（pt/s；见 step 第 2/7 步）
        self._body_k0_frac = float(np.clip(body_k0_frac, 0.0, 1.0))
        self._body_ramp_pt = max(float(body_ramp_pt), 1.0)
        self._fling_lift = max(float(fling_lift), 0.0)
        self._fling_lift_max = max(float(fling_lift_max), 0.0)
        self._throw_arc_z = max(float(throw_arc_z), 0.0)
        self._grab_lift_v0 = max(float(grab_lift_v0), 0.0)
        self._grab_lift_gain = max(float(grab_lift_gain), 0.0)
        self._grab_lift_max = max(float(grab_lift_max), 0.0)
        self._grab_lift_rate = max(float(grab_lift_rate), 0.0)
        self._grab_nbrs = None             # 抓点邻域（±2 格）粒子下标：牵引皮带用
        self._pin_leash = max(float(pin_leash_pt), 1.0)
        self._pin_edge_k = max(float(pin_edge_k), 1.05)
        self._grab_direct = None
        self._grab_lift = 0.0     # 抓点当前抬升高度（pt；快拖时 > 0 → 毯子被拎起垂挂）
        self._plastic_rate = max(float(plastic_rate), 0.0)
        self._plastic_thresh = float(np.clip(plastic_thresh, 0.05, 1.0))
        self._plastic_min_frac = float(np.clip(plastic_min_frac, 0.05, 1.0))
        self._plastic_grow_thresh = 1.03   # 拉伸方向记忆更新触发比（> 该比例才重记，上限 rest0）
        self._pose_k = max(float(pose_k), 0.0)
        self._pose_z = None                # 形状保持目标高度（None = 自由；capture 时写入）
        self._pose_pending = 0             # 松手后延迟捕获的剩余 tick 数（见 step）
        self._pose_delay = max(int(pose_delay_ticks), 0)
        self._pose_active = False          # 形状保持是否生效（捕获后 True；抓住/重摆/窗口到点 → False）
        self._pose_left = 0                # 窗口倒计时（仅 pose_hold_ticks > 0 时使用）
        self._pose_hold_ticks = int(pose_hold_ticks)
        self._pose_gain = float(np.clip(pose_gain, 0.05, 1.0))
        self._creep_tick = 0
        self._diag_group_from = 4
        self._clean_tick = 0
        self._sep_max_dz = max(float(sep_max_dz), 0.0)
        self._sep_min_depth = max(float(sep_min_depth), 0.5)
        self._clean_interval = max(int(clean_interval), 1)
        self._clean_idle_ticks = max(int(clean_idle_ticks), 0)
        self._clean_max_iters = max(int(clean_max_iters), 1)
        self._clean_idle_left = 0
        self._sep_touch = None            # 本帧被退化清理推过的粒子（step 清零其竖直速度）
        self._clean_events = 0            # 累计处理过的贴合对数（诊断/测试用）
        self._motion_gate = 1.0
        self._support_prev = np.zeros(self._n, dtype=bool)   # 上一帧接触掩码（形状保持用）
        self._motion_eps = 0.01   # 「算运动中」的门限：低于它视为静止（允许进入收尾）
        self._sx = sx
        self._sy = sy

        # ---- 状态：3D 位置 + 3D 速度 ----
        gx = np.arange(cols, dtype=np.float64) * sx
        gy = np.arange(rows, dtype=np.float64) * sy
        gxx, gyy = np.meshgrid(gx, gy)  # (rows, cols)，行主序
        self._pos = np.zeros((self._n, 3), dtype=np.float64)
        self._pos[:, 0] = gxx.ravel()
        self._pos[:, 1] = gyy.ravel()
        self._vel = np.zeros((self._n, 3), dtype=np.float64)

        # ---- 几何输出 buffer（复用，不逐帧重建）----
        self._verts = np.zeros((self._n, 3), dtype=np.float32)
        uu, vv = np.meshgrid(
            np.arange(cols, dtype=np.float64) / (cols - 1),
            np.arange(rows, dtype=np.float64) / (rows - 1),
        )
        self._uv = np.empty((self._n, 2), dtype=np.float32)
        self._uv[:, 0] = uu.ravel()
        self._uv[:, 1] = vv.ravel()
        a = (np.arange(rows - 1)[:, None] * cols + np.arange(cols - 1)[None, :]).astype(np.uint32).ravel()
        quads = np.empty((a.size, 6), dtype=np.uint32)
        quads[:, 0] = a
        quads[:, 1] = a + 1
        quads[:, 2] = a + cols + 1
        quads[:, 3] = a
        quads[:, 4] = a + cols + 1
        quads[:, 5] = a + cols
        self._indices = quads.ravel()

        # ---- 距离约束：结构(H,V) + 剪切(D1,D2)，按棋盘分色拆 8 组 ----
        rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        self._rr = rr.ravel()
        self._cc = cc.ravel()
        self._edge_i0_h = (rr[:, :-1] * cols + cc[:, :-1]).ravel()
        self._edge_par_h = (rr + cc)[:, :-1].ravel() & 1
        self._edge_i0_v = (rr[:-1, :] * cols + cc[:-1, :]).ravel()
        self._edge_par_v = (rr + cc)[:-1, :].ravel() & 1
        self._edge_i0_d = (rr[:-1, :-1] * cols + cc[:-1, :-1]).ravel()
        self._edge_par_d = rr[:-1, :-1].ravel() & 1
        # 压缩屈曲只看结构边（横+竖）
        self._b_i0 = None   # _rebuild_constraints 里建
        self._b_i1 = None
        self._b_rest = None
        self._rebuild_constraints()

        # ---- 自碰撞：xy 哈希运行期工作区 ----
        self._last_move = 0.0   # 上一帧最大粒子位移（准静态收尾的自适应门控）

        # ---- 交互/休眠状态 ----
        self._bumps = bumps
        self._grabbed = False
        self._grab_idx: int | None = None
        self._grab_target = np.zeros(2, dtype=np.float64)
        self._grab_radius = 40.0
        self._grab_anchor = np.zeros(2, dtype=np.float64)
        self._last_target = np.zeros(2, dtype=np.float64)
        self._drag_hist: list[tuple[float, float, float]] = []
        self._pull: tuple[float, float, float] | None = None
        self._asleep = False
        self._ke = 0.0
        self._floor_h = np.zeros(self._n, dtype=np.float64)   # 本 tick 地板高度（= 图标隆起）
        self._ao = np.zeros(self._n, dtype=np.float32)        # 折层间遮蔽（渲染用，见 ao()）

    # ---- 内部：(重)建尺寸相关约束表（编辑模式改尺寸后调用） ----
    def _rebuild_constraints(self) -> None:
        """按当前 width_pt/height_pt 重建 rest 长度相关的约束表（边索引不随尺寸变）。"""
        cols = self.cols
        sx = self.width_pt / (cols - 1)
        sy = self.height_pt / (self.rows - 1)
        self._sx, self._sy = sx, sy
        diag = float(np.hypot(sx, sy))
        i0_h, i0_v, i0_d = self._edge_i0_h, self._edge_i0_v, self._edge_i0_d
        par_h, par_v, par_d = self._edge_par_h, self._edge_par_v, self._edge_par_d
        self._groups = []
        self._structure_groups = []
        self._strain_rest0 = []      # 与 _structure_groups 对齐：原始材料长度（拉伸上限用）
        self._plastic_rests = []     # [(i0, i1, rest数组, rest0, 触发比, 是否折痕瞬时)]
        for p in (0, 1):
            h = i0_h[par_h == p]
            v = i0_v[par_v == p]
            d = i0_d[par_d == p]
            gh = (h, h + 1, np.full(h.size, sx, dtype=np.float64))            # 结构横
            gv = (v, v + cols, np.full(v.size, sy, dtype=np.float64))         # 结构竖
            gd1 = (d, d + cols + 1, np.full(d.size, diag, dtype=np.float64))  # 剪切 ↘
            gd2 = (d + 1, d + cols, np.full(d.size, diag, dtype=np.float64))  # 剪切 ↙
            self._groups.extend([gh, gv, gd1, gd2])
            self._structure_groups.extend([gh, gv])
            self._strain_rest0.extend([sx, sy])   # 与 _structure_groups 对齐（拉伸上限用 rest0）
            for g, r0 in ((gh, sx), (gv, sy), (gd1, diag), (gd2, diag)):
                self._plastic_rests.append((g[0], g[1], g[2], r0, self._plastic_thresh, False))
        self._b_i0 = np.concatenate([i0_h, i0_v])
        self._b_i1 = np.concatenate([i0_h + 1, i0_v + cols])
        self._b_rest = np.concatenate([np.full(i0_h.size, sx), np.full(i0_v.size, sy)])
        # 弯曲刚度：(i, i+2) 软距离约束（rest = 2×间距）——浅波纹被拉直、大折叠保留，
        # 布料不会退化成折纸（v3 的塑性折痕场在 3D 里不需要：折痕由真实几何承载）。
        # v5：同组再加**折痕圆角下限**（crease_min，见 _solve_bend）——禁止锐折，
        # 折痕必须拱成有体积的圆角（观感对齐参考视频，§12.17）。
        rr, cc = np.meshgrid(np.arange(self.rows), np.arange(cols), indexing="ij")
        bh0 = (rr[:, :-2] * cols + cc[:, :-2]).ravel()
        # 分组要沿 (i,i+2) 链**交替**（不是按 c/r 的奇偶）：链上相邻两对共享端点，
        # 同组会让 fancy-index 原地更新互相覆盖、圆角下限形同虚设（2026-10-09 实测）。
        bh_par = (cc[:, :-2] // 2).ravel() & 1
        bv0 = (rr[:-2, :] * cols + cc[:-2, :]).ravel()
        bv_par = (rr[:-2, :] // 2).ravel() & 1
        self._bend_groups = []
        self._crease_min = []
        for p in (0, 1):
            rxh = np.full(int((bh_par == p).sum()), 2.0 * sx, dtype=np.float64)
            rxv = np.full(int((bv_par == p).sum()), 2.0 * sy, dtype=np.float64)
            self._bend_groups.append((bh0[bh_par == p], bh0[bh_par == p] + 2, rxh))
            self._bend_groups.append((bv0[bv_par == p], bv0[bv_par == p] + 2 * cols, rxv))
            # 折痕对（i,i+2）：触发阈值 = 圆角下限附近（进了硬约束区就是真折了，不是浅波），
            # **瞬间**记进 rest（真毯子一折就留痕）；结构边则按 plastic_rate 蠕变。
            crease_trigger = min(self._crease_ratio * 1.01, 1.0)
            self._plastic_rests.append((bh0[bh_par == p], bh0[bh_par == p] + 2, rxh,
                                        2.0 * sx, crease_trigger, True))
            self._plastic_rests.append((bv0[bv_par == p], bv0[bv_par == p] + 2 * cols, rxv,
                                        2.0 * sy, crease_trigger, True))
            self._crease_min.append(2.0 * sx * self._crease_ratio)
            self._crease_min.append(2.0 * sy * self._crease_ratio)
        # 对角折痕（v6.4）：只约束轴向 (i,i+2) 时，**斜着折**会出现 0 半径的锐折——
        # 实测 (2 行 + 2 列) 的对角粒子被折到 0.05pt（肉眼就是"纸片锐折"，且自碰撞的
        # 近邻豁免管不到它，留下互穿）。补两条对角 (i,i±2 格) 的同一个圆角下限：
        # 斜折也必须有体积，观感与互穿一起解决。
        drest = 2.0 * float(np.hypot(sx, sy))
        dmin = drest * self._crease_ratio
        dk1 = (rr[:-2, :-2] * cols + cc[:-2, :-2]).ravel()          # (r,c) → (r+2,c+2)
        dk1_par = ((rr[:-2, :-2] + cc[:-2, :-2]) // 4).ravel() & 1
        dk2 = (rr[:-2, 2:] * cols + cc[:-2, 2:]).ravel()            # (r,c+2) → (r+2,c)
        dk2_par = (((rr[:-2, 2:] - cc[:-2, 2:]) + 4 * cols) // 4).ravel() & 1
        self._diag_group_from = len(self._bend_groups)   # 对角组起始下标（_solve_bend 门控用）
        for p in (0, 1):
            for base, par, step in ((dk1, dk1_par, 2 * cols + 2), (dk2, dk2_par, 2 * cols - 2)):
                a = base[par == p]
                rxd = np.full(a.size, drest, dtype=np.float64)
                self._bend_groups.append((a, a + step, rxd))
                self._plastic_rests.append((a, a + step, rxd, drest, crease_trigger, True))
                self._crease_min.append(dmin)
        self._cell = max(sx, sy)                             # 自碰撞哈希格边长

    # ---- 内部：塑性蠕变（材料记忆——折起来不会自己摊平） ----
    def _plastic_creep(self) -> None:
        """边把 rest 就地朝当前距离改写（经典塑性 rest 长度记忆）。

        为什么需要：理想弹性（rest 恒定）下，压缩/弯曲约束 + 重力会合力把折一点点推回
        平面——真机观感是「折起来后上面那层慢慢溶解、恢复平整」，而参考效果里的毯子
        **有厚度和重量**，折起来就留着。真实织物受压会永久变形，这里用 rest 蠕变近似。

        规则：
        - 压缩方向：进折痕区（`crease_ratio` 圆角下限附近 = 真折了）**当场**记住；
          结构边按压缩深度自动缩放速率（`plastic_rate`）蠕变；下限 `plastic_min_frac×rest0`
          （防越折越小、堆成一团）。
        - 拉伸方向**双向**更新（上限 rest0）：用手把折拖平 = 材料重新屈服成平面，
          毯子不会「记住折痕后拉不平」，也不会越拖越松垮。
        - 只在「手在布上」或「真被揉动」（motion_gate）时写入：落定后形状就地冻结。
        「抚平折痕 / 重置摆放 / 改尺寸」走 `_rebuild_constraints()` 重建 rest → 记忆清零。
        """
        if self._plastic_rate <= 0.0:
            return
        if not (self._grabbed or self._motion_gate > self._motion_eps):
            return
        # 每 3 tick 写一次（塑性是慢过程，观测不到差别）——它是 16 组的向量活，占 ~6.5%
        self._creep_tick += 1
        if self._creep_tick % 3 != 0:
            return
        pos = self._pos
        rate = self._plastic_rate
        min_frac = self._plastic_min_frac
        grow_thresh = self._plastic_grow_thresh
        for i0, i1, rest, r0, thresh, instant in self._plastic_rests:
            d = pos[i1] - pos[i0]
            dist = np.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2])
            floor = r0 * min_frac
            limit = r0 * thresh
            need = dist < limit
            if need.any():
                if instant:
                    rest[need] = np.maximum(dist[need], floor)
                else:
                    target = np.minimum(dist, limit)
                    k = np.clip((r0 - dist) / np.maximum(r0 * (1.0 - thresh), 1e-9), 0.0, 2.0)
                    new = rest + (target - rest) * (rate * k)
                    rest[need] = np.maximum(new[need], floor)
            grown = dist > rest * grow_thresh
            if grown.any():
                rest[grown] = np.minimum(
                    rest[grown] + (dist[grown] - rest[grown]) * rate, r0)

    def _separate_degenerate_overlaps(self, done: np.ndarray | None = None) -> int:
        """清理退化重叠（返回处理对数）：网格上**材料不直连**却几乎重合的两块布。

        事件来源：自碰撞只推**抬起**的粒子（两层都平躺在地板高度、Δz≈0 时完全没有互斥），
        且近邻豁免靠「材料是否直连」判据——斜折的两层正好落在两条规则的缝里，留下
        ~0.05pt 级互穿（**贴在一起的表面深度差≈0 → 逐像素锯齿"撕口"，用户截图里的穿模**）。
        这里按「实际距离 vs 网格距离」判定：实际距离远小于网格距离 = 中间折了、是不同层
        → 把上层沿 z 推到碰撞间距；材料直连（同层，局部压缩/堆叠）跳过。
        向量化（与 _collide 同一套「候选优先」哈希套路），单次 <1ms，可在捕获前与
        运行中周期性调用（见 step）。

        v6.9（2026-10-09，实测驱动）：旧实现把**所有** cross 对都往上推（不看 Δz），
        实测推了 203 对、仍有 184 对残留——因为 z 差已有 13pt 的正常折层本来就不需要推，
        而真正贴合的 15 对（Δz≈0）被重力与地板立刻压回，**净效果为零**。
        三个修正：
        ① 只处理真正会争深度的对（`|Δz| < max_dz`）——正常折层不动，开销与副作用都降一个量级；
        ② 共面时（|Δz|≈0）无法靠高度判断谁在上，改用「离地面高度 + 网格序」定序，
           保证同一对里始终推同一个粒子（否则两层互相推、净位移抵消）；
        ③ 把本帧被推过的粒子记进 `self._sep_touch`，由 `step` 清零其竖直速度
           （非弹性接触；否则"推上去→重力压回→再推"会形成能量累积，实测会把毯子推得轻飘）。

        `done`（可选，长度 n 的 bool）：已经抬过的粒子——调用方（`_settle_overlaps`）
        在迭代之间传进来，保证**同一个粒子在一次清理里最多被抬一次**。没有它时，
        同一粒子会在后续轮次里为**不同**的邻居再次被抬，高度逐轮累加
        （实测把普通拖动凭空顶高 17pt）。
        """
        pos = self._pos
        self._sep_touch = None
        if self._collision_sep <= 0.0:
            return 0
        sep = self._collision_sep
        max_dz = self._sep_max_dz
        cell = self._cell
        keys = np.floor(pos[:, 0] / cell).astype(np.int64) * 1000003 \
            + np.floor(pos[:, 1] / cell).astype(np.int64)
        order = np.argsort(keys, kind="stable")
        sk = keys[order]
        rr, cc = self._rr, self._cc
        sx, sy = self._sx, self._sy
        n = self._n
        idx_all = np.arange(n)
        cand_i, cand_j = [], []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                tgt = np.floor(pos[:, 0] / cell).astype(np.int64) + dx
                tgt = tgt * 1000003 + np.floor(pos[:, 1] / cell).astype(np.int64) + dy
                lo = np.searchsorted(sk, tgt, side="left")
                hi = np.searchsorted(sk, tgt, side="right")
                cnt = hi - lo
                if int(cnt.sum()) == 0:
                    continue
                starts = np.repeat(lo, cnt)
                offs = np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt)
                jj = order[starts + offs]
                ii = np.repeat(idx_all, cnt)
                m = ii < jj
                if m.any():
                    cand_i.append(ii[m])
                    cand_j.append(jj[m])
        if not cand_i:
            return 0
        ii = np.concatenate(cand_i)
        jj = np.concatenate(cand_j)
        # 候选优先：先按真距离筛（廉价），再对候选算网格距离
        d3 = pos[ii] - pos[jj]
        dist = np.sqrt(d3[:, 0] * d3[:, 0] + d3[:, 1] * d3[:, 1] + d3[:, 2] * d3[:, 2])
        cand = np.nonzero((dist < sep) & (dist > 1e-6))[0]
        if cand.size == 0:
            return 0
        dr = np.abs(rr[ii[cand]] - rr[jj[cand]]).astype(np.float64)
        dc = np.abs(cc[ii[cand]] - cc[jj[cand]]).astype(np.float64)
        gdist = np.sqrt(dc * dc * sx * sx + dr * dr * sy * sy)
        cross = dist[cand] <= 0.55 * np.maximum(gdist, 1e-9)
        # 只留「贴在一起、深度会争夺」的对：|Δz| 小到 depth_bias 也分不开
        dz = np.abs(d3[cand][:, 2])
        bad = np.nonzero(cross & (dz < max_dz))[0]
        if bad.size == 0:
            return 0
        a = ii[cand][bad]
        b = jj[cand][bad]
        d = dist[cand][bad]
        if done is not None and done.any():
            keep = ~(done[a] | done[b])
            if not keep.any():
                self._sep_touch = None
                return 0
            a, b, d = a[keep], b[keep], d[keep]
        za, zb = pos[a, 2], pos[b, 2]
        # 谁在上面：离地高度优先；两者都在地板高度（共面、最常见的退化态）时用网格序
        # 定序——同一个粒子始终被推，避免"两层互推、净位移抵消"。
        offa = za - self._floor_h[a]
        offb = zb - self._floor_h[b]
        up = np.where(offa > offb, a, np.where(offb > offa, b,
                      np.where(a >= b, a, b)))
        # 抬到「正好分开」即可：修锯齿只要几 pt 的高度差，抬到布料厚度会把普通拖动
        # 也顶高（实测 +10pt）。已经被自碰撞分开的对（|Δz| 已够大）本来就不在候选里。
        # 抬到「正好分开」即可：修锯齿只要几 pt 的高度差，抬到布料厚度（sep+1=16pt）会把
        # 普通拖动也凭空顶高 ~10pt（实测 29→39.6，击穿「普通拖不拎起」守卫）。
        # 共面时的**真实穿透量** = sep − 实际间距，只有它才值得补（上层本来就该在
        # sep 之上）；否则只补到 min_depth。
        want = np.maximum(self._sep_min_depth, np.where(dz[bad] < 1e-6, sep - d, 0.0))
        lift = want - dz[bad]
        np.maximum(lift, 0.0, out=lift)
        # 同一粒子可能落在多对里：取最大抬升，避免重复累加把它弹飞。
        order = np.argsort(lift, kind="stable")
        up_s = up[order]
        lift_s = lift[order]
        _, last = np.unique(up_s[::-1], return_index=True)
        sel = up_s.size - 1 - last
        pos[up_s[sel], 2] += lift_s[sel]
        touch = np.zeros(self._n, dtype=bool)
        touch[up_s[sel]] = True
        self._sep_touch = touch
        return int(sel.size)

    def _settle_overlaps(self) -> int:
        """迭代清理退化重叠，直到没有 |Δz| < sep_max_dz 的贴合对（或达到迭代上限）。

        为什么需要多轮：一次 `_separate_degenerate_overlaps` 只把**每对的上层**抬到
        `sep+1`，抬起来的粒子会跟**相邻的第三层**贴到一起（三层堆叠），下一轮才轮到它。
        实测折角后需要 ~2 轮收敛；上限 `clean_max_iters` 只是防病态输入，正常远达不到。

        返回本次处理的对数；`self._sep_touch` 保留最后一轮被推过的粒子（调用方清零其
        竖直速度——非弹性接触，防止"推上去→重力压回→再推"的能量累积）。
        """
        total = 0
        done = np.zeros(self._n, dtype=bool)
        for _ in range(self._clean_max_iters):
            n = self._separate_degenerate_overlaps(done)
            if n <= 0:
                self._sep_touch = None
                break
            total += n
            if self._sep_touch is not None:
                done |= self._sep_touch      # 同一粒子本次清理不再重复抬升
        self._clean_events += total
        return total

    def _capture_pose(self) -> None:
        """「放下就记住」：把当前整片形状写进 rest（所有约束组，钳在 [min_frac, 1.0]×rest0），
        并记下抬起粒子的高度作为**形状保持弹簧**的目标（`_pose_z`）。

        为什么单靠 rest 不够：rest 缩短只是把「回弹力」变弱，重力仍会直接压塌折起来的
        形状（实测：rest 记忆后折顶仍从 32pt 塌到 24pt）——真毯子折起来立着是因为材料
        有挺度/厚度。这里再补一条刚度弹簧（远大于重力，见 step 第 4b 步）把抬起的粒子
        按回捕获高度；地板与自碰撞永远优先（在弹簧之后应用）。

        只在**松手且没甩出**时调用；用户再抓住它（grab）时目标清空 → 完全自由、可拖平。
        """
        pos = self._pos
        min_frac = self._plastic_min_frac
        self._settle_overlaps()                   # 别把互穿冻进形状（见该方法 docstring）
        for i0, i1, rest, r0, _thresh, _instant in self._plastic_rests:
            d = pos[i1] - pos[i0]
            dist = np.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2])
            rest[:] = np.clip(dist, r0 * min_frac, r0)
        if self._pose_z is None:
            self._pose_z = np.empty(self._n, dtype=np.float64)
        self._pose_z[:] = pos[:, 2]
        self._pose_active = True
        self._pose_left = max(self._pose_hold_ticks, 0)

    # ---- 内部：约束求解（单组，全向量化，3D；拉伸全刚度 / 压缩软——织物不可推） ----
    def _solve_group(self, i0: np.ndarray, i1: np.ndarray, rest: float) -> None:
        pos = self._pos
        d = pos[i1] - pos[i0]
        dist = np.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2])
        diff = (dist - rest) / np.maximum(dist, 1e-9)
        # 单向：拉伸（diff>0）全额修正；压缩（diff<0）只按 compression_soft 修正——
        # 否则约束会把"卷起来的布料"强行推开摊平，折痕/翻折根本存不住（实测 zmax
        # 拖动中 37pt → 落定 5s 只剩 9.6pt）。
        stiff = np.where(diff > 0.0, 1.0, self._compression_soft)
        corr = d * (0.5 * stiff * diff)[:, None]
        pos[i0] += corr
        pos[i1] -= corr

    def _dist3(self, i0: np.ndarray, i1: np.ndarray) -> np.ndarray:
        d = self._pos[i1] - self._pos[i0]
        return np.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2])

    def _solve_bend(self) -> None:
        """弯曲约束（一次全量，两个分区，互不打架）：

        - **柔性区**（|p_{i+2}-p_i| ≥ crease_min）：经典双向软约束（bend_soft）——
          把大波浪抹向平直、小起伏保持圆润（"厚毯"的板状回弹）。
        - **折痕区**（|p_{i+2}-p_i| < crease_min）：全刚度**推开**到 crease_min——
          这就是"折痕必须有圆角"：材料在此不能再折得更尖，多余长度只能往上拱，
          于是折脊是有体积的圆弧（v4 的锐折会看到纸一样的折线 + 顶视时两侧投影开裂，
          这正是用户报的"太差"的主因之一，见 §12.17）。

        两个分区以 crease_min 为界互斥 → 同一对粒子不会一边被拉、一边被推（否则会在
        折痕处对冲出抖动）。
        """
        w = self._bend_soft
        pos = self._pos
        # 对角组（gi ≥ _diag_group_from）的**全局**门控：整块布完全平铺时才跳过。
        # 不能用「组内最小距离」这种局部门控——它会在「跳过→放松→再进入」之间闪烁，
        # 与形状保持叠成极限环（实测 ke 卡在 192 永不入睡 + 两个回归测试变红）；
        # 全局判据（有没有任何抬升）稳定，只在真正平铺的常态下省这份开销。
        diag_on = bool((pos[:, 2] > self._floor_h + 6.0).any())
        for gi, ((i0, i1, rest), cmin) in enumerate(zip(self._bend_groups, self._crease_min)):
            if gi >= self._diag_group_from and not diag_on:
                continue
            d = pos[i1] - pos[i0]
            dist = np.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2])
            inv = 1.0 / np.maximum(dist, 1e-9)
            # 柔性区：双向软约束（比 rest 长的拉回、比 rest 短的轻推）
            k_soft = np.where(dist >= cmin, 0.5 * w * (dist - rest) * inv, 0.0)
            # 折痕区：全刚度推开到圆角下限（dist < cmin → 修正量为负 = 两端沿弦**分开**；
            # 符号必须与标准 PBD 形式一致：k ∝ (dist - 目标距离)，写反就成了"拉得更紧"，
            # 会主动把布揉成一团——2026-10-08 实测：符号写反时拖动后 bbox 收缩到 0.12）
            k_hard = np.where(dist < cmin, 0.5 * (dist - cmin) * inv, 0.0)
            k = k_soft + k_hard
            pos[i0] += d * k[:, None]
            pos[i1] -= d * k[:, None]

    def _strain_limit(self) -> None:
        """结构边的硬拉伸上限（布料近乎不可伸）：超过 max_stretch×**rest0** 的边两端互相拉回。

        注意用 rest0（原始材料长度）而不是塑性记忆后的 rest：记忆只负责「回弹力弱」
        （距离约束的压缩分支是软的），材料真正的拉伸上限始终是 1.12×原始长度——否则
        折痕被记住后（rest 变短）会被拉伸上限锁死，用户想把折拖平都拖不动。

        与弹性约束的区别：只处理「过长」的边、一遍就消掉超出部分（不做松弛）→
        拉力沿织物近似逐格 1:1 传播（「抓住一点拖整块毯子」整块跟手的关键）。
        组内无共享端点（棋盘分色）→ fancy-index 原地更新安全。
        """
        if self._max_stretch <= 1.0:
            return
        pos = self._pos
        pin = self._grab_idx if self._grabbed else None
        for (i0, i1, _rest), r0 in zip(self._structure_groups, self._strain_rest0):
            d = pos[i1] - pos[i0]
            dist = np.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2])
            excess = dist - r0 * self._max_stretch
            mask = excess > 0.0
            if not mask.any():
                continue
            k = (excess[mask] / np.maximum(dist[mask], 1e-9))[:, None]
            corr = d[mask] * (0.5 * k)
            a0, a1 = i0[mask], i1[mask]
            if pin is not None:
                # 钉点不能动（它被钉在手上）→ 该端修正全部转给自由端。
                m0 = a0 == pin
                m1 = a1 == pin
                if m0.any():
                    pos[a1[m0]] -= corr[m0] * 2.0
                if m1.any():
                    pos[a0[m1]] += corr[m1] * 2.0
                rest_m = ~(m0 | m1)
                if rest_m.any():
                    pos[a0[rest_m]] += corr[rest_m]
                    pos[a1[rest_m]] -= corr[rest_m]
            else:
                pos[a0] += corr
                pos[a1] -= corr

    # ---- 内部：地板高度（= 图标隆起场；无隆起时 0） ----
    def _update_floor(self) -> None:
        if self._bumps is not None:
            self._floor_h = np.asarray(
                self._bumps.heights(self._pos[:, 0], self._pos[:, 1]), dtype=np.float64)
        else:
            self._floor_h.fill(0.0)

    def _apply_floor(self) -> np.ndarray:
        """z ≥ 地板高度（硬约束）；返回接触掩码（供速度吸收用）。"""
        pos = self._pos
        below = pos[:, 2] < self._floor_h
        if below.any():
            pos[below, 2] = self._floor_h[below]
        return pos[:, 2] <= self._floor_h + 0.8

    # ---- 内部：压缩屈曲（压缩量 → 高度；低频种子 + 按局部高度饱和） ----
    def _box1d(self, a: np.ndarray, r: int, axis: int) -> np.ndarray:
        """沿 axis 的宽度 (2r+1) 盒式滤波（cumsum，边缘 clamp）——网格空间低通用。"""
        if r <= 0:
            return a
        pad = np.pad(a, [(r + 1, r) if i == axis else (0, 0) for i in range(a.ndim)],
                     mode="edge")
        c = np.cumsum(pad, axis=axis)
        n = a.shape[axis]
        lo = c.take(np.arange(0, n), axis=axis)
        hi = c.take(np.arange(2 * r + 1, n + 2 * r + 1), axis=axis)
        return (hi - lo) / float(2 * r + 1)

    # ---- 渲染支持：折层间遮蔽（v6「厚度」轮，只读、不参与动力学） ----
    def _update_ao(self) -> None:
        """xy 粗网格「局部顶高 − 自身高度」→ 折层遮蔽近似（写 self._ao）。

        用于渲染层的顶点色压暗：翻折时上层布压住下层 → 下层在折缝处变暗，
        两层之间才有「厚度/接触阴影」。只读几何、不改位置/速度 → 不影响休眠与稳定性。
        """
        pos = self._pos
        cell = 24.0
        x0 = float(pos[:, 0].min()) - cell
        y0 = float(pos[:, 1].min()) - cell
        gx = max(int((float(pos[:, 0].max()) - x0) / cell) + 2, 3)
        gy = max(int((float(pos[:, 1].max()) - y0) / cell) + 2, 3)
        fx = np.clip((pos[:, 0] - x0) / cell, 0.0, gx - 1.001)
        fy = np.clip((pos[:, 1] - y0) / cell, 0.0, gy - 1.001)
        ix = fx.astype(np.int64)
        iy = fy.astype(np.int64)
        # 用**离地高度**（z − 地板）而不是绝对高度：趴在图标隆包上的布离地≈0 →
        # 不会在隆包四周被误判成"被压住"（否则每个图标周围会出现一圈假暗影）。
        off = pos[:, 2] - self._floor_h
        top = np.zeros(gx * gy, dtype=np.float64)
        np.maximum.at(top, iy * gx + ix, off)                # 每格最高（离地）点
        t = top.reshape(gy, gx)
        pad = np.full((gy + 2, gx + 2), -1e9, dtype=np.float64)
        pad[1:-1, 1:-1] = t
        tp = t.copy()                                        # 3×3 局部顶高（不跨边界环绕）
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                np.maximum(tp, pad[dy:dy + gy, dx:dx + gx], out=tp)
        # 双线性采样（防粗网格边界在顶点色上出现条带）
        wx = fx - ix
        wy = fy - iy
        i1 = np.minimum(ix + 1, gx - 1)
        j1 = np.minimum(iy + 1, gy - 1)
        h = (tp[iy, ix] * (1.0 - wx) * (1.0 - wy) + tp[iy, i1] * wx * (1.0 - wy)
             + tp[j1, ix] * (1.0 - wx) * wy + tp[j1, i1] * wx * wy)
        np.clip((h - off - 3.0) / 14.0, 0.0, 1.0, out=self._ao)

    def _buckling_inject(self) -> None:
        """压缩量 → 向上的位置偏移，**在网格空间先低通**（v5 观感关键修正）。

        v4 逐粒子注入 → 拱起的波长 = 网格尺度（~13pt）→ 顶视是一层碎皱（"揉过的纸"）。
        真实厚毯的屈曲（Euler）波长是百 pt 量级：低通后的种子只长出**大而光滑**的拱，
        这才是参考视频里"厚毯被推起来"的观感；材料仍在拱弧上自然铺开（弧长守恒）。
        """
        if self._motion_gate <= self._motion_eps:
            return
        dist = self._dist3(self._b_i0, self._b_i1)
        thr = np.maximum(self._buckling_dead, self._buckling_min_ratio * self._b_rest)
        excess = np.maximum(self._b_rest - dist - thr, 0.0) * 0.5
        if not excess.any():
            return
        # 每粒子当前承担的多余长度（两端各半）
        acc = np.bincount(self._b_i0, weights=excess, minlength=self._n)
        acc += np.bincount(self._b_i1, weights=excess, minlength=self._n)
        # 低频种子（两轴盒式滤波 ×2 遍 ≈ 高斯 σ≈4 格 ≈ 50pt）
        g = acc.reshape(self.rows, self.cols)
        r = self._buckling_blur_cells
        for _ in range(2):
            g = self._box1d(self._box1d(g, r, 0), r, 1)
        # 局部高度饱和门：已经拱起来的材料不再注入（多余长度已转为弧长、压缩自会解除）。
        # 这是「不成为能量泵」的关键——稳定褶皱的注入被门关掉，quasi-static 时
        # motion_gate 再兜底归零（否则实测 ke 卡 10^4 永不入睡）。
        gate = np.clip(1.0 - self._pos[:, 2] / self._fold_cap_z, 0.0, 1.0)
        dz = np.minimum(g.ravel() * self._buckling, self._buckling_cap) * gate
        self._pos[:, 2] += dz * self._motion_gate

    # ---- 内部：自碰撞（xy 哈希 + 3D 距离；只迭代抬起粒子） ----
    def _collide(self) -> np.ndarray:
        """返回本帧参与过碰撞的粒子掩码（调用方对其做接触耗散）。"""
        pos = self._pos
        touched = np.zeros(self._n, dtype=bool)
        if self._collision_iters <= 0 or self._collision_sep <= 0.0:
            return touched
        # v6.1: 抬起阈值 0.3→0.8pt，更早激活碰撞检测（防折叠初期穿透）
        lifted = np.nonzero(pos[:, 2] > self._floor_h + 0.8)[0]
        if lifted.size == 0:
            return touched
        cell = self._cell
        # 全体粒子按 (x,y) 哈希（升序键 + searchsorted 取邻居区段，无 Python 逐粒子循环）
        keys = np.floor(pos[:, 0] / cell).astype(np.int64) * 1000003 \
            + np.floor(pos[:, 1] / cell).astype(np.int64)
        order = np.argsort(keys, kind="stable")
        skeys = keys[order]
        rr, cc = self._rr, self._cc
        sep = self._collision_sep
        relax = self._collision_relax
        for _ in range(self._collision_iters):
            corr = np.zeros_like(pos)
            moved = False
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    tgt = np.floor(pos[lifted, 0] / cell).astype(np.int64) + dx
                    tgt = tgt * 1000003 \
                        + np.floor(pos[lifted, 1] / cell).astype(np.int64) + dy
                    lo = np.searchsorted(skeys, tgt, side="left")
                    hi = np.searchsorted(skeys, tgt, side="right")
                    cnt = hi - lo
                    tot = int(cnt.sum())
                    if tot == 0:
                        continue
                    starts = np.repeat(lo, cnt)
                    offs = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
                    jj = order[starts + offs]
                    ii = np.repeat(lifted, cnt)
                    d3 = pos[ii] - pos[jj]
                    dist = np.sqrt(d3[:, 0] * d3[:, 0] + d3[:, 1] * d3[:, 1] + d3[:, 2] * d3[:, 2])
                    # 「近邻豁免」= 材料**直连**（同层），不是「网格 ≤2 格」：
                    # 网格近邻但实际距离远小于网格距离 = 中间折了 → **不同层**，必须互斥。
                    # 2026-10-09 实测：斜折时 (2,2) 对角两块布会并到 0.05pt，旧阈值把它当近邻
                    # 放过 → 贴在一起的表面深度差≈0 → 逐像素锯齿"撕口"（用户截图里的穿模）。
                    # 性能：**先在候选（真的比 sep 近）里筛**，只对候选做网格距离判定——
                    # 对所有对都算网格距离会让开销翻倍（实测 23s→53s）。
                    cand = np.nonzero((dist < sep) & (dist > 1e-6))[0]
                    if cand.size == 0:
                        continue
                    gdx = (cc[ii[cand]] - cc[jj[cand]]).astype(np.float64) * self._sx
                    gdy = (rr[ii[cand]] - rr[jj[cand]]).astype(np.float64) * self._sy
                    gdist = np.sqrt(gdx * gdx + gdy * gdy)
                    straight = dist[cand] > 0.55 * np.maximum(gdist, 1e-9)
                    idx = cand[~straight]
                    if idx.size == 0:
                        continue
                    k = (relax * (sep - dist[idx]) / dist[idx])[:, None]
                    push = d3[idx] * k
                    # 只推「自己」（i 是抬起粒子）：完整解出自己的侵入量 → 上层粒子
                    # 在重力下每帧被顶回，净位移 0、不振荡也不持续下沉。
                    np.add.at(corr, ii[idx], push)
                    touched[ii[idx]] = True
                    moved = True
            if not moved:
                break
            pos += corr
            np.clip(pos[:, 2], 0.0, self._max_z, out=pos[:, 2])
        return touched

    # ---- 交互（由 RugOverlay 的鼠标事件驱动，主线程） ----
    def grab(self, x: float, y: float, radius: float = 40.0) -> bool:
        """尝试在 (x, y)（主屏本地 pt）抓取布料：最近粒子的 **xy 平面**距离 ≤ radius 即命中。

        已抓取时再次调用视为重抓（更新抓点）。未命中返回 False 且不改变任何状态。
        """
        x = float(x)
        y = float(y)
        radius = float(radius)
        d2 = (self._pos[:, 0] - x) ** 2 + (self._pos[:, 1] - y) ** 2
        idx = int(np.argmin(d2))
        if d2[idx] > radius * radius:
            return False  # 未命中：不改变任何状态
        self._grabbed = True
        self._grab_idx = idx
        self._grab_target = np.array([x, y], dtype=np.float64)
        self._grab_radius = radius
        self._last_target = self._grab_target.copy()
        self._drag_hist = []
        # 整块助力基准（v5）：只记抓取瞬间的**质心**与手位置。拖动时助力只驱动质心
        # 跟上「手累计位移」——**不做形状记忆**（v4 把每颗粒子往「原形状刚性平移后」
        # 的位置拉：既是形状恢复力（抹平褶皱），又在大位移时把整布拉成橡皮膜
        # （拉伸 30% → 松手回弹 → 材料无去处 → 堆成一团，= 用户报的"太差"主因）。
        # 抓点邻域（±2 格，含自身）：牵引皮带的参考系（见 _pin_grab）
        _r, _c = divmod(idx, self.cols)
        _rr = np.arange(max(_r - 2, 0), min(_r + 3, self.rows))
        _cc = np.arange(max(_c - 2, 0), min(_c + 3, self.cols))
        _gi, _gj = np.meshgrid(_rr, _cc, indexing='ij')
        self._grab_nbrs = (_gi * self.cols + _gj).ravel()
        # 用行列坐标找四邻居。直接对线性下标加减 1 会在每行边界把
        # (row, 0) 错连到上一行末列，抓角时等价于凭空拉出一条跨行长边，
        # 这正是「鼠标一点就冒出长尖刺」的根因之一。
        _direct = []
        for _dr, _dc in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            _nr, _nc = _r + _dr, _c + _dc
            if 0 <= _nr < self.rows and 0 <= _nc < self.cols:
                _direct.append(_nr * self.cols + _nc)
        self._grab_direct = np.asarray(_direct, dtype=np.int64)   # 直接相连的邻居（边应变用）
        self._grab_center0 = self._pos[:, :2].mean(axis=0).copy()
        self._grab_anchor = self._grab_target.copy()
        self._grab_lift = 0.0
        self._pose_z = None       # 手一碰就自由（形状保持关闭，可随意拖平/重折）
        self._pose_active = False
        self._pose_pending = 0
        self.wake()
        return True

    def drag_to(self, x: float, y: float) -> None:
        """把抓点目标移动到 (x, y)（主屏本地 pt）。未抓取时为 no-op。"""
        if not self._grabbed:
            return
        self._grab_target = np.array([float(x), float(y)], dtype=np.float64)

    def _pin_grab(self) -> None:
        """把抓点钉到目标位置——带**牵引皮带**：钉点不许跑在局部材料前面太远。

        z 不钉 0 很重要：抓点可能本来就落在折层上（z > 0），钉到地板会把它按下去。
        皮带（v6.7）：原来钉点被**精确**钉在手的位置上，手一动材料跟不上（求解器每帧
        3 轮 + 拉伸上限是硬约束）→ 钉点甩出一根几百 pt 的**尖刺**（用户截图：拖动时
        「有个尖尖」）。现在钉点最多领先局部材料 `pin_leash_pt`（≈4.5 格），超出部分
        回拉——手再快，毯子也只是「跟不上」（这就是重量感），不会再长出一根针。
        """
        i = self._grab_idx
        want = np.empty(3, dtype=np.float64)
        want[0] = self._grab_target[0]
        want[1] = self._grab_target[1]
        if self._grab_lift > 0.0:
            want[2] = max(float(self._floor_h[i]), self._grab_lift)
        else:
            want[2] = self._pos[i, 2]
        nbr = self._grab_nbrs
        if nbr is not None and nbr.size > 0:
            # 皮带只限**平面**：针/尖刺是平面现象；若把 3D 距离一起钳，抬升的竖直分量
            # 会被一起缩掉（实测：猛拽 80pt/帧时抓点 z 直接变 0，抬升功能失效）。
            local = self._pos[nbr].mean(axis=0)
            dx = float(want[0] - local[0])
            dy = float(want[1] - local[1])
            dist = float(np.hypot(dx, dy))
            if dist > self._pin_leash:
                sc = self._pin_leash / dist
                want[0] = local[0] + dx * sc
                want[1] = local[1] + dy * sc
        # 局部边应变上限：钉点与**直接相连**邻居之间的边不许超过 pin_edge_k×格距。
        # 用户截图里的"尖尖"就是一条被拉到 6 倍的边（实测 6.3×）；折起来是压缩方向、
        # 不受这条限制，所以折叠手势完全不受影响（实测折保持不变）。
        direct = self._grab_direct
        if direct is not None and direct.size > 0:
            # 逐条投影到「抓点-邻居」的最大允许边长内。旧实现只减去所有
            # 违规边的平均量，邻居位置差异较大时仍会留下 5~6 倍网格边，
            # 最终画面就会出现针状三角。重复几轮是很小的局部约束求解，
            # 但能保证每次写回抓点前都不会重新制造长边。
            lim = self._pin_edge_k * max(self._sx, self._sy)
            for _ in range(8):
                changed = False
                for _j in direct:
                    delta = want[:2] - self._pos[_j, :2]
                    dd = float(np.hypot(delta[0], delta[1]))
                    if dd <= lim:
                        continue
                    if dd > 1e-9:
                        want[:2] = self._pos[_j, :2] + delta * (lim / dd)
                    else:
                        want[:2] = self._pos[_j, :2]
                    changed = True
                if not changed:
                    break
        self._pos[i] = want

    def _update_grab_lift(self, dtc: float) -> None:
        """快拖时抓点离开桌面：目标高度随拖动速度上升，指数平滑跟进/回落。

        参考效果里「拖着毯子走」时毯子是半悬着垂挂出大折的，不是平贴着桌面滑。
        慢拖（< grab_lift_v0）仍保持贴地滑的旧手感，快拖才把它拎起来。
        """
        target = 0.0
        hist = self._drag_hist
        if len(hist) >= 2:
            dt_sum = sum(e[2] for e in hist)
            if dt_sum > 1e-6:
                v = float(np.hypot(hist[-1][0] - hist[0][0], hist[-1][1] - hist[0][1])) / dt_sum
                target = float(np.clip((v - self._grab_lift_v0) * self._grab_lift_gain,
                                       0.0, self._grab_lift_max))
        k = min(1.0, self._grab_lift_rate * dtc)
        self._grab_lift += (target - self._grab_lift) * k

    def release(self, fling: bool = True) -> None:
        """释放抓点。重复调用幂等。

        fling=True 且最近若干帧平均速度超阈值时，给全体粒子注入 **xy 平移初速度 +
        角速度 + 竖直初速度**（抛掷：毯子腾空飞出去，空中靠重力垂挂出大折，落地靠
        非弹性接触停住）。竖直分量 v6.3 新增：抓点附近抬得高、远端滞后 → 空中自然
        垂挂；fling_lift=0 时回到旧行为（只贴地滑）。
        """
        if not self._grabbed:
            return
        flung = False
        if fling and len(self._drag_hist) >= 2:
            hist = self._drag_hist
            dt_sum = sum(e[2] for e in hist)
            if dt_sum > 1e-6:
                v = np.array(
                    [hist[-1][0] - hist[0][0], hist[-1][1] - hist[0][1]],
                    dtype=np.float64,
                ) / dt_sum
                speed = float(np.hypot(v[0], v[1]))
                if speed > self._fling_threshold:
                    if speed > self._fling_max:
                        v *= self._fling_max / speed
                    center = self._pos[:, :2].mean(axis=0)
                    r = self._pos[self._grab_idx, :2] - center
                    cross = r[0] * v[1] - r[1] * v[0]
                    omega = float(np.clip(0.5 * cross / (float(r @ r) + 4.0e4), -6.0, 6.0))
                    rel = self._pos[:, :2] - center
                    self._vel[:, 0] += v[0] - omega * rel[:, 1]
                    self._vel[:, 1] += v[1] + omega * rel[:, 0]
                    vz = min(speed * self._fling_lift, self._fling_lift_max)
                    if vz > 0.0:
                        # 抓点附近抬得高（"拎起来甩"），最远端 0.45 倍 → 空中垂挂而非平移
                        gi = self._grab_idx
                        d = np.hypot(self._pos[:, 0] - self._pos[gi, 0],
                                     self._pos[:, 1] - self._pos[gi, 1])
                        dmax = float(d.max())
                        w = 1.0 - 0.55 * (d / dmax) if dmax > 1e-6 else np.ones(self._n)
                        self._vel[:, 2] = np.maximum(self._vel[:, 2], 0.0) + vz * w
                    flung = True
        # 「放下就记住」（用户：折起来之后老是自动恢复/变得平整）：没甩出时**延迟**捕获形状
        # （_pose_delay 个 tick，先让自碰撞把层推开/落定一点），之后 rest + 高度目标一起冻结；
        # 甩出去则不捕获（空中姿态不留）。
        if not flung:
            self._pose_pending = self._pose_delay
        else:
            self._pose_z = None
            self._pose_active = False
            self._pose_pending = 0
        self._grabbed = False
        self._grab_idx = None
        self._grab_nbrs = None
        self._grab_direct = None
        self._grab_center0 = None
        self._grab_lift = 0.0
        self._drag_hist = []
        self.wake()

    def throw_in(
        self,
        target_center: tuple[float, float],
        from_corner: str = "top_left",
    ) -> None:
        """「铺上」动画入口：毯子从屏幕一角飞入并落在 target_center（xy 平面内的目标；
        v6.3 起带一条竖直弧线：拱高 ≈ throw_arc_z²/2g，落地由非弹性接触接管不弹跳）。

        from_corner: "top_left" | "top_right" | "bottom_left" | "bottom_right"；
        非法值抛 ValueError。整布刚体平移到对应角落（屏外、保持形状），注入朝
        target_center 的初速度 + 旋转，之后由 step() 驱动（闭环拉动收尾）。
        """
        if from_corner not in _CORNERS:
            raise ValueError(
                f"from_corner 必须是 {_CORNERS} 之一，得到 {from_corner!r}"
            )
        tx = float(target_center[0])
        ty = float(target_center[1])
        w, h = self.width_pt, self.height_pt
        m = 60.0  # 屏外余量：整布（保持原尺寸）完全移出屏幕
        anchors = {
            "top_left": (-0.5 * w - m, -0.5 * h - m),
            "top_right": (1.5 * w + m, -0.5 * h - m),
            "bottom_left": (-0.5 * w - m, 1.5 * h + m),
            "bottom_right": (1.5 * w + m, 1.5 * h + m),
        }
        anchor = np.array(anchors[from_corner], dtype=np.float64)
        c2 = self._pos[:, :2].mean(axis=0)
        self._pos[:, 0] += anchor[0] - c2[0]
        self._pos[:, 1] += anchor[1] - c2[1]
        self._pos[:, 2] = 0.0
        d = np.array([tx, ty], dtype=np.float64) - anchor
        dist = float(np.hypot(d[0], d[1]))
        if dist > 1e-9:
            speed = min(dist / self._throw_flight_time, self._throw_max_speed)
            self._vel[:, 0] = (d[0] / dist) * speed
            self._vel[:, 1] = (d[1] / dist) * speed
        else:
            self._vel[:, :2] = 0.0
        # v6.3：飞入带一条弧线（拱高 ≈ v²/2g；throw_arc_z=0 时回到旧行为，z 始终贴地）
        self._vel[:, 2] = self._throw_arc_z
        sign = 1.0 if from_corner in ("top_left", "bottom_right") else -1.0
        rel = self._pos[:, :2] - anchor
        self._vel[:, 0] += -sign * self._throw_spin * rel[:, 1]
        self._vel[:, 1] += sign * self._throw_spin * rel[:, 0]
        self._grabbed = False
        self._grab_idx = None
        self._drag_hist = []
        self._pull = (tx, ty, 0.0)
        self.wake()

    # ---- 步进与休眠 ----
    def step(self, dt: float) -> None:
        """推进一个 tick；dt 单位秒（典型 1/30~1/60），内部 clamp 到 [1/240, 0.05]。

        一个 tick：地板高度更新 → 阻尼/跟手/牵引 → 重力 → 积分 → 抓点钉住 →
        压缩屈曲 → 距离约束（运动 3 轮 / 准静态 settle 轮）→ 拉伸上限 → 地板约束 →
        自碰撞 → 速度回写 + 摩擦/接触吸收 → 休眠判定。is_asleep() 后调用为 no-op。
        """
        if self._asleep:
            # 入睡不等于几何已经正确：折起来的互穿是**冻在入睡态**里的（实测入睡那一刻
            # 仍有 10 对 |Δz|<3pt 的贴合）。所以入睡后再维持 `clean_idle_ticks` 帧只跑
            # 退化清理（不积分、不耗散），把贴合推开就彻底静默——之后 step 仍是 no-op。
            if self._clean_idle_left > 0:
                self._settle_overlaps()
                self._clean_idle_left -= 1
            return
        dtc = min(max(float(dt), _DT_MIN), _DT_MAX)
        pos = self._pos
        vel = self._vel

        self._update_floor()

        # 1) 动量阻尼（分级安定：ke 越小越强，加速收敛入睡）
        ke_now = float((vel * vel).sum())
        f = self._damping
        if not self._grabbed and ke_now < 20.0:
            f = 0.80
        elif not self._grabbed and ke_now < 200.0:
            f = 0.90
        vel *= f

        # 2) 抓取中「整块跟手」助力（v5：**只驱动质心**，无形状记忆、无内部拉力——
        #    拖动中的褶皱/波浪由手钉点 + 材料张力 + 地面摩擦自然产生，
        #    助力只负责"整块毯子跟着手走"的手感，不得参与形状）
        if self._grabbed and self._grab_center0 is not None:
            hand_delta = self._grab_target - self._grab_anchor
            shift = pos[:, :2].mean(axis=0) - self._grab_center0
            lag = hand_delta - shift                       # 质心还差多少没跟上
            vm = vel[:, :2].mean(axis=0)
            # v6.6「先聚拢、再拖走」：助力随**累计手位移**从 body_k0 斜坡到 body_k。
            # 为什么：恒定助力要么太松（拉边角时整块几乎不动、像被粘住），要么太紧
            # （整块当刚性片滑动 = 用户报的"没有重量感"）。参考视频的节奏是——先局部
            # 起皱/聚拢（材料在摩擦下就地堆起来），拉得越久整块越跟着走（毯子被"拖走"）。
            # 实测（拉边角 611pt）：定 k=110 时整块滑动比 0.52（像薄片）；斜坡模型下
            # 起手 ~0.2 倍助力（镜 2.3× 的局部聚拢），拉到 ~500pt 后助力满值（整块跟随）。
            pull = float(np.hypot(hand_delta[0], hand_delta[1]))
            ramp = min(pull / max(self._body_ramp_pt, 1e-9), 1.0)
            k_eff = self._body_k * (self._body_k0_frac + (1.0 - self._body_k0_frac) * ramp)
            # v6.10 根因（2026-10-09 实测）：手**停住**时这个助力仍在持续泵能量。
            # 为什么：助力的目标是「质心应该跟上**全部**手位移」，但抓点被牵引皮带 +
            # 摩擦锚住，质心最多只能走到 ~850/1050 → `lag` 永远剩 200~340pt。于是
            # `k_eff*lag` 变成一个**永不消失的常力**：手停着不动，整块仍以 ~60pt/s 慢速
            # 蠕动（实测 20 帧里质心漂 707→851pt），折痕区被反复揉动、永不收敛
            # （asleep 恒 False）——这就是用户看到的「折痕位置一直抖、像在融合」。
            # 修法：`lag` 加**死区**。手一动 lag 立刻超阈值 → 助力照常（"拖走"手感不变）；
            # 手一停 lag 掉进死区 → 助力归零 → 毯子能真正安定（可入睡）。
            # v6.10 关键修复：助力必须**只由「手在动」驱动**，并给质心速度另加阻尼。
            # 为什么（2026-10-09 实测）：助力的目标是「质心跟上**全部**手位移」，但抓点被
            # 牵引皮带 + 摩擦锚住，质心最多只能走到 850/1050 → `lag` 永远剩 200~340pt。
            # 于是 `k_eff*lag` 成了**永不消失的常力**：手停着不动，整块仍以 ~60pt/s 缓慢蠕动
            # （实测 40 帧里质心又漂了 266pt、lag 从 357 掉到 90 才停），折痕区被反复揉动、
            # 永不收敛（asleep 恒 False）——这就是「折痕位置一直抖、像在融合」。
            # 死区（lag）治不了它（lag 会一路漂到死区内，漂移照样发生）；正确的开关是**手速**：
            # ⚠️ 必须用**瞬时手速**（本帧鼠标位移/dt），不能用累计位移：累计量在手停住
            # 之后永远大于 0（实测 hand_delta 恒 1050pt）→ 门控恒开、等于没修。
            gain = min(self._hand_speed / self._body_hold_speed, 1.0) \
                if self._hand_speed < 1e8 else 1.0
            ax = gain * (k_eff * lag[0]) - self._body_c * vm[0]
            ay = gain * (k_eff * lag[1]) - self._body_c * vm[1]
            vel[:, 0] += ax * dtc
            vel[:, 1] += ay * dtc

        # 3) throw_in 闭环拉动（xy；临界阻尼防过冲）
        if self._pull is not None:
            tx, ty, elapsed = self._pull
            cm = pos[:, :2].mean(axis=0)
            vm = vel[:, :2].mean(axis=0)
            vel[:, 0] += (self._pull_k * (tx - cm[0]) - self._pull_c * vm[0]) * dtc
            vel[:, 1] += (self._pull_k * (ty - cm[1]) - self._pull_c * vm[1]) * dtc
            elapsed += dtc
            if elapsed > self._pull_timeout or np.hypot(tx - cm[0], ty - cm[1]) < 15.0:
                self._pull = None
            else:
                self._pull = (tx, ty, elapsed)

        # 4) 重力（-z）+ 速度护栏 + verlet/PBD 积分 + 运动门控
        grav_dv = self._gravity * dtc
        vel[:, 2] -= grav_dv
        sp = np.sqrt((vel * vel).sum(axis=1))
        over = sp > self._v_max
        if over.any():
            vel[over] *= (self._v_max / sp[over])[:, None]
        # 运动门控：xy 平均速度 + 竖直平均速度。竖直必须计入——否则「腾空抛掷/垂挂」
        # 这种 xy 静止、纯 z 运动的姿态会被判成 quasi-static，被 settle_move_cap 把
        # 每帧位移钳到 1pt（实测：注入 vz=500 只升到 18pt 就被按回，飞不起来）。
        # 但要**减掉本帧刚加的重力增量**：接触粒子在上一帧末尾已被回写清零，剩下的
        # 才是真实竖直运动；不减掉则每颗粒子都带 -g·dt≈-40pt/s 的假运动，门控被钉在
        # 0.195 → 永不进入收尾（实测入睡从 2.9s 拖到 17~36s）。
        # 也**不能**挪到帧尾用回写速度：那样「约束修正 → 速度 → 门控」自激，破坏收敛。
        v_xy = float(np.hypot(vel[:, 0], vel[:, 1]).mean())
        v_z = float(np.abs(vel[:, 2] + grav_dv).mean())
        self._motion_gate = float(min(max(v_xy / 80.0, v_z / 200.0, 0.0), 1.0))
        # 4b) 「放下就记住」：延迟捕获（先让自碰撞把层推开）+ 保持窗口倒计时
        #     （保持本身在 7d 用**位置投影**实现，不动速度 → 不阻塞入睡）
        if not self._grabbed and self._pose_pending > 0:
            self._pose_pending -= 1
            if self._pose_pending == 0:
                self._capture_pose()
        if (self._pose_active and not self._grabbed and self._pose_hold_ticks > 0
                and self._pose_left > 0):
            self._pose_left -= 1
            if self._pose_left == 0:
                self._pose_active = False   # 窗口到点：形状交给 rest 记忆
        pos_old = pos.copy()
        pos += vel * dtc

        # 5) 抓点钉住（xy 始终钉；快拖时抓点抬离桌面 → 毯子被拎起垂挂出大折，v6.3）
        if self._grabbed:
            self._update_grab_lift(dtc)
            self._pin_grab()

        # 6) 压缩屈曲（用积分后、约束前的自然压缩量）
        self._buckling_inject()

        # 7) 距离约束：运动期 iterations 轮；准静态按上一帧最大位移自适应收尾。
        #    准静态期两道上限（防自激，见 §12.16 调参记录）：
        #    a) 每帧修正位移 ≤ settle_move_cap（欠松弛 → 求解器单调收敛而不是来回蹦）；
        #    b) 修正量与重力/积分的速度只按 settle_vel_credit 计入速度回写
        #       （不收敛的褶皱堆里"修正→速度→运动→再修正"会永动，实测毯子永不入睡）。
        # v6.10：**手握但手不动**（保持态）要按准静态收尾，否则折痕永远在抖。
        # 为什么（2026-10-09 实测）：原来 `quiet` 排除了 `_grabbed`，于是「按住不动」时
        # 永远走 `_grab_iters=3` 的运动分支、且拿不到 `settle_move_cap` 欠松弛与
        # `settle_vel_credit` 的速度折算；而 `damping=0.972` 每帧只削 2.8% 动能，
        # 求解器与摩擦之间的残留极限环就长期留在十几 pt/帧的量级
        # （实测全片最大位移 9~16pt/帧、逐帧渲染差分 20~32 万像素，画面就是持续抖动）。
        # 判据用**手速**（不是 `_grabbed`）：手速≈0 = 用户只是按着看 → 收敛；
        # 手在动 → 照旧走拖动的完整求解（手感不变）。
        hand_v = float(np.hypot(self._grab_target[0] - self._last_target[0],
                                self._grab_target[1] - self._last_target[1])) / dtc \
            if self._grabbed else 1e9
        self._hand_speed = hand_v
        holding = self._grabbed and hand_v <= self._body_hold_speed
        quiet = (not self._grabbed or holding) and self._pull is None \
            and self._motion_gate <= self._motion_eps
        pos_mid = pos.copy()
        if (self._grabbed and not holding) or self._pull is not None \
                or self._motion_gate > self._motion_eps:
            iters = self._grab_iters if self._grabbed else self._iterations
        elif self._last_move > 2.0:
            iters = min(self._settle_iterations, 8)
        elif self._last_move > 0.3:
            iters = max(self._iterations, 4)
        else:
            iters = 1
        for _ in range(iters):
            for i0, i1, rest in self._groups:
                self._solve_group(i0, i1, rest)
            for _ in range(self._bend_passes):
                self._solve_bend()
            if self._grabbed:
                self._pin_grab()

        # 7b) 结构边拉伸上限（布料近乎不可伸）。抓取中多跑几遍：拉伸上限逐格传播，
        #     拖动时手速可达 80pt/帧（4~5 格），5 遍追不上 → 钉点邻域被拉开成针。
        # v6.10：**保持态（握但不动）用普通遍数，不要 15 遍**。15 遍是为"手每帧跳 80pt
        # 追得上"设计的（见上）；手停住时它反而有害：拉伸上限**只处理拉伸方向**，
        # 在已经堆满褶皱的保持态里会被反复触发，把长边一格格拉回、与压缩约束来回顶，
        # 残留的十几 pt/帧跳动主要来自这里（实测保持态单帧修正可达 43.7pt）。
        # 判据同第 7 步：手速门控 `holding`。
        _sp = self._grab_strain_passes if (self._grabbed and not holding) \
            else self._strain_passes
        for _ in range(_sp):
            self._strain_limit()
        if self._grabbed:
            self._pin_grab()
        if quiet and self._settle_move_cap > 0.0:
            delta = pos - pos_mid
            d = np.sqrt(delta[:, 0] * delta[:, 0] + delta[:, 1] * delta[:, 1]
                        + delta[:, 2] * delta[:, 2])
            over = d > self._settle_move_cap
            if over.any():
                pos[over] = pos_mid[over] + delta[over] * (
                    self._settle_move_cap / d[over])[:, None]

        # 7c) 塑性蠕变（材料记忆）：把「被压住的地方」记进 rest → 折/堆叠不会自己摊平。
        #     放在约束求解之后：用的是解完的几何；只在真被揉动（motion_gate）时写入。
        self._plastic_creep()

        # 7c2) 退化清理：自碰撞管不到的缝（两层都平躺在地板高度 → 自碰撞只收 z>地板+0.8
        #      的粒子，完全失效）会留下 |Δz|≈0 的贴合 —— 渲染上就是逐像素争夺深度，
        #      顶视看是一圈亮锯齿"撕口"（用户截图折脊那圈）。
        #      v6.9 实测修正三处（旧实现净效果为零，见 _separate_degenerate_overlaps）：
        #      ① 只处理 |Δz| < sep_max_dz 的对；② 共面时定序推同一个粒子；
        #      ③ 保持态**每帧**清（静置期按 clean_interval 兜底），入睡后再维持
        #      `clean_idle_ticks` 帧（见 step 开头的入 sleep 分支）——互穿正是冻在
        #      这些状态里的（实测入睡那一刻仍有 10 对 Δz<3pt 被冻进形状）。
        #      迭代到无贴合对为止（单次 <1ms），所以不需要靠"每 N 帧跑一次"省开销。
        #      手**正在移动**时不跑：拖动中的贴对是"正在被揉的褶皱"，此刻插抬升会改变
        #      手感（实测慢拖整块最高 11.2→16.0pt，击穿「普通拖不拎起」守卫）。
        #      但**握着不动**时必须跑——v6.10 实测那正是"融合"发生的时候：保持态里
        #      折层会互相越贴越近（最近距离 6.6→**1.25pt**、贴合对 13→**73**），
        #      再往后就在渲染上粘成一片（用户报的"融合"）。判据用第 7 步的 `holding`。
        self._clean_tick += 1
        due = ((not self._grabbed or holding) and
               (True if self._pose_active or self._motion_gate > self._motion_eps
                else self._clean_tick >= self._clean_interval))
        if due:
            self._clean_tick = 0
            self._settle_overlaps()

        # 7d) 形状保持（「放下就记住」的**位置投影**）：把捕获时抬起的粒子按回捕获高度。
        #     用位置投影而不是速度弹簧：弹簧会在「接触排除」的交替下留下 0.6pt/s 级极限环、
        #     阻塞入睡（实测 ke 反复冲到 2e3、永不入睡）；位置投影的净位移≈抵消重力 →
        #     速度回写看到 ~0、可正常入睡，同时把姿态稳稳压住（实测下陷 <0.2pt）。
        #     接触优先：上一帧被地板/下层托住的粒子不参与（否则会把自碰撞刚推开的粒子拽回去）。
        #     抓取时 _pose_z 已清空 → 完全自由；窗口结束/重摆也会清空。
        if self._pose_active and self._pose_z is not None and not self._grabbed:
            m = (self._pose_z > self._floor_h + 1.0) & ~self._support_prev
            if m.any():
                err = self._pose_z[m] - pos[m, 2]
                cap = self._pose_gain * np.maximum(np.abs(err), 0.25)   # 软投影：小误差也给一点力
                pos[m, 2] += np.clip(self._pose_gain * err, -cap, cap) * 0.5

        # 8) 地板约束（z ≥ 图标高度）+ 自碰撞
        contact = self._apply_floor()
        touched = self._collide()
        np.clip(pos[:, 2], self._floor_h, self._max_z, out=pos[:, 2])
        # 碰撞推出可能再次把抓点邻边拉过限值；抓点是用户手里钉住的
        # 约束，必须在所有几何修正完成后再做一次局部投影，否则极快拖动
        # 时仍会在最后一帧留下针状边。
        if self._grabbed:
            self._pin_grab()

        # 9) 速度回写 = 本 tick 实际位移/dt（含约束修正 → 约束天然耗能），再施地面摩擦
        #    注意：退化清理的抬升只算**本帧**的接触（sep_touch），不能并进 _support_prev
        #    ——那个掩码会一直累积，被并进去的粒子会永久丢掉竖直速度（实测：猛拽抬起的
        #    材料不再下落、抛掷也飞不起来）。
        sep_touch = self._sep_touch
        self._last_move = float(np.abs(pos - pos_old).max())
        np.subtract(pos, pos_old, out=vel)
        vel /= dtc
        if quiet and self._settle_vel_credit < 1.0:
            vel *= self._settle_vel_credit
        fre = self._friction
        if self._grabbed and self._grab_friction_scale < 1.0:
            fre *= max(self._grab_friction_scale, 0.0)
        if fre > 0.0:
            sp = np.hypot(vel[:, 0], vel[:, 1])
            drop = fre * dtc
            scale = np.maximum(sp - drop, 0.0) / np.maximum(sp, 1e-12)
            vel[:, 0] *= scale
            vel[:, 1] *= scale
        if self._collision_damp < 1.0 and touched.any():
            vel[touched, :2] *= self._collision_damp   # 接触摩擦（xy）：褶皱堆不抖振
        support = contact | touched
        if sep_touch is not None:
            support |= sep_touch
        self._support_prev = support      # 供下一帧的形状保持投影做「接触优先」判断（见 7d）
        if support.any():
            # 非弹性接触：被地板/下层托住的粒子不保留竖直速度——否则重力每帧压出
            # vz，与碰撞推出构成永不衰减的抖振（实测 ke 卡在 ~10²，毯子永不入睡）。
            vel[support, 2] = 0.0
        if self._pose_active and self._pose_z is not None and not self._grabbed:
            # 保持中的粒子：竖直速度**冻结**（×0.05）。位置投影本身不注入速度，但重力每帧
            # 递增值与「接触排除」交替会让残余微动长期存在（实测 ke 卡 ~192 / 95s 不入睡）。
            # 姿态由位置投影保证，速度清零不会影响姿态；只降能量、不会自激。
            mk = self._pose_z > self._floor_h + 1.0
            if mk.any():
                vel[mk, 2] *= 0.05

        # 10) 抓点历史（供 fling 估计速度）与休眠判定
        if self._grabbed:
            self._drag_hist.append(
                (float(self._grab_target[0]), float(self._grab_target[1]), dtc)
            )
            if len(self._drag_hist) > 6:
                self._drag_hist.pop(0)
            self._last_target = self._grab_target.copy()
        ke = float((vel * vel).sum())
        self._ke = ke
        self._update_ao()
        # 入睡阈值：形状保持生效时放宽（保持把姿态钉住，不会漂移；残差来自投影与重力的
        # 数值平衡，实测 ~0.007pt/帧、肉眼不可见，但严格阈值卡在 1.0 之外会让毯子永不入睡、
        # 白烧 CPU——保持态实测每帧 ~2ms × 30fps）。
        if not self._grabbed and self._pull is None and ke < self._sleep_eps:
            self._asleep = True
            self._clean_idle_left = self._clean_idle_ticks

    def is_asleep(self) -> bool:
        """kinetic_energy() < ε 且未抓取 → True。休眠中遇事件须被 wake() 拉起。"""
        return self._asleep

    def wake(self) -> None:
        """强制唤醒，直到再次收敛才允许休眠。"""
        self._asleep = False
        self._clean_idle_left = 0

    def set_bumps(self, field: BumpField | None) -> None:
        """替换 BumpField（None = 无隆起/降级模式）；下次 step() 生效。配合 wake() 使用。"""
        self._bumps = field

    # ---- 摆放（编辑模式拖角缩放/旋转 与 「摆放回来」共用） ----
    def placement(self) -> tuple[float, float, float, float]:
        """当前摆放 (center_x, center_y, width_pt, height_pt)。

        角度不由 sim 跟踪（自由拖动后姿态任意）——编辑模式的角度状态由调用方持有。
        """
        c = self._pos[:, :2].mean(axis=0)
        return float(c[0]), float(c[1]), float(self.width_pt), float(self.height_pt)

    def set_placement(self, center, width_pt: float, height_pt: float, angle_rad: float,
                      prev_angle_rad: float | None = None,
                      zero_velocity: bool = False) -> None:
        """把布料刚性摆成「中心 center、尺寸 width×height、绕中心旋转 angle」的规整矩形（摊平到桌面）。

        - 编辑模式（拖角缩放 / 拖旋转柄）：逐帧调用，位置写成规整网格、rest 长度同步重建，
          速度按尺寸比缩放 + 按角度差旋转（prev_angle_rad 给出上一帧角度）→ 手感连续；
        - 「摆放回来」：zero_velocity=True，直接得到一个静止的规整毯子。
        """
        w = max(float(width_pt), 40.0)
        h = max(float(height_pt), 40.0)
        cx, cy = float(center[0]), float(center[1])
        old_w = float(self.width_pt)
        old_h = float(self.height_pt)
        dx = w / (self.cols - 1)
        dy = h / (self.rows - 1)
        gx = np.arange(self.cols, dtype=np.float64) * dx - w * 0.5
        gy = np.arange(self.rows, dtype=np.float64) * dy - h * 0.5
        gxx, gyy = np.meshgrid(gx, gy)                 # (rows, cols) 行主序
        ca, sa = float(np.cos(angle_rad)), float(np.sin(angle_rad))
        lx = gxx.ravel()
        ly = gyy.ravel()
        self._pos[:, 0] = cx + ca * lx - sa * ly
        self._pos[:, 1] = cy + sa * lx + ca * ly
        # 刚性重摆 = 摊平：z 落在（新位置的）地板高度上
        self._update_floor()
        self._pos[:, 2] = self._floor_h
        self._vel[:, 2] = 0.0
        # 形状记忆/保持一并清零：重摆过的毯子按新几何当 rest（否则「抚平折痕」「改尺寸」
        # 之后会被旧的形状保持弹簧拽回折起来的样子——实测摊平后仍有 31.9pt 高度）
        self._pose_z = None
        self._pose_pending = 0
        if zero_velocity:
            self._vel[:, 0] = 0.0
            self._vel[:, 1] = 0.0
        else:
            scale = 0.5 * ((w / max(old_w, 1e-6)) + (h / max(old_h, 1e-6)))
            self._vel[:, :2] *= scale
            if prev_angle_rad is not None:
                da = float(angle_rad) - float(prev_angle_rad)
                if abs(da) > 1e-9:
                    cd, sd = float(np.cos(da)), float(np.sin(da))
                    vx = self._vel[:, 0].copy()
                    vy = self._vel[:, 1].copy()
                    self._vel[:, 0] = cd * vx - sd * vy
                    self._vel[:, 1] = sd * vx + cd * vy
            # 切角/拖动中：清掉抓取绑定，避免与编辑手势互相拉扯
            self._grabbed = False
            self._grab_idx = None
            self._grab_center0 = None
            self._drag_hist = []
        self.width_pt = w
        self.height_pt = h
        self._rebuild_constraints()
        self._update_ao()
        self._asleep = False
        self._pull = None

    def flatten_folds(self) -> None:
        """摊平（菜单「抚平折痕（摊平毯子）」）：把布料**刚性重摆**成当前中心/尺寸的规整矩形。

        折痕/翻折的本质是布面在平面内的折返（z 只是一部分），只清 z 不够——重新铺成
        规整网格才是真「摊平」。保留中心与尺寸（角度回到 0，与自由拖动后的姿态无关）。
        """
        cx, cy, w, h = self.placement()
        self.set_placement((cx, cy), w, h, 0.0, zero_velocity=True)
        self.wake()

    # ---- 几何输出（RugOverlay 每帧读取，重建 SCNGeometry） ----
    def vertices(self) -> np.ndarray:
        """返回 (N, 3) float32 C-contiguous 数组：x, y（主屏本地 pt，top-left, y-down）
        与 z（高度 pt，≥ 地板高度、向上为正）。N = cols*rows。行主序 index = row*cols + col。
        """
        self._verts[:, 0] = self._pos[:, 0]
        self._verts[:, 1] = self._pos[:, 1]
        self._verts[:, 2] = self._pos[:, 2]
        return self._verts

    def uv(self) -> np.ndarray:
        """返回 (N, 2) 数组，与 vertices() 行序一致：u = col/(cols-1) 从左到右，
        v = row/(rows-1) 从上到下。复用同一 buffer，不逐帧重建。
        """
        return self._uv

    # ---- 渲染支持查询（v6「厚度」轮新增；只读、不参与动力学） ----
    def floor_heights(self) -> np.ndarray:
        """返回 (N,) float32：本 tick 每个粒子下方的地板高度（= 图标隆起；无隆起为 0）。

        渲染层用它算「离地高度」：静止微浮雕在贴地时满幅、被抬起/翻折时衰减为 0。
        """
        return np.asarray(self._floor_h, dtype=np.float32)

    def ao(self) -> np.ndarray:
        """返回 (N,) float32 ∈ [0,1]：折层间遮蔽近似（被上层布料压住的粒子接近 1）。

        由 xy 粗网格「局部顶高 − 自身高度」算出（见 _update_ao），随 step() 更新。
        渲染层按顶点色压暗折缝 → 翻折两层间出现接触阴影（v6「厚度」观感的关键之一）。
        纯渲染数据：不读时钟、不改任何动力学状态。
        """
        return self._ao

    def indices(self) -> np.ndarray:
        """返回 (M,) uint32 三角列表（与 spike build_indices 同环绕顺序）。
        必须复用同一 buffer（每次调用返回同一数组）。
        """
        return self._indices

    # ---- 查询 ----
    def nearest_distance(self, x: float, y: float) -> float:
        """(x, y)（主屏本地 pt）到最近粒子的 xy 平面距离（pt，不含 z）。"""
        d2 = (self._pos[:, 0] - float(x)) ** 2 + (self._pos[:, 1] - float(y)) ** 2
        return float(np.sqrt(d2.min()))

    def kinetic_energy(self) -> float:
        """Σ|v|²（内部单位，3D 速度，仅用于休眠阈值比较与单测断言）。"""
        return float((self._vel * self._vel).sum())
