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
        damping: float = 0.985,
        gravity: float = 1200.0,          # z 向重力（pt/s²）
        friction: float = 4000.0,         # 面内 (x,y) 地面摩擦（pt/s 速度削减；"厚毯"：毯子不轻易
                                          # 滑走 → 手上的位移只能靠材料折叠消化，折痕才会出现）
        buckling: float = 0.7,            # 压缩屈曲：多余长度 → 向上位置偏移的系数
                                          # （必须 > 重力每帧 1.33pt 的量级，否则拱起被重力压回）
        buckling_step_cap: float = 2.5,   # 单帧屈曲注入上限（pt；防猛冲打炸）
        buckling_dead_zone: float = 0.4,  # 压缩死区（pt）：滤掉欠收敛的微量压缩
        buckling_min_ratio: float = 0.10, # 压缩超过 rest×该比例才注入（普通拖动只起浅纹）
        buckling_blur_cells: int = 4,     # 屈曲种子的网格空间低通半径（格；σ≈4 格≈50pt）——
                                          # 拱起波长必须远大于格距，否则顶视是"揉纸"碎皱
        fold_cap_z: float = 40.0,         # 屈曲饱和高度（pt）：拱到这么高就不再注入（防能量泵）
        bend_soft: float = 0.45,          # 弯曲刚度（柔性区双向 (i,i+2) 约束）：波浪圆润、
                                          # 落定后保留（0 = 折纸感且落定就摊平，实测见 §12.16）
        crease_ratio: float = 0.90,       # 折痕圆角下限（= |p_{i+2}-p_i| / (2×间距) 的最小值）。
                                          # 0.90 ↔ 曲率半径 ≈ 1.2 格 ≈ 15pt：**禁止剪纸式锐折**，
                                          # 材料被迫拱起成有体积的圆角折脊（v5 观感关键，见 §12.17）。
                                          # 1.0 = 只允许完全平直（不可用）；0 = 关闭（回到 v4 锐折）
        collision_sep: float = 9.0,       # 自碰撞最小间距（两层翻折的“布厚”）
        collision_iters: int = 3,
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
        compression_soft: float = 0.40,   # 压缩方向距离约束刚度比例（"推"要能传出去：手的推挤
                                          # 在毯身上形成**一片**压缩区 → 低频屈曲长成大波浪；
                                          # 0.1 时推挤不外传、0.9 时把折痕推开，实测两端都差）
        grab_friction_scale: float = 0.35,
        body_k: float = 30.0,             # 「整块跟手」助力（只驱动**质心**跟随手移动；越大毯子越
                                          # "飘"（不易出折痕），越小越沉，实测 30 = 沉而能用）
        body_c: float = 14.0,             # 助力阻尼（≈临界阻尼）
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
        self._compression_soft = float(np.clip(compression_soft, 0.0, 1.0))
        self._grab_friction_scale = float(grab_friction_scale)
        self._body_k = float(body_k)
        self._body_c = float(body_c)
        self._motion_gate = 1.0
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
        for p in (0, 1):
            h = i0_h[par_h == p]
            v = i0_v[par_v == p]
            d = i0_d[par_d == p]
            self._groups.append((h, h + 1, sx))              # 结构横
            self._groups.append((v, v + cols, sy))           # 结构竖
            self._groups.append((d, d + cols + 1, diag))     # 剪切 ↘
            self._groups.append((d + 1, d + cols, diag))     # 剪切 ↙
            self._structure_groups.append((h, h + 1, sx))
            self._structure_groups.append((v, v + cols, sy))
        self._b_i0 = np.concatenate([i0_h, i0_v])
        self._b_i1 = np.concatenate([i0_h + 1, i0_v + cols])
        self._b_rest = np.concatenate([np.full(i0_h.size, sx), np.full(i0_v.size, sy)])
        # 弯曲刚度：(i, i+2) 软距离约束（rest = 2×间距）——浅波纹被拉直、大折叠保留，
        # 布料不会退化成折纸（v3 的塑性折痕场在 3D 里不需要：折痕由真实几何承载）。
        # v5：同组再加**折痕圆角下限**（crease_min，见 _solve_bend）——禁止锐折，
        # 折痕必须拱成有体积的圆角（观感对齐参考视频，§12.17）。
        rr, cc = np.meshgrid(np.arange(self.rows), np.arange(cols), indexing="ij")
        bh0 = (rr[:, :-2] * cols + cc[:, :-2]).ravel()
        bh_par = cc[:, :-2].ravel() & 1
        bv0 = (rr[:-2, :] * cols + cc[:-2, :]).ravel()
        bv_par = rr[:-2, :].ravel() & 1
        self._bend_groups = []
        self._crease_min = []
        for p in (0, 1):
            self._bend_groups.append((bh0[bh_par == p], bh0[bh_par == p] + 2, 2.0 * sx))
            self._bend_groups.append((bv0[bv_par == p], bv0[bv_par == p] + 2 * cols, 2.0 * sy))
            self._crease_min.append(2.0 * sx * self._crease_ratio)
            self._crease_min.append(2.0 * sy * self._crease_ratio)
        self._cell = max(sx, sy)                             # 自碰撞哈希格边长

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
        for (i0, i1, rest), cmin in zip(self._bend_groups, self._crease_min):
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
        """结构边的硬拉伸上限（布料近乎不可伸）：超过 max_stretch×rest 的边两端互相拉回。

        与弹性约束的区别：只处理「过长」的边、一遍就消掉超出部分（不做松弛）→
        拉力沿织物近似逐格 1:1 传播（「抓住一点拖整块毯子」整块跟手的关键）。
        组内无共享端点（棋盘分色）→ fancy-index 原地更新安全。
        """
        if self._max_stretch <= 1.0:
            return
        pos = self._pos
        pin = self._grab_idx if self._grabbed else None
        for i0, i1, rest in self._structure_groups:
            d = pos[i1] - pos[i0]
            dist = np.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2])
            excess = dist - rest * self._max_stretch
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
        lifted = np.nonzero(pos[:, 2] > self._floor_h + 0.3)[0]
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
                    # 跳过网格近邻（结构上本就在一起的粒子不参与互斥）
                    near = (np.abs(rr[ii] - rr[jj]) <= 2) & (np.abs(cc[ii] - cc[jj]) <= 2)
                    d3 = pos[ii] - pos[jj]
                    dist = np.sqrt(d3[:, 0] * d3[:, 0] + d3[:, 1] * d3[:, 1] + d3[:, 2] * d3[:, 2])
                    idx = np.nonzero((dist < sep) & ~near & (dist > 1e-6))[0]
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
        self._grab_center0 = self._pos[:, :2].mean(axis=0).copy()
        self._grab_anchor = self._grab_target.copy()
        self.wake()
        return True

    def drag_to(self, x: float, y: float) -> None:
        """把抓点目标移动到 (x, y)（主屏本地 pt）。未抓取时为 no-op。"""
        if not self._grabbed:
            return
        self._grab_target = np.array([float(x), float(y)], dtype=np.float64)

    def release(self, fling: bool = True) -> None:
        """释放抓点。重复调用幂等。

        fling=True 且最近若干帧平均速度超阈值时，给全体粒子注入 **xy 平移初速度 +
        角速度**（抛掷飞行；z 交给重力自然落回）；释放后数秒内 is_asleep()。
        """
        if not self._grabbed:
            return
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
        self._grabbed = False
        self._grab_idx = None
        self._grab_center0 = None
        self._drag_hist = []
        self.wake()

    def throw_in(
        self,
        target_center: tuple[float, float],
        from_corner: str = "top_left",
    ) -> None:
        """「铺上」动画入口：毯子从屏幕一角飞入并落在 target_center（xy 平面内，z=0 平铺）。

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
        self._vel[:, 2] = 0.0
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
            ax = self._body_k * lag[0] - self._body_c * vm[0]
            ay = self._body_k * lag[1] - self._body_c * vm[1]
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

        # 4) 重力（-z）+ 速度护栏 + verlet/PBD 积分（门控用 xy 平均速度）
        vel[:, 2] -= self._gravity * dtc
        sp = np.sqrt((vel * vel).sum(axis=1))
        over = sp > self._v_max
        if over.any():
            vel[over] *= (self._v_max / sp[over])[:, None]
        self._motion_gate = float(min(
            np.hypot(vel[:, 0], vel[:, 1]).mean() / 80.0, 1.0))
        pos_old = pos.copy()
        pos += vel * dtc

        # 5) 抓点钉住（只钉 xy——手在桌面上拖）
        if self._grabbed:
            pos[self._grab_idx, 0] = self._grab_target[0]
            pos[self._grab_idx, 1] = self._grab_target[1]

        # 6) 压缩屈曲（用积分后、约束前的自然压缩量）
        self._buckling_inject()

        # 7) 距离约束：运动期 iterations 轮；准静态按上一帧最大位移自适应收尾。
        #    准静态期两道上限（防自激，见 §12.16 调参记录）：
        #    a) 每帧修正位移 ≤ settle_move_cap（欠松弛 → 求解器单调收敛而不是来回蹦）；
        #    b) 修正量与重力/积分的速度只按 settle_vel_credit 计入速度回写
        #       （不收敛的褶皱堆里"修正→速度→运动→再修正"会永动，实测毯子永不入睡）。
        quiet = (not self._grabbed) and self._pull is None \
            and self._motion_gate <= self._motion_eps
        pos_mid = pos.copy()
        if self._grabbed or self._pull is not None or self._motion_gate > self._motion_eps:
            iters = self._iterations
        elif self._last_move > 2.0:
            iters = min(self._settle_iterations, 8)
        elif self._last_move > 0.3:
            iters = max(self._iterations, 4)
        else:
            iters = 1
        for _ in range(iters):
            for i0, i1, rest in self._groups:
                self._solve_group(i0, i1, rest)
            self._solve_bend()
            if self._grabbed:
                pos[self._grab_idx, 0] = self._grab_target[0]
                pos[self._grab_idx, 1] = self._grab_target[1]

        # 7b) 结构边拉伸上限（布料近乎不可伸）
        for _ in range(self._strain_passes):
            self._strain_limit()
        if self._grabbed:
            pos[self._grab_idx, 0] = self._grab_target[0]
            pos[self._grab_idx, 1] = self._grab_target[1]
        if quiet and self._settle_move_cap > 0.0:
            delta = pos - pos_mid
            d = np.sqrt(delta[:, 0] * delta[:, 0] + delta[:, 1] * delta[:, 1]
                        + delta[:, 2] * delta[:, 2])
            over = d > self._settle_move_cap
            if over.any():
                pos[over] = pos_mid[over] + delta[over] * (
                    self._settle_move_cap / d[over])[:, None]

        # 8) 地板约束（z ≥ 图标高度）+ 自碰撞
        contact = self._apply_floor()
        touched = self._collide()
        np.clip(pos[:, 2], self._floor_h, self._max_z, out=pos[:, 2])

        # 9) 速度回写 = 本 tick 实际位移/dt（含约束修正 → 约束天然耗能），再施地面摩擦
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
        if support.any():
            # 非弹性接触：被地板/下层托住的粒子不保留竖直速度——否则重力每帧压出
            # vz，与碰撞推出构成永不衰减的抖振（实测 ke 卡在 ~10²，毯子永不入睡）。
            vel[support, 2] = 0.0

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
        if not self._grabbed and self._pull is None and ke < self._sleep_eps:
            self._asleep = True

    def is_asleep(self) -> bool:
        """kinetic_energy() < ε 且未抓取 → True。休眠中遇事件须被 wake() 拉起。"""
        return self._asleep

    def wake(self) -> None:
        """强制唤醒，直到再次收敛才允许休眠。"""
        self._asleep = False

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
