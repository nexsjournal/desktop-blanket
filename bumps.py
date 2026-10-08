"""bumps.py — BumpField：桌面图标位置 → 静态目标高度场。

纯逻辑模块：只依赖 numpy；无 I/O、无时钟读取、无线程、无 AppKit；
同一输入重复调用结果确定。语义唯一来源：contracts/interfaces.py 的 BumpField
docstring（开发文档 §7.3），公式：

    A_i  = base_height_pt * (1 + min(log2(1 + size_bytes / 2**20), 2.0) * size_gain)
    c_i  = (x_pt + dx, y_pt + dy)          # calibration 偏移在本类内应用
    z(p) = min(max_height_pt, Σ_i A_i * exp(-|p - c_i|² / (2 * sigma_pt²)))

文件越大 A 越大；同格多图标的高斯直接相加再截断（堆叠效应）。
icons 为空 → 恒为 0。坐标一律主屏本地屏幕点 pt（top-left、y 向下）。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # 仅类型标注；运行时零依赖 contracts，模块可独立导入
    from contracts.interfaces import IconInfo

__all__ = ["BumpField"]


class BumpField:
    """图标位置 → 静态目标高度场（"taller stacks, bigger bumps"）。"""

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
            calibration: (dx, dy)，screen_point = finder_point + calibration。
            sigma_pt: 高斯宽度（文档初值 ≈45pt）。
            base_height_pt: 隆起基础高度（文档初值 10pt）。
            size_gain: 文件大小增益系数（文档初值 0.6）。
            max_height_pt: 叠加截断上限（文档初值 28pt）。
        """
        width_pt = float(width_pt)
        height_pt = float(height_pt)
        if width_pt <= 0.0 or height_pt <= 0.0:
            raise ValueError("width_pt/height_pt 必须为正")
        if sigma_pt <= 0.0:
            raise ValueError("sigma_pt 必须为正")

        dx, dy = float(calibration[0]), float(calibration[1])
        cs_x: list[float] = []
        cs_y: list[float] = []
        amps: list[float] = []
        for icon in (icons or []):
            cx = float(icon.x_pt) + dx
            cy = float(icon.y_pt) + dy
            # 裁剪屏外图标（标定后的中心落在主屏外则丢弃）
            if not (0.0 <= cx <= width_pt and 0.0 <= cy <= height_pt):
                continue
            size_bytes = max(float(getattr(icon, "size_bytes", 0) or 0), 0.0)
            gain = min(float(np.log2(1.0 + size_bytes / 1048576.0)), 2.0)
            amps.append(base_height_pt * (1.0 + gain * float(size_gain)))
            cs_x.append(cx)
            cs_y.append(cy)

        self.width_pt = width_pt
        self.height_pt = height_pt
        self.sigma_pt = float(sigma_pt)
        self._cx = np.asarray(cs_x, dtype=np.float64)
        self._cy = np.asarray(cs_y, dtype=np.float64)
        self._amp = np.asarray(amps, dtype=np.float64)
        self._two_sigma2 = 2.0 * float(sigma_pt) ** 2
        self._max_height = float(max_height_pt)

    def heights(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """查询高度场（ClothSim 每次重建 z 目标时批量调用）。

        Args:
            xs / ys: 同形状的 numpy 数组（任意形状），主屏本地屏幕点 pt。

        Returns:
            与输入同形状的 np.ndarray：z 高度 pt（≥0，向上）。不修改输入数组。
        """
        xs_a = np.asarray(xs, dtype=np.float64)
        ys_a = np.asarray(ys, dtype=np.float64)
        if self._cx.size == 0:
            return np.zeros(xs_a.shape, dtype=np.float64)
        # (..., M) 一次性广播全部图标，再沿图标轴加权求和
        d2 = (xs_a[..., None] - self._cx) ** 2 + (ys_a[..., None] - self._cy) ** 2
        z = np.exp(-d2 / self._two_sigma2) @ self._amp
        return np.minimum(z, self._max_height)
