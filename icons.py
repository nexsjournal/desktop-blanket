"""icons.py — 桌面图标感知（只读）：osascript 查 Finder + os.stat 补大小 + 变化通知 + 坐标标定。

实现 contracts/interfaces.py 的 IconSensor / IconSensorError / IconInfo 契约。

安全（开发文档 §4，写进实现而非自觉）：
  - 对 ~/Desktop 与 Finder 只读：AppleScript 只取 name / position / 桌面窗口 bounds，
    脚本中不含任何写指令；本地只做 os.stat / os.scandir（均为只读）；
    绝不读写 .DS_Store（query 里防御性跳过，绝不 stat 它）。
  - 唯一权限依赖：「自动化 → Finder」（AppleEvent）。权限被拒（-1743）/ 超时 /
    解析失败 / 标定失败 → IconSensorError，由 RugApp 捕获进入降级模式；
    本模块不重试（不二次弹权限）。

AppleScript 输出格式（可单测，_parse_icons_output 消费）::

    第 1 行            记录数 N
    随后 N 组、每组 2 行：第 1 行 = 文件名；第 2 行 = "x y"
                        （Finder 桌面视图坐标：桌面左上原点、y 向下、pt）

  某一项 position 读取失败会被 AppleScript 的 try 跳过且不计数；行数与 N 不符
  按解析失败处理（宁降级不出错数据）。局限：文件名含换行会破坏行结构 → 解析失败
  → IconSensorError（同样走降级，不产生错误坐标）。

线程模型（契约：主线程单线程，本模块不创建线程/锁）：
  start() 须在主线程调用。FSEvents 流回调运行在 FSEvents 自己的系统线程，
  只置 threading.Event 脏标志；on_change 由主线程 NSTimer 轮询脏标志后触发
  （FSEvents 模式 0.5s / 轮询降级模式 1.5s，G4 要求变化后 ≤2s 触发重建）。
  FSEvents 导入或流启动失败 → 降级为主线程 NSTimer 轮询（os.scandir 快照对比，
  检测到变化同样只置脏，on_change 仍走主线程脏标志路径）。

日志：RUG-INFO / RUG-WARN / RUG-ERROR 单行 stdout，仅生命周期与异常摘要，
禁止逐条图标打印（§5）。
"""
from __future__ import annotations

import math
import os
import subprocess
import threading
from dataclasses import dataclass
from typing import Callable

from Foundation import NSRunLoop, NSTimer

try:  # 契约类型唯一来源（contracts 是正规包）；仅当无法按包导入时才退回本地等价定义
    from contracts.interfaces import IconInfo, IconSensorError
except ImportError:  # pragma: no cover — icons.py 被单独拷走运行时的兜底
    @dataclass
    class IconInfo:  # noqa: D101 — 与 contracts.interfaces.IconInfo 逐字段等价
        name: str
        x_pt: float
        y_pt: float
        size_bytes: int

    class IconSensorError(RuntimeError):  # noqa: D101
        """图标感知失败：权限被拒（-1743）/ 超时 / 解析失败 / 标定失败。"""


__all__ = ["IconInfo", "IconSensorError", "IconSensor"]

_OSASCRIPT_TIMEOUT_S = 5.0   # 契约：osascript 超时 ≈5s
_FS_LATENCY_S = 0.3          # FSEvents 事件合并延迟
_FS_TICK_S = 0.5             # FSEvents 模式：主线程消费脏标志的周期
_POLL_INTERVAL_S = 1.5       # 降级轮询周期（契约允许 1~2s；G4 ≤2s）
_SKIP_NAME = ".DS_Store"     # 安全红线 S2：不读写

# 只读查询：桌面项（desktop 的 items）的 name + position。逐项 try（position 偶发不可读时
# 跳过该项且不计数），首行计数保证解析端可校验结构完整性。
#
# 真机踩坑（2026-10-08，macOS 26）：旧写法 `(get desktop items)` **编译失败**
# （osascript error -2741 "Expected \",\" but found plural class name"——Finder 词典不再
# 接受「desktop items」这个复数类名）；规范写法 `items of the desktop` 可用。此处按候选
# 顺序尝试（与 _CALIB_SCRIPTS 同策略），编译+执行成功的第一个生效。
def _make_icon_script(target_expr: str) -> str:
    return (
        'tell application "Finder"\n'
        '	set _out to ""\n'
        '	set _n to 0\n'
        f'	repeat with f in (get {target_expr})\n'
        '		try\n'
        '			set _p to position of f\n'
        '			set _out to _out & (name of f) & linefeed & ((item 1 of _p) as text) '
        '& " " & ((item 2 of _p) as text) & linefeed\n'
        '			set _n to _n + 1\n'
        '		end try\n'
        '	end repeat\n'
        '	return ((_n as text) & linefeed & _out)\n'
        'end tell'
    )


_ICON_SCRIPTS: tuple[tuple[str, str], ...] = (
    ("items-of-desktop", _make_icon_script("items of the desktop")),
    ("every-item-of-desktop", _make_icon_script("every item of the desktop")),
)

# 标定候选（同一 Automation→Finder 权限，只读）。按序尝试，第一个成功者生效；
# 全部失败 → IconSensorError（调用方回退 (0.0, 24.0)）。
# bounds 语义：Finder 窗口 {left, top, right, bottom}，top-left 原点、y 向下。
# 桌面窗口 bounds 的 left/top 即 Finder 桌面坐标原点在主屏本地坐标中的偏移。
# 真机实测（macOS 26）：`desktop window` 已不可得（-1728），`window of desktop` 可用
# （本机返回 0,0,2560,1440 → dx=dy=0，与桌面图标窗口覆盖整屏一致），故把可用写在前。
_CALIB_SCRIPTS: tuple[tuple[str, str], ...] = (
    ("window-of-desktop", 'tell application "Finder" to get bounds of window of desktop'),
    ("desktop-window", 'tell application "Finder" to get bounds of desktop window'),
)


def _parse_icons_output(text: str) -> list[tuple[str, float, float]]:
    """解析 osascript 输出为 [(name, x_pt, y_pt), ...]；结构不符 → ValueError。"""
    lines = text.split("\n")
    while lines and not lines[-1].strip():  # 去掉结尾 linefeed 产生的空行
        lines.pop()
    if not lines:
        raise ValueError("输出为空")
    try:
        count = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"首行不是记录数: {lines[0]!r}") from exc
    if count < 0 or len(lines) != 1 + 2 * count:
        raise ValueError(f"行数与记录数不符: count={count}, 有效行={len(lines)}")
    records: list[tuple[str, float, float]] = []
    for i in range(count):
        name = lines[1 + 2 * i]
        pos_line = lines[2 + 2 * i]
        parts = pos_line.split()
        if len(parts) != 2:
            raise ValueError(f"第 {i + 1} 条 position 行格式错误: {pos_line!r}")
        try:
            x_pt, y_pt = float(parts[0]), float(parts[1])
        except ValueError as exc:
            raise ValueError(f"第 {i + 1} 条坐标不是数字: {pos_line!r}") from exc
        if not (math.isfinite(x_pt) and math.isfinite(y_pt)):
            raise ValueError(f"第 {i + 1} 条坐标非有限值: {pos_line!r}")
        if not name:
            raise ValueError(f"第 {i + 1} 条文件名为空")
        records.append((name, x_pt, y_pt))
    return records


def _one_line(text: str, limit: int = 300) -> str:
    """多行文本 → 单行（换行折成 " / "，去空行）并截断。

    契约/contract.md §5：stdout 日志必须单行；stderr 原文常含换行，直接拼进
    IconSensorError 会让调用方的 `RUG-WARN <msg>` 破成多行。
    """
    flat = " / ".join(
        seg.strip() for seg in str(text or "").splitlines() if seg.strip()
    )
    return flat[:limit] + "…" if len(flat) > limit else flat


def _parse_bounds_output(text: str) -> tuple[float, float, float, float]:
    """解析 AppleScript bounds 输出 "{left, top, right, bottom}"；不符 → ValueError。"""
    raw = (text or "").strip().strip("{}").strip()
    parts = [p.strip() for p in raw.split(",") if p.strip()] if raw else []
    if len(parts) != 4:
        raise ValueError(f"期望 4 个 bounds 数值，得到: {text!r}")
    try:
        left, top, right, bottom = (float(p) for p in parts)
    except ValueError as exc:
        raise ValueError(f"bounds 含非数值（如 missing value）: {text!r}") from exc
    return left, top, right, bottom


def _is_fatal_sensor_error(exc: IconSensorError) -> bool:
    """该错误是否「换候选脚本也救不了」：权限被拒（-1743）或 Finder 超时。

    这两类与脚本内容无关；继续试下一个候选只会浪费时间（权限被拒时还可能多弹权限窗），
    因此 query() 的候选循环遇到它们立即上抛。
    """
    msg = str(exc)
    return "-1743" in msg or "超时" in msg


class IconSensor:
    """只读感知 ~/Desktop：图标位置/大小 + 变化通知 + Finder→屏幕坐标标定。

    query() 须主线程调用（子进程阻塞最多数秒）；start()/stop() 须主线程调用
    （NSTimer 挂在当前 run loop，RugApp 在主线程调用 → 主 run loop）。
    """

    def __init__(self, desktop_path: str | None = None) -> None:
        """desktop_path 默认 ~/Desktop；仅用于只读 stat 与变化监听。"""
        self._desktop_path = os.path.expanduser(desktop_path) if desktop_path else os.path.expanduser("~/Desktop")
        self._dirty = threading.Event()  # FSEvents 系统线程置位；主线程 tick_ 消费
        self._on_change: Callable[[], None] | None = None
        self._timer = None               # Foundation.NSTimer（主线程 run loop）
        self._fs_stream = None           # FSEventStreamRef
        self._fs_mod = None              # 已加载的 FSEvents 模块（None=轮询降级）
        self._poll_snapshot: dict[str, tuple[int, int, bool]] | None = None
        self._calibration_cache: tuple[float, float] | None = None

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def query(self) -> list[IconInfo]:
        """返回当前桌面项快照 list[IconInfo]（顺序不限）。

        osascript 只读 AppleScript 取 name + position → os.stat 本地补 size_bytes
        （文件已消失 → 0；.DS_Store 防御性跳过，绝不 stat）。失败 → IconSensorError。
        脚本按 _ICON_SCRIPTS 候选顺序尝试（应对不同 macOS 版本的 Finder 词典差异）；
        权限被拒 / Finder 超时属「换脚本也救不了」的致命错误，立即抛出、不重试
        （也避免多弹权限窗）。
        """
        stdout = None
        errors: list[str] = []
        for label, script in _ICON_SCRIPTS:
            try:
                stdout = self._osascript(script)
                break
            except IconSensorError as exc:
                errors.append(f"[{label}] {exc}")
                if _is_fatal_sensor_error(exc):
                    raise
        if stdout is None:
            raise IconSensorError("Finder 桌面项查询全部候选脚本失败：" + "；".join(errors))
        try:
            records = _parse_icons_output(stdout)
        except ValueError as exc:
            raise IconSensorError(
                f"Finder 桌面项输出解析失败: {exc}；原始输出前 200 字符: {stdout[:200]!r}"
            ) from exc
        result: list[IconInfo] = []
        for name, x_pt, y_pt in records:
            if name == _SKIP_NAME:  # S2：.DS_Store 不读不写
                continue
            result.append(
                IconInfo(name=name, x_pt=x_pt, y_pt=y_pt, size_bytes=self._stat_size(name))
            )
        return result

    def _stat_size(self, name: str) -> int:
        """os.stat 只读补文件大小；文件不存在/不可读 → 0（契约）。"""
        try:
            return os.stat(os.path.join(self._desktop_path, name)).st_size
        except OSError:
            return 0

    def _osascript(self, script: str) -> str:
        """执行一条只读 AppleScript；非零返回/超时/启动失败 → IconSensorError。"""
        try:
            proc = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=_OSASCRIPT_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired as exc:
            raise IconSensorError(
                f"osascript 超时（>{_OSASCRIPT_TIMEOUT_S:.0f}s），Finder 未响应"
            ) from exc
        except OSError as exc:
            raise IconSensorError(f"osascript 启动失败: {exc}") from exc
        if proc.returncode != 0:
            # 单行化（contract.md §5）：stderr 多行原文折成 " / " 并截断到 300 字符
            err = _one_line(proc.stderr or "", 300)
            if "-1743" in err:
                raise IconSensorError(
                    "自动化权限被拒（AppleEvent -1743）：需在 系统设置→隐私与安全性→自动化 "
                    f"中授权控制 Finder。详情: {err or '<stderr 为空>'}"
                )
            raise IconSensorError(
                f"osascript 失败 (rc={proc.returncode}): {err or '<stderr 为空>'}"
            )
        return proc.stdout

    # ------------------------------------------------------------------
    # 变化通知
    # ------------------------------------------------------------------
    def start(self, on_change: Callable[[], None]) -> None:
        """订阅桌面变化通知（须主线程调用；幂等：重复 start 先停旧流）。

        优先 FSEvents 只读监听 desktop_path；import 失败或流启动失败 → 降级为
        1.5s 主线程 NSTimer 轮询（os.scandir 快照对比）。on_change 保证主线程调用。
        """
        if not callable(on_change):
            raise TypeError("on_change 必须是可调用对象")
        self.stop()
        self._on_change = on_change
        self._dirty.clear()
        use_fs = False
        try:
            import FSEvents  # 延迟导入：缺失/损坏时走降级（契约允许）
            self._start_fs_stream(FSEvents)
            self._fs_mod = FSEvents
            use_fs = True
        except Exception as exc:
            print(
                f"RUG-WARN IconSensor: FSEvents 不可用（{exc}），降级为 "
                f"{_POLL_INTERVAL_S:.1f}s 主线程轮询",
                flush=True,
            )
        if use_fs:
            self._poll_snapshot = None
            interval = _FS_TICK_S
        else:
            self._poll_snapshot = self._snapshot()
            interval = _POLL_INTERVAL_S
        try:
            self._timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                interval, self, "tick:", None, True
            )
        except Exception:
            self.stop()  # 流已启动但 timer 失败：清理干净再报错（S6 无残留）
            raise
        print(
            f"RUG-INFO IconSensor.start: mode={'FSEvents' if use_fs else 'polling'} "
            f"interval={interval:.1f}s path={self._desktop_path}",
            flush=True,
        )

    def stop(self) -> None:
        """停止通知。幂等；之后可再次 start()。"""
        timer, self._timer = self._timer, None
        if timer is not None:
            try:
                timer.invalidate()
            except Exception as exc:
                print(f"RUG-WARN IconSensor.stop: NSTimer 清理异常: {exc}", flush=True)
        stream, fs_mod = self._fs_stream, self._fs_mod
        self._fs_stream, self._fs_mod = None, None
        if stream is not None and fs_mod is not None:
            try:
                fs_mod.FSEventStreamStop(stream)
                fs_mod.FSEventStreamInvalidate(stream)
                fs_mod.FSEventStreamRelease(stream)
            except Exception as exc:
                print(f"RUG-WARN IconSensor.stop: FSEvents 流清理异常: {exc}", flush=True)
        self._dirty.clear()
        self._on_change = None
        self._poll_snapshot = None

    def _start_fs_stream(self, fs_mod) -> None:
        """创建并启动 FSEvents 只读流。回调在 FSEvents 系统线程：只置脏标志。"""
        sensor = self

        def _stream_callback(_stream, _client_info, _num_events, _paths, _flags, _ids):
            sensor._dirty.set()
            return None

        stream = fs_mod.FSEventStreamCreate(
            None,
            _stream_callback,
            None,
            [self._desktop_path],
            fs_mod.kFSEventStreamEventIdSinceNow,
            _FS_LATENCY_S,
            fs_mod.kFSEventStreamCreateFlagNone,
        )
        if not stream:
            raise RuntimeError("FSEventStreamCreate 返回空流")
        scheduled = False
        try:
            run_loop = NSRunLoop.currentRunLoop().getCFRunLoop()  # 主线程 run loop
            fs_mod.FSEventStreamScheduleWithRunLoop(
                stream, run_loop, fs_mod.kCFRunLoopDefaultMode
            )
            scheduled = True
            if not fs_mod.FSEventStreamStart(stream):
                raise RuntimeError("FSEventStreamStart 返回失败")
        except Exception:
            if scheduled:
                try:
                    fs_mod.FSEventStreamInvalidate(stream)
                except Exception:
                    pass
            try:
                fs_mod.FSEventStreamRelease(stream)
            except Exception:
                pass
            raise
        self._fs_stream = stream

    def _snapshot(self) -> dict[str, tuple[int, int, bool]]:
        """desktop_path 只读快照：name -> (mtime_ns, size, is_dir)。

        `_SKIP_NAME`（.DS_Store）在 stat 之前就跳过（安全红线 S2：不读不写，
        连 stat 都不做）。
        """
        snap: dict[str, tuple[int, int, bool]] = {}
        with os.scandir(self._desktop_path) as entries:
            for entry in entries:
                if entry.name == _SKIP_NAME:  # S2：绝不 stat .DS_Store
                    continue
                try:
                    st = entry.stat(follow_symlinks=False)
                except OSError:
                    continue  # 竞态：条目已消失，下轮对比自然收敛
                snap[entry.name] = (
                    st.st_mtime_ns,
                    st.st_size,
                    entry.is_dir(follow_symlinks=False),
                )
        return snap

    def tick_(self, _timer=None) -> None:
        """NSTimer("tick:") 回调（主线程）：轮询降级检测 + 消费脏标志 → on_change()。

        也可被单测直接调用。脏标志由 FSEvents 回调（系统线程）或本函数的轮询分支置位；
        消费后清除，保证一次变化至多触发一次 on_change。
        """
        if self._poll_snapshot is not None:
            try:
                snapshot = self._snapshot()
            except OSError as exc:
                print(f"RUG-WARN IconSensor: 桌面轮询失败: {exc}", flush=True)
            else:
                if snapshot != self._poll_snapshot:
                    self._poll_snapshot = snapshot
                    self._dirty.set()
        if self._dirty.is_set():
            self._dirty.clear()
            on_change = self._on_change
            if on_change is not None:
                try:
                    on_change()
                except Exception as exc:
                    print(f"RUG-ERROR IconSensor: on_change 回调异常: {exc}", flush=True)

    # ------------------------------------------------------------------
    # 标定
    # ------------------------------------------------------------------
    def calibration(self) -> tuple[float, float]:
        """一次性标定，返回 (dx, dy)：screen_point = finder_point + (dx, dy)。

        AppleScript 只读 Finder 桌面窗口 bounds，取 left/top 为偏移；结果实例内缓存。
        全部候选失败 → IconSensorError（调用方 RugApp 回退 (0.0, 24.0) 并打 RUG-WARN）。
        """
        if self._calibration_cache is not None:
            return self._calibration_cache
        last_error: IconSensorError | None = None
        for label, script in _CALIB_SCRIPTS:
            try:
                left, top, right, bottom = _parse_bounds_output(self._osascript(script))
                if not all(math.isfinite(v) for v in (left, top, right, bottom)):
                    raise ValueError(f"bounds 含非有限值: {(left, top, right, bottom)}")
                if right <= left or bottom <= top:
                    raise ValueError(f"bounds 不是有效矩形: {(left, top, right, bottom)}")
            except (IconSensorError, ValueError) as exc:
                last_error = exc if isinstance(exc, IconSensorError) else IconSensorError(str(exc))
                continue
            dx, dy = float(left), float(top)
            self._calibration_cache = (dx, dy)
            print(f"RUG-INFO IconSensor.calibration: dx={dx:.1f} dy={dy:.1f} ({label})", flush=True)
            return self._calibration_cache
        raise IconSensorError(f"Finder 桌面窗口标定失败: {last_error}")
