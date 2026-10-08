"""tests/test_icons.py — IconSensor 契约验收（contract.md §3B/§5 + interfaces.py icons 部分）。

全程 mock：不真实调用 osascript、不创建 FSEvents 流、不读 ~/Desktop
（IconSensor 一律以 desktop_path=tmp_path 构造）。被测实现 icons.py 只读。

测试策略：
  - AppleScript 输出解析有模块级入口 _parse_icons_output / _parse_bounds_output → 直接单测；
  - query()/calibration() 通过替换 icons 模块内的 subprocess 命名空间 mock 子进程；
  - start()/stop() 通过 monkeypatch sys.modules["FSEvents"]=None 强制走降级轮询分支，
    并用桩 NSTimer 避免真实 run loop 依赖；on_change 触发用实现提供的 tick_() 直接驱动。
"""
from __future__ import annotations

import subprocess
import sys
import types

import pytest

m = pytest.importorskip("icons")

icons = m
IconSensor = m.IconSensor
IconSensorError = m.IconSensorError
parse_icons = m._parse_icons_output
parse_bounds = m._parse_bounds_output

# 安全红线（开发文档 §4）：AppleScript 只读 —— 不得出现任何写操作动词/越权目标
_FORBIDDEN_TOKENS = (
    "make new", "delete", "erase", "duplicate", "move ", "remove ",
    "set position", "desktoppicture", "defaults write", "killall",
)


def _assert_readonly_script(script: str) -> None:
    low = script.lower()
    for token in _FORBIDDEN_TOKENS:
        assert token not in low, f"AppleScript 含疑似写操作/越权目标: {token!r}"


def make_sensor(tmp_path) -> IconSensor:
    # desktop_path 只指向 tmp_path，绝不触碰真实 ~/Desktop
    return IconSensor(desktop_path=str(tmp_path))


def patch_subprocess(monkeypatch, run_fn) -> None:
    """把 icons 模块内的 subprocess 替换为仅含 run/TimeoutExpired 的桩命名空间。

    （icons.py 顶层 `import subprocess` 后按模块属性使用；替换 icons.subprocess
    只影响被测模块，不污染全局 subprocess。TimeoutExpired 保留真类供 except 匹配。）
    """
    monkeypatch.setattr(
        icons, "subprocess",
        types.SimpleNamespace(run=run_fn, TimeoutExpired=subprocess.TimeoutExpired),
        raising=True,
    )


class _StubTimer:
    """NSTimer 桩：记录创建与 invalidate，避免测试依赖真实 run loop。"""

    def __init__(self) -> None:
        self.created_with: tuple = ()
        self.invalidated = False

    @classmethod
    def scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(cls, *args):
        t = cls()
        t.created_with = args
        return t

    def invalidate(self) -> None:
        self.invalidated = True


# ---------------------------------------------------------------------------
# _parse_icons_output（AppleScript 输出解析，模块级入口直接单测）
# ---------------------------------------------------------------------------
def test_parse_icons_output_multi_record():
    out = "3\na.txt\n1 2\nb.txt\n3.5 4.5\nc.txt\n0 0\n"
    assert parse_icons(out) == [
        ("a.txt", 1.0, 2.0),
        ("b.txt", 3.5, 4.5),
        ("c.txt", 0.0, 0.0),
    ]


def test_parse_icons_output_zero_records_and_trailing_newlines():
    assert parse_icons("0\n") == []
    assert parse_icons("0\n\n\n") == []          # 结尾多余空行应被容忍
    assert parse_icons("1\nx.txt\n7 9\n\n\n") == [("x.txt", 7.0, 9.0)]


def test_parse_icons_output_empty_output_raises():
    with pytest.raises(ValueError):
        parse_icons("")


def test_parse_icons_output_first_line_not_count_raises():
    with pytest.raises(ValueError):
        parse_icons("abc\nx.txt\n1 2\n")


def test_parse_icons_output_line_count_mismatch_raises():
    """声明 2 条但只有 1 组数据 → 行数与记录数不符 → ValueError。"""
    with pytest.raises(ValueError):
        parse_icons("2\na.txt\n1 2\n")


def test_parse_icons_output_non_numeric_coordinate_raises():
    with pytest.raises(ValueError):
        parse_icons("1\nx.txt\nabc 5\n")


def test_parse_icons_output_non_finite_coordinate_raises():
    """nan/inf 可被 float() 解析但非有限值 → 必须拒绝（宁降级不出错数据）。"""
    with pytest.raises(ValueError):
        parse_icons("1\nx.txt\nnan 5\n")
    with pytest.raises(ValueError):
        parse_icons("1\nx.txt\ninf -3\n")


def test_parse_icons_output_empty_name_raises():
    with pytest.raises(ValueError):
        parse_icons("1\n\n5 6\n")


# ---------------------------------------------------------------------------
# _parse_bounds_output（标定输出解析）
# ---------------------------------------------------------------------------
def test_parse_bounds_output_ok():
    assert parse_bounds("{0, 24, 1440, 900}") == (0.0, 24.0, 1440.0, 900.0)
    assert parse_bounds("  {10.5, 24, 1450.5, 900}\n") == (10.5, 24.0, 1450.5, 900.0)


def test_parse_bounds_output_invalid_raises():
    with pytest.raises(ValueError):
        parse_bounds("1 2 3")                       # 不是 4 个数
    with pytest.raises(ValueError):
        parse_bounds("{1, 2, missing value, 4}")    # 含非数值


# ---------------------------------------------------------------------------
# query()：mock osascript 子进程
# ---------------------------------------------------------------------------
def test_query_happy_path_and_readonly_osascript_call(tmp_path, monkeypatch):
    """正常查询：osascript 参数合法、脚本只读、size_bytes 来自本地 os.stat。"""
    (tmp_path / "notes.txt").write_bytes(b"x" * 1234)
    calls: list = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return types.SimpleNamespace(returncode=0, stdout="1\nnotes.txt\n100 200\n", stderr="")

    patch_subprocess(monkeypatch, fake_run)
    infos = make_sensor(tmp_path).query()

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] == "osascript"          # 走 osascript 子进程
    assert "-e" in args
    script = args[-1]
    _assert_readonly_script(script)        # 安全断言：脚本无写操作动词
    assert 3.0 <= kwargs.get("timeout", 0) <= 7.0   # 契约：osascript 超时 ≈5s

    assert len(infos) == 1
    info = infos[0]
    assert info.name == "notes.txt"
    assert info.x_pt == pytest.approx(100.0)
    assert info.y_pt == pytest.approx(200.0)
    assert info.size_bytes == 1234         # os.stat(tmp_path/notes.txt).st_size


def test_query_stat_missing_file_size_zero(tmp_path, monkeypatch):
    """契约：os.stat 失败（文件已消失）→ size_bytes == 0，不抛异常。"""

    def fake_run(args, **kwargs):
        return types.SimpleNamespace(returncode=0, stdout="1\nghost.txt\n50 60\n", stderr="")

    patch_subprocess(monkeypatch, fake_run)
    infos = make_sensor(tmp_path).query()  # tmp_path 中不存在 ghost.txt
    assert len(infos) == 1
    assert infos[0].name == "ghost.txt"
    assert infos[0].size_bytes == 0


def test_query_skips_ds_store_and_never_stats_it(tmp_path, monkeypatch):
    """安全红线 S2：.DS_Store 不进结果、绝不 stat。"""
    (tmp_path / "a.txt").write_text("hello")
    sensor = make_sensor(tmp_path)
    stat_calls: list[str] = []
    real_stat = sensor._stat_size

    def recording_stat(name: str) -> int:
        stat_calls.append(name)
        return real_stat(name)

    sensor._stat_size = recording_stat

    def fake_run(args, **kwargs):
        return types.SimpleNamespace(
            returncode=0,
            stdout="2\n.DS_Store\n10 10\na.txt\n20 30\n",
            stderr="",
        )

    patch_subprocess(monkeypatch, fake_run)
    infos = sensor.query()
    assert [i.name for i in infos] == ["a.txt"]
    assert infos[0].size_bytes == 5
    assert ".DS_Store" not in stat_calls


def test_query_permission_denied_1743_raises_with_reason(tmp_path, monkeypatch):
    """§5：AppleEvent -1743（自动化权限被拒）→ IconSensorError，message 含原因。"""

    def fake_run(args, **kwargs):
        return types.SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="execution error: Not authorized to send Apple events to Finder (-1743).",
        )

    patch_subprocess(monkeypatch, fake_run)
    with pytest.raises(IconSensorError) as ei:
        make_sensor(tmp_path).query()
    assert "-1743" in str(ei.value)


def test_query_timeout_raises(tmp_path, monkeypatch):
    """§5：osascript 超时（TimeoutExpired）→ IconSensorError。"""

    def fake_run(args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args, timeout=5.0)

    patch_subprocess(monkeypatch, fake_run)
    with pytest.raises(IconSensorError):
        make_sensor(tmp_path).query()


def test_query_parse_failure_raises(tmp_path, monkeypatch):
    """§5：osascript 成功但输出行数与记录数不符（解析失败）→ IconSensorError。"""

    def fake_run(args, **kwargs):
        return types.SimpleNamespace(returncode=0, stdout="2\na.txt\n1 2\n", stderr="")

    patch_subprocess(monkeypatch, fake_run)
    with pytest.raises(IconSensorError) as ei:
        make_sensor(tmp_path).query()
    assert "解析失败" in str(ei.value)


def test_query_generic_script_failure_raises(tmp_path, monkeypatch):
    """osascript 非零返回且无 -1743 → 仍必须报 IconSensorError（含 stderr 原因）。"""

    def fake_run(args, **kwargs):
        return types.SimpleNamespace(returncode=1, stdout="", stderr="syntax error: blah")

    patch_subprocess(monkeypatch, fake_run)
    with pytest.raises(IconSensorError) as ei:
        make_sensor(tmp_path).query()
    assert "syntax error" in str(ei.value)


def test_query_osascript_launch_failure_raises(tmp_path, monkeypatch):
    """osascript 无法启动（OSError）→ IconSensorError。"""

    def fake_run(args, **kwargs):
        raise OSError("osascript binary missing")

    patch_subprocess(monkeypatch, fake_run)
    with pytest.raises(IconSensorError):
        make_sensor(tmp_path).query()


# ---------------------------------------------------------------------------
# calibration()
# ---------------------------------------------------------------------------
def test_calibration_success_cached_and_readonly(tmp_path, monkeypatch):
    """标定成功：取 bounds left/top 为 (dx,dy)；结果实例内缓存（只调一次子进程）。"""
    calls: list[str] = []

    def fake_run(args, **kwargs):
        script = args[-1]
        calls.append(script)
        _assert_readonly_script(script)   # 标定脚本同样必须只读
        return types.SimpleNamespace(returncode=0, stdout="{0, 24, 1440, 900}\n", stderr="")

    patch_subprocess(monkeypatch, fake_run)
    sensor = make_sensor(tmp_path)
    assert sensor.calibration() == (0.0, 24.0)
    assert sensor.calibration() == (0.0, 24.0)   # 第二次命中缓存
    assert len(calls) == 1


def test_calibration_all_candidates_fail_raises(tmp_path, monkeypatch):
    """两个候选脚本全部失败（子进程报错）→ IconSensorError，且候选都被尝试过。"""
    calls: list = []

    def fake_run(args, **kwargs):
        calls.append(args[-1])
        return types.SimpleNamespace(returncode=1, stdout="", stderr="some appleevent error")

    patch_subprocess(monkeypatch, fake_run)
    with pytest.raises(IconSensorError):
        make_sensor(tmp_path).calibration()
    assert len(calls) == len(icons._CALIB_SCRIPTS)   # 逐个候选都试过


def test_calibration_invalid_bounds_raises(tmp_path, monkeypatch):
    """bounds 不是有效矩形（right<=left）→ 视为失败 → IconSensorError。"""

    def fake_run(args, **kwargs):
        return types.SimpleNamespace(returncode=0, stdout="{10, 20, 5, 20}", stderr="")

    patch_subprocess(monkeypatch, fake_run)
    with pytest.raises(IconSensorError):
        make_sensor(tmp_path).calibration()


# ---------------------------------------------------------------------------
# start / stop / 变化通知（FSEvents 不可用 → 降级轮询）
# ---------------------------------------------------------------------------
def _force_polling(monkeypatch) -> None:
    """sys.modules 里塞 None → `import FSEvents` 抛 ImportError → 实现走降级轮询。"""
    monkeypatch.setitem(sys.modules, "FSEvents", None)
    monkeypatch.setattr(icons, "NSTimer", _StubTimer, raising=True)


def test_start_fallback_polling_triggers_on_change(tmp_path, monkeypatch, capsys):
    """FSEvents 不可用 → 降级为主线程轮询且不崩；桌面变化经 tick_ 触发一次 on_change。"""
    _force_polling(monkeypatch)
    sensor = make_sensor(tmp_path)  # 空目录
    fired: list[int] = []
    sensor.start(lambda: fired.append(1))

    # 降级路径有 RUG-WARN 日志（§5 约定：降级打 WARN 继续跑）
    out = capsys.readouterr().out
    assert "RUG-WARN" in out

    sensor.tick_()                    # 无变化 → 不触发
    assert fired == []

    (tmp_path / "new.txt").write_text("hi")   # 桌面出现新文件
    sensor.tick_()                    # 快照对比发现变化 → 恰好触发一次
    assert fired == [1]

    sensor.tick_()                    # 快照已更新 → 不重复触发
    assert fired == [1]

    sensor.stop()
    sensor.tick_()                    # stop 后不再触发、不崩
    assert fired == [1]


def test_start_stop_idempotent_and_clean(tmp_path, monkeypatch, capsys):
    """start/stop 幂等：stop 可先于 start 调用；重复 start 先停旧流；重复 stop 无害。"""
    _force_polling(monkeypatch)
    sensor = make_sensor(tmp_path)
    fired: list[int] = []
    sensor.stop()                     # 未 start 先 stop：幂等、不崩

    sensor.start(lambda: fired.append(1))
    timer1 = sensor._timer
    assert isinstance(timer1, _StubTimer)
    assert not timer1.invalidated

    sensor.start(lambda: fired.append(1))     # 重复 start：先停旧 timer
    assert timer1.invalidated
    timer2 = sensor._timer
    assert isinstance(timer2, _StubTimer) and timer2 is not timer1

    sensor.stop()
    sensor.stop()                     # 重复 stop：幂等
    assert timer2.invalidated
    assert fired == []                # 全程未产生虚假回调


# ---------------------------------------------------------------------------
# AppleScript 候选脚本「本机可编译」回归（真机踩坑 2026-10-08）
# ---------------------------------------------------------------------------
def _osacompile_ok(script: str):
    """用 osacompile 只编译脚本（不发 Apple Event、不需要权限、不弹窗）。

    返回 True/False；本机没有 osacompile 时返回 None（调用方 skip）。
    """
    import os as _os
    import tempfile
    exe = "/usr/bin/osacompile"
    if not _os.path.exists(exe):
        return None
    with tempfile.TemporaryDirectory() as d:
        src = _os.path.join(d, "s.applescript")
        out = _os.path.join(d, "s.scpt")
        with open(src, "w", encoding="utf-8") as f:
            f.write(script)
        proc = subprocess.run([exe, "-o", out, src], capture_output=True,
                              text=True, timeout=60)
        return proc.returncode == 0


def test_iconscript_candidates_compile_on_this_mac():
    """桌面项查询的每个候选脚本都必须能在本机编译通过。

    回归背景：旧写法 `(get desktop items)` 在 macOS 26 上编译失败（error -2741
    "Expected \",\" but found plural class name"，Finder 词典不再接受该复数类名），
    导致运行时整体降级为「无隆起」。此测试只编译（不触发权限弹窗），
    未来 macOS 再改词典时可第一时间发现。
    """
    for label, script in m._ICON_SCRIPTS:
        ok = _osacompile_ok(script)
        if ok is None:
            pytest.skip("本机没有 /usr/bin/osacompile")
        assert ok, f"图标查询候选脚本编译失败: {label}"


def test_calib_script_candidates_compile_on_this_mac():
    """标定候选脚本同样必须能编译（同一类字典兼容问题的回归）。"""
    for label, script in m._CALIB_SCRIPTS:
        ok = _osacompile_ok(script)
        if ok is None:
            pytest.skip("本机没有 /usr/bin/osacompile")
        assert ok, f"标定候选脚本编译失败: {label}"


def test_query_falls_back_to_next_candidate(monkeypatch):
    """候选脚本回退：第 1 个候选「编译失败」时自动改用第 2 个；致命的权限错误立即上抛。"""
    calls: list[str] = []

    def fake_osascript(script: str) -> str:
        calls.append(script)
        if "items of the desktop" in script:
            raise IconSensorError('osascript 失败 (rc=1): syntax error: -*-2741***')
        return "1\nfoo.txt\n10 20\n"

    sensor = IconSensor(desktop_path="/tmp")
    monkeypatch.setattr(sensor, "_osascript", fake_osascript)
    items = sensor.query()
    assert len(calls) == 2, "应在第一个候选失败后尝试第二个"
    assert [i.name for i in items] == ["foo.txt"]

    calls.clear()

    def deny(script: str) -> str:
        calls.append(script)
        raise IconSensorError("自动化权限被拒（AppleEvent -1743）")

    monkeypatch.setattr(sensor, "_osascript", deny)
    with pytest.raises(IconSensorError):
        sensor.query()
    assert len(calls) == 1, "-1743 属致命错误，不应继续尝试其它候选（避免多弹权限窗）"
