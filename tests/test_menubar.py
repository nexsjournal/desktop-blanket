"""tests/test_menubar.py — menubar.py 的离屏回归（不建窗口、不跑事件循环、不弹窗）。

锁三件事：
  ① 落位判定 `settled()`：macOS 26 未落位时读到的是 34×0 @ (0,0) 的假值，必须判为「未落位」；
  ② 交接报告文件 `write_report` / `wait_report` 的 token 匹配与原子写（父进程只认自己那轮）；
  ③ 菜单栏模板图标 `template_image` 三档 rep + template 标记；缺文件时返回 None（调用方退回 icns）。

真机部分（LaunchServices 启动 vs shell 启动的可见性差异、交接真的能拿到落位）无法离屏覆盖，
见 menubar.py 顶部实测记录与 scripts/probe_status_item.py。
"""
from __future__ import annotations

from pathlib import Path

import pytest

menubar = pytest.importorskip("menubar", reason="menubar.py 依赖 AppKit/Foundation")
settings_ui = pytest.importorskip("settings_ui")

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- 替身
class _FakeWindow:
    def __init__(self, x=0.0, y=0.0, w=34.0, h=0.0, visible=True):
        self._x, self._y, self._w, self._h, self._v = x, y, w, h, visible

    def frame(self):
        from Foundation import NSMakeRect
        return NSMakeRect(self._x, self._y, self._w, self._h)

    def isVisible(self):
        return self._v


class _FakeItem:
    def __init__(self, window):
        self._window = window

    def button(self):
        return self

    def window(self):
        return self._window


def _item(x=0.0, y=0.0, w=34.0, h=0.0, visible=True):
    return _FakeItem(_FakeWindow(x, y, w, h, visible))


# --------------------------------------------------------------------------- ① 落位判定
def test_settled_false_for_degenerate_window():
    """未落位时是 34×0 @ (0,0)（macOS 26 实测假值）——不能当成可见。"""
    assert menubar.settled(_item()) is False
    assert menubar.read_frame(_item())[3] == 0.0


def test_settled_true_for_real_menu_bar_geometry():
    """真机落位实测：x=1101 y=1130 34×39（y 是翻转坐标下的菜单栏顶部）。"""
    assert menubar.settled(_item(1101.0, 1130.0, 34.0, 39.0)) is True


def test_settled_false_when_window_hidden_or_missing():
    assert menubar.settled(_item(1101.0, 1130.0, 34.0, 39.0, visible=False)) is False
    assert menubar.read_frame(_FakeItem(None)) is None
    assert menubar.settled(None) is False


def test_await_slot_immediate_paths():
    """max_s=0：落位就直接返回 True（不泵 runloop），没落位立刻 False。"""
    ok, dt = menubar.await_slot(_item(1101.0, 1130.0, 34.0, 39.0), max_s=0.0)
    assert ok is True and dt >= 0.0
    ok, _ = menubar.await_slot(_item(), max_s=0.0)
    assert ok is False


# --------------------------------------------------------------------------- ② 交接报告
def test_report_path_next_to_settings(tmp_path):
    assert menubar.report_path(str(tmp_path / "settings.json")) == \
        str(tmp_path / "menubar-handoff.json")


def test_write_and_wait_report_token_match(tmp_path):
    path = str(tmp_path / "menubar-handoff.json")
    menubar.write_report(path, "tok-1", True, [1101.0, 1130.0, 34.0, 39.0, True])
    got = menubar.wait_report(path, "tok-1", timeout=0.5)
    assert got is not None and got["visible"] is True and got["token"] == "tok-1"
    assert got["frame"] == [1101.0, 1130.0, 34.0, 39.0, True]
    # 旧 token 不能被认领（父进程只认自己那轮）
    assert menubar.wait_report(path, "tok-old", timeout=0.05) is None


def test_wait_report_timeout_without_file(tmp_path):
    assert menubar.wait_report(str(tmp_path / "nope.json"), "tok", timeout=0.05) is None


def test_write_report_failure_is_swallowed(tmp_path):
    """写不进去（父目录不存在）只当没写成，不抛——交接失败会走提示路径。"""
    menubar.write_report(str(tmp_path / "missing" / "x.json"), "tok", False, None)


# --------------------------------------------------------------------------- ③ 图标与配置
def test_template_image_from_repo_assets():
    img = menubar.template_image(str(ROOT / "assets" / "logo"))
    assert img is not None, "assets/logo/menubar*.png 缺失（跑 scripts/make_logo_assets.py）"
    assert bool(img.isTemplate()) is True, "菜单栏图标必须是 template（系统按菜单栏配色反色）"
    assert len(img.representations()) >= 3, "应有 @1x/@2x/@3x 三档 rep"


def test_template_image_missing_dir_returns_none(tmp_path):
    assert menubar.template_image(str(tmp_path / "nope")) is None


def test_spawn_handoff_bad_executable_returns_none_pid():
    """子进程起不来 → (token, None)，调用方走提示而不是崩。"""
    token, pid = menubar.spawn_handoff(["/nonexistent/python", "-c", ""], "/tmp/x.json")
    assert isinstance(token, str) and len(token) == 12
    assert pid is None


def test_version_string_is_str():
    assert isinstance(menubar.version_string(), str)
    assert menubar.version_string() != ""


def test_settings_schema_has_hint_key():
    """「每版本最多提示一次」依赖 settings 里这个键，别在改动中把它弄丢。"""
    assert settings_ui.DEFAULTS.get("menubar_hint_version") == ""
