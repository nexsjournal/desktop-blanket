#!/usr/bin/env python3
"""safety_check.py — 桌面毛毯安全自验工具（integrator 专有，contract.md §4 安全红线）。

用途：在真机 selftest / 手测前后各取一次快照，compare 证明运行过程没有动
桌面环境（新旧两代壁纸存储、桌面文件、Finder 桌面项位置）。

子命令：
  snapshot   输出 JSON 到 stdout（重定向保存）：
             - desktoppicture.db 的 md5（~/Library/Application Support/Dock/，旧版）
             - com.apple.wallpaper/Store/ 递归清单（相对路径+大小+md5，macOS 26+；
               目录不存在记 exists=False，绝不创建）
             - ~/Desktop 文件名/大小/mtime 排序列表（os.scandir 只读，含隐藏文件）
             - Finder 桌面项位置列表（osascript 只读查 name/position）
             注意：Finder 查询会弹「自动化→Finder」权限窗，默认跳过；
             仅当显式加 --with-finder 标志才执行（无人值守安全）。
  compare B A  逐项 diff 两份快照：全等 → SAFETY-OK；有差异 →
             SAFETY-DIFF <明细>（每条一行）并 exit 1。

非 JSON 的提示一律走 stderr，不污染 stdout。退出码：0 成功/全等，1 有差异，
2 用法或 IO 错误。

安全：本工具自身对 ~/Desktop、desktoppicture.db、com.apple.wallpaper/Store/、
Finder 全程只读；不写任何文件（JSON 由调用方重定向落盘），目录不存在也绝不创建。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

DB_PATH = os.path.expanduser("~/Library/Application Support/Dock/desktoppicture.db")
WALLPAPER_STORE_PATH = os.path.expanduser(
    "~/Library/Application Support/com.apple.wallpaper/Store")
DESKTOP_PATH = os.path.expanduser("~/Desktop")
OSASCRIPT_TIMEOUT_S = 15.0

# 只读查询 Finder 桌面项：name + position（与 icons.py 的查询同款，无任何写指令）。
# 注意：旧写法 `(get desktop items)` 在 macOS 26 上编译失败（error -2741，Finder 词典
# 不再接受该复数类名），改用规范写法 `items of the desktop`（2026-10-08 实测）。
_FINDER_SCRIPT = (
    'tell application "Finder"\n'
    '	set _out to ""\n'
    '	set _n to 0\n'
    '	repeat with f in (get items of the desktop)\n'
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


def _warn(msg: str) -> None:
    print(f"[safety_check] {msg}", file=sys.stderr)


def _md5_file(path: str) -> str | None:
    """文件 md5（分块只读）；文件不存在/不可读 → None。"""
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError as exc:
        _warn(f"md5 失败 {path}: {exc}")
        return None


def _wallpaper_store_files() -> dict[str, dict] | None:
    """递归收集 com.apple.wallpaper/Store/：relpath -> {size, md5}。

    目录不存在 → None（记 exists=False，绝不创建——安全红线：不碰壁纸存储）。
    """
    if not os.path.isdir(WALLPAPER_STORE_PATH):
        return None
    files: dict[str, dict] = {}
    for root, _dirs, names in os.walk(WALLPAPER_STORE_PATH):
        for name in names:
            full = os.path.join(root, name)
            rel = os.path.relpath(full, WALLPAPER_STORE_PATH)
            try:
                st = os.stat(full)
                files[rel] = {"size": int(st.st_size), "md5": _md5_file(full)}
            except OSError as exc:
                _warn(f"wallpaper Store 跳过 {rel}: {exc}")
    return files


def _desktop_files() -> list[dict]:
    """~/Desktop 只读快照：name/size/mtime_ns/is_dir，按文件名排序。

    follow_symlinks=False（不跟出桌面目录）；竞态消失的条目跳过。
    """
    items: list[dict] = []
    with os.scandir(DESKTOP_PATH) as entries:
        for entry in entries:
            try:
                st = entry.stat(follow_symlinks=False)
            except OSError as exc:
                _warn(f"stat 跳过 {entry.name}: {exc}")
                continue
            items.append({
                "name": entry.name,
                "size": int(st.st_size),
                "mtime_ns": int(st.st_mtime_ns),
                "is_dir": bool(entry.is_dir(follow_symlinks=False)),
            })
    items.sort(key=lambda d: d["name"])
    return items


def _finder_items() -> dict | list:
    """osascript 只读查 Finder 桌面项位置；返回 [{name,x,y},...] 或 {"error": ...}。

    会弹「自动化→Finder」权限窗：仅 snapshot --with-finder 时调用。
    """
    try:
        proc = subprocess.run(
            ["osascript", "-e", _FINDER_SCRIPT],
            capture_output=True, text=True, timeout=OSASCRIPT_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"osascript 超时（>{OSASCRIPT_TIMEOUT_S:.0f}s）"}
    except OSError as exc:
        return {"error": f"osascript 启动失败: {exc}"}
    if proc.returncode != 0:
        return {"error": (proc.stderr or "").strip() or f"rc={proc.returncode}"}
    lines = proc.stdout.split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return {"error": "输出为空"}
    try:
        count = int(lines[0].strip())
    except ValueError:
        return {"error": f"首行不是记录数: {lines[0]!r}"}
    if count < 0 or len(lines) != 1 + 2 * count:
        return {"error": f"行数与记录数不符: count={count}, 行={len(lines)}"}
    items: list[dict] = []
    for i in range(count):
        name = lines[1 + 2 * i]
        parts = lines[2 + 2 * i].split()
        if len(parts) != 2:
            return {"error": f"第 {i + 1} 条 position 行格式错误"}
        try:
            x, y = float(parts[0]), float(parts[1])
        except ValueError:
            return {"error": f"第 {i + 1} 条坐标不是数字"}
        items.append({"name": name, "x": x, "y": y})
    items.sort(key=lambda d: d["name"])
    return items


def cmd_snapshot(with_finder: bool) -> int:
    """取安全快照，JSON 输出到 stdout。"""
    snap = {
        "version": 1,
        "generated_at": subprocess.run(
            ["date", "+%Y-%m-%dT%H:%M:%S%z"], capture_output=True, text=True
        ).stdout.strip(),
        "desktoppicture_db": {
            "path": DB_PATH,
            "exists": os.path.exists(DB_PATH),
            "md5": _md5_file(DB_PATH),
        },
        "wallpaper_store": {
            "path": WALLPAPER_STORE_PATH,
            "exists": os.path.isdir(WALLPAPER_STORE_PATH),
            # None = 目录不存在；{} = 空目录；否则 relpath -> {size, md5}
            "files": _wallpaper_store_files(),
        },
        "desktop_files": _desktop_files(),
        # None = 本次未查询（未加 --with-finder）；{"error":...} = 查询失败
        "finder_items": _finder_items() if with_finder else None,
    }
    json.dump(snap, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    if with_finder:
        _warn("已执行 Finder 只读查询（--with-finder）：若弹权限窗请人工确认")
    else:
        _warn("Finder 查询已跳过（默认；加 --with-finder 启用，会弹权限窗）")
    return 0


def _diff_desktop_files(before: list, after: list) -> list[str]:
    b = {d["name"]: d for d in before}
    a = {d["name"]: d for d in after}
    out: list[str] = []
    for name in sorted(b.keys() - a.keys()):
        out.append(f"desktop_files: 消失 {name!r}")
    for name in sorted(a.keys() - b.keys()):
        out.append(f"desktop_files: 新增 {name!r} {a[name]}")
    for name in sorted(b.keys() & a.keys()):
        if b[name] != a[name]:
            out.append(f"desktop_files: 变化 {name!r} before={b[name]} after={a[name]}")
    return out


def cmd_compare(before_path: str, after_path: str) -> int:
    """逐项 diff 两份快照；全等 → SAFETY-OK，有差异 → SAFETY-DIFF 明细 + exit 1。"""
    try:
        with open(before_path, encoding="utf-8") as f:
            b = json.load(f)
        with open(after_path, encoding="utf-8") as f:
            a = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"SAFETY-ERROR 快照读取失败: {exc}", file=sys.stderr)
        return 2

    diffs: list[str] = []
    bdb, adb = b.get("desktoppicture_db") or {}, a.get("desktoppicture_db") or {}
    if bdb.get("md5") != adb.get("md5"):
        diffs.append(
            f"desktoppicture.db md5 变化: {bdb.get('md5')} -> {adb.get('md5')}"
            f"（exists {bdb.get('exists')} -> {adb.get('exists')}）"
        )
    bwp, awp = b.get("wallpaper_store") or {}, a.get("wallpaper_store") or {}
    if bwp.get("exists") != awp.get("exists"):
        diffs.append(
            f"wallpaper_store exists 变化: {bwp.get('exists')} -> {awp.get('exists')}"
        )
    elif bwp.get("exists"):
        bfiles, afiles = bwp.get("files") or {}, awp.get("files") or {}
        for rel in sorted(bfiles.keys() - afiles.keys()):
            diffs.append(f"wallpaper_store: 消失 {rel}")
        for rel in sorted(afiles.keys() - bfiles.keys()):
            diffs.append(f"wallpaper_store: 新增 {rel} {afiles[rel]}")
        for rel in sorted(bfiles.keys() & afiles.keys()):
            if bfiles[rel] != afiles[rel]:
                diffs.append(f"wallpaper_store: 变化 {rel} {bfiles[rel]} -> {afiles[rel]}")
    diffs += _diff_desktop_files(b.get("desktop_files") or [], a.get("desktop_files") or [])

    bf, af = b.get("finder_items"), a.get("finder_items")
    if (bf is None) != (af is None):
        diffs.append(
            "finder_items 快照方式不一致（一边有 Finder 查询一边没有；"
            "请两边都用相同 --with-finder 设置重新取快照）"
        )
    elif bf is not None and bf != af:
        if isinstance(bf, dict) or isinstance(af, dict):
            diffs.append(f"finder_items 查询错误态变化: {bf} -> {af}")
        else:
            bn = {d["name"]: d for d in bf}
            an = {d["name"]: d for d in af}
            for name in sorted(bn.keys() - an.keys()):
                diffs.append(f"finder_items: 消失 {name!r}")
            for name in sorted(an.keys() - bn.keys()):
                diffs.append(f"finder_items: 新增 {name!r} {an[name]}")
            for name in sorted(bn.keys() & an.keys()):
                if bn[name] != an[name]:
                    diffs.append(
                        f"finder_items: 位置变化 {name!r} "
                        f"({bn[name]['x']},{bn[name]['y']}) -> ({an[name]['x']},{an[name]['y']})"
                    )

    if diffs:
        for d in diffs:
            print(f"SAFETY-DIFF {d}")
        return 1
    print("SAFETY-OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="safety_check", description="桌面毛毯安全自验")
    p.add_argument("--with-finder", action="store_true",
                   help="snapshot 时执行 Finder 只读查询（会弹自动化权限窗，默认跳过）")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("snapshot", help="输出安全快照 JSON 到 stdout")
    cmp_p = sub.add_parser("compare", help="比较两份快照")
    cmp_p.add_argument("before")
    cmp_p.add_argument("after")
    args = p.parse_args(argv)
    if args.cmd == "snapshot":
        return cmd_snapshot(with_finder=bool(args.with_finder))
    if args.cmd == "compare":
        return cmd_compare(args.before, args.after)
    p.print_usage(sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
