/* launcher.c — 「桌面毛毯.app」的主可执行文件（Mach-O）。
 *
 * 职责：把本进程替换成包内自带的 Python 运行时去执行包内 rug.py，
 * 与仓库目录 / .venv / 系统 Python 完全无关（app 自包含）。
 *
 * 为什么必须是 Mach-O 而不是 shell 脚本（2026-10-09 实测）：
 *   主可执行文件是脚本时 `codesign` 无法封存 bundle，`codesign -v` 报
 *   "a sealed resource is missing or invalid"——这种「半有效签名 + 隔离属性」
 *   在别人机器上会表现为「应用已损坏，请移到废纸篓」（比完全未签名更难恢复：
 *   右键打开救不回来）。换成 Mach-O 后 codesign -v 通过（valid on disk）。
 *
 * 路径处理：_NSGetExecutablePath 拿到自身路径 → dirname → Contents/MacOS →
 * 上一级 Contents → realpath 规范化（避免 argv 里带 ".."，Python 的运行时
 * 前缀识别依赖干净的绝对路径）→ execv 到
 *   <Contents>/Resources/runtime/bin/python3.11 <Contents>/Resources/app/rug.py <原参数>
 * execv 让 python 顶替本进程成为 app 主进程：LaunchServices / TCC 归属
 * （Finder 自动化权限弹窗显示「桌面毛毯」而不是某个 python）都算在这个 app 上。
 *
 * PYTHONDONTWRITEBYTECODE：包内代码在运行时写 __pycache__ 里的 .pyc 会改动 app 包内容、
 * 破坏 codesign 封存（codesign -v 报 "file added"→ 别人机器上可能报「应用已损坏」）。
 * .pyc 已在构建时预编译并随包签名，运行时只读不改。
 */
#include <libgen.h>
#include <mach-o/dyld.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char **argv) {
    char exe[4096];
    uint32_t size = sizeof(exe);
    if (_NSGetExecutablePath(exe, &size) != 0) {
        return 1;
    }

    char *macos_dir = dirname(exe);                     /* .../Contents/MacOS */
    char contents[4096], resolved[4096];
    snprintf(contents, sizeof(contents), "%s/..", macos_dir);
    if (realpath(contents, resolved) == NULL) {
        return 1;
    }

    char py[4096], script[4096];
    snprintf(py, sizeof(py), "%s/Resources/runtime/bin/python3.11", resolved);
    snprintf(script, sizeof(script), "%s/Resources/app/rug.py", resolved);

    setenv("PYTHONDONTWRITEBYTECODE", "1", 1);
    if (getenv("PATH") == NULL) {   /* Finder 启动时一般有默认 PATH；缺失时补上（osascript 要用） */
        setenv("PATH", "/usr/bin:/bin:/usr/sbin:/sbin", 1);
    }

    char **newargv = malloc(sizeof(char *) * (argc + 2));
    if (newargv == NULL) {
        return 1;
    }
    newargv[0] = py;
    newargv[1] = script;
    for (int i = 1; i < argc; i++) {
        newargv[i + 1] = argv[i];
    }
    newargv[argc + 1] = NULL;

    execv(py, newargv);
    perror("execv");   /* 只在 exec 失败时到达 */
    return 1;
}
