# archive/ — 历史产物留档

这里放的是**已经被取代、不要再使用**的打包脚本与文档，留档只为追溯（2026-10-09 整理）。

| 文件 | 为什么归档 |
| --- | --- |
| `create_app.sh` | 生成「Desktop Rug.app」空壳。启动器路径算错（从 `Contents/MacOS` 上溯三级到 `dist/`，再 `./run.sh`——文件不存在），双击**必然毫无反应** |
| `create_final_app.sh` | 生成「桌面毛毯.app」空壳。启动器把项目绝对路径 `cd /Users/lex/Code/ProjStudy/desktop-blanket` 写死 → 拖到 /Applications 也依赖本仓库；发给别人 100% 打不开 |
| `setup.py` | py2app 打包配置。本项目 venv 是 uv 的 python-build-standalone，py2app 0.28 会崩（`zlib.__file__` AttributeError）→ 构建从来没能成功产出 app |
| `start_rug.command`、`桌面毛毯-桌面临时启动.command` | 桌面双击跑脚本的临时办法（会开一个终端窗口），已被真正的 app 取代 |
| `打包说明.md`、`快速开始打包.md` | 教人跑 py2app（`./build_app.sh` 旧版）——该路线不可用，且文档描述与产物不符 |
| `应用安装完成.md`、`应用使用指南.md` | 声称 app 已装在 `/Applications/桌面毛毯.app`、`/Applications/Desktop Rug.app`——实际两者都不存在；且描述「点开 app 后要先去菜单栏点铺上」（现在是启动即铺上） |

> 三个空壳 app **共用同一个 `CFBundleIdentifier`（com.desktop.rug）**，是「启动台里出现两个应用 / 幽灵条目」的直接来源。现在的唯一打包入口是仓库根目录的 `build_app.sh`，产物 `/Applications/桌面毛毯.app` + `dist/桌面毛毯-<版本>.dmg`。
