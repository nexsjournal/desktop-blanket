# assets/logo — 品牌图标

**唯一来源**：`applogo.svg`。改 logo = 替换这个文件，然后重建资产（不要手改生成物）。

```bash
.venv/bin/python scripts/make_logo_assets.py --check   # 终端 ASCII 预览，不写文件
.venv/bin/python scripts/make_logo_assets.py           # 重建下面这些
```

| 文件 | 是什么 | 谁用 |
| --- | --- | --- |
| `applogo.svg` | **源**：品牌 SVG（1024×1024，近黑底板 + 白色标记） | 人工维护 |
| `menubar.png` / `@2x` / `@3x` | 菜单栏**模板图标**（18pt，黑色 + alpha 蒙版；标记里的半透明层次保留在 alpha 里） | `menubar.py` 装进 `NSStatusItem`（template，系统按菜单栏配色反色） |
| `logo-mark-1024.png` | 透明底标记（去掉底板） | 文档 / 宣传素材 |
| （仓库根）`icon.icns` + `icon.iconset/` + `icon_temp.png` | macOS 应用图标：1024 画布内 824×824 圆角主体，按各档像素现场矢量渲染 | `build_app.sh` → 包内 `AppIcon.icns`（Finder/Dock/DMG） |

注意：

- 生成脚本用 **AppKit 的 NSImage 直接读 SVG**（macOS 13+ 支持），所以矢量锐度保留到 16px；需要 macOS 13+ 才能重建资产（打包本来就是 macOS 专案）。
- 菜单栏图标**必须**是 template（黑 + alpha）：彩色图在浅色菜单栏上会糊成一团。生成脚本按 alpha 通道转换，SVG 里的 `fill-opacity` 会变成对应的半透明档位。
- 菜单栏图标与「落位判定/交接」的排障见开发文档 §12.20；探针脚本 `scripts/probe_status_item.py`。
