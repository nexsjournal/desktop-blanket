# 桌面毛毯（Desktop Rug）

![macOS 11+](https://img.shields.io/badge/platform-macOS%2011%2B-000000?logo=apple&logoColor=white)
![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![tests: 144 passed](https://img.shields.io/badge/tests-144%20passed-brightgreen)

在 macOS 桌面上「铺」一块有真实布料物理的毯子，把你的桌面图标盖在下面——**把烂摊子扫到毯子底下**。

一句话说清它是什么：**一个把桌面当成桌面的小玩具**——纯 Python（numpy 自写 3D 布料求解器 + PyObjC/SceneKit 渲染），
窗口层级卡在「Finder 桌面图标之上、所有普通窗口之下」，毯子以外的像素完全穿透，
不碰壁纸、不碰你的文件、不联网，退出即复原。

毯子悬浮在「桌面图标之上、所有普通窗口之下」，可以抓住任意位置拖动、甩出去会飞行旋转落地；拖动时**起大而圆润的波浪/折**、松手后缓缓平息；**拖住一个角横着划过毯身，会把毯子真折起来**（折层压在自己身上、露出变暗的背面——v5 真 3D 布料 + 圆角折痕）；毯子下面的图标会顶起隆包，堆得越高包越大；毯子以外的区域鼠标完全穿透，桌面照常可点。**退出后一切复原**。

> 参考效果：[X 原帖 @terkelg](https://x.com/terkelg/status/2107540718461886464)（视频留档 `refer/`）。原作是 desktop.cleaning 的 "Rugs"，无公开技术细节，本项目为从零自行设计实现。
> 设计/实现细节见 **[开发文档.md](开发文档.md)**（架构、里程碑、安全边界、验收清单）。

---

## 安装

### 只想用（不用编译）

去 **[Releases](https://github.com/nexsjournal/desktop-blanket/releases/latest)** 下载 **`DesktopRug-<版本>.dmg`**：

1. 双击 DMG 打开 → 把「桌面毛毯」拖进 `Applications`（窗口里已放好 `Applications` 快捷方式）；
2. 第一次打开会被 Gatekeeper 拦（应用只有 ad-hoc 签名、未公证）：右键点应用 →「打开」→ 再点「打开」；macOS 15+ 还要去 系统设置 → 隐私与安全性 →「仍要打开」。之后双击即开。
3. 想彻底卸载：应用拖进废纸篓；连设置一起清 `rm -rf ~/Library/Application\ Support/DesktopRug`。

> 发布包只打 **arm64**（Apple Silicon）。Intel Mac 请用下面的源码方式自行构建。

### 自己构建（源码 → app → DMG）

```bash
./build_app.sh        # 构建 → 自检 → 生成 DMG → 安装到 /Applications → 启动
```

| 产物 | 用途 |
| --- | --- |
| `/Applications/桌面毛毯.app` | 双击即用。**自包含**：内置 Python 运行时 + 依赖 + 贴图，不依赖本仓库目录（把仓库删了它照跑） |
| `dist/DesktopRug-6.9.0.dmg` | 发给别人：打开 DMG → 把「桌面毛毯」拖进 Applications |

- **打开应用后屏幕上立刻出现毯子**（默认「启动即铺上」）。应用是**常驻菜单栏**形态（像 Clash）：平时不占 Dock，只在菜单栏放毛毯图标（品牌标记的模板变体，见 `assets/logo/`）；**打开设置窗时才临时进 Dock**（此时 ⌘Q 可用），关窗收回。
- **菜单栏图标一定看得见**（v6.8）：macOS 26 有个坑——`LaunchServices` 启动（= 双击 / 登录项）的进程拿不到菜单栏落位（项被丢到屏幕外，`isVisible=False`），而从命令行启动就正常。应用现在会自检这一点，发现没落位就**自动把活交接给一个非 LS 启动的子进程**，子进程能拿到图标；两路都拿不到时才弹一次说明（指向 系统设置 → 控制中心 → 菜单栏）。机制与实测见 [开发文档 §12.20](开发文档.md)。
- **在毯子上点右键**是**最可靠的入口**（不依赖菜单栏）：掀开毯子／调整尺寸／抚平折痕／重置／设置／退出。
- 构建脚本自带**自检门禁**：末尾用包内运行时跑 `--selftest`——真的把毯子铺上并用 `CGWindowListCopyWindowInfo` 校验窗口层级，必须出现 `SELFTEST-OK` 才产出，否则构建报错退出。
- 应用是 **ad-hoc 签名、未公证**的（没有 Developer ID 证书；自己编自己用没问题）。发给别人时对方首次打开会被 Gatekeeper 拦：右键点应用 →「打开」→ 再点「打开」；macOS 15+ 可能还要去 系统设置 → 隐私与安全性 →「仍要打开」。要做到双击即开不弹窗，需要 Apple 开发者账号（$99/年）做 Developer ID 签名 + 公证——本项目没做。
- 应用数据：设置 `~/Library/Application Support/DesktopRug/settings.json`、日志 `~/Library/Logs/DesktopRug.log`、单实例锁 `instance.lock`（同目录）。包内代码只读，运行时不往 app 包里写任何东西（否则会破坏签名封存）。
- 卸载：`/Applications/桌面毛毯.app` 拖废纸篓即可；连设置一起清 `rm -rf ~/Library/Application\ Support/DesktopRug`。
- 开机自启（可选）：系统设置 → 通用 → 登录项 → 添加 `/Applications/桌面毛毯.app`。

## 快速开始（源码运行／开发）

要求：macOS（Apple Silicon，实测 macOS 26）、Python 3.11+、[uv](https://docs.astral.sh/uv/)（或自备 venv）。

```bash
cd desktop-blanket
uv venv .venv && uv pip install -r requirements.txt   # 首次
./run.sh                 # 启动即铺上毯子；按 F11/显示桌面 就能看见
./run.sh --start-hidden  # 只驻留菜单栏，不自动铺上（旧默认行为）
```

`./run.sh` 启动后 Dock 与**菜单栏**各有毛毯图标（源码运行也设了 Dock 图标）；源码运行时的设置存在项目内 `settings.json`（打包成 app 后才改用 `~/Library/Application Support/DesktopRug/`）。

### 怎么玩

| 操作 | 效果 |
| --- | --- |
| 启动 | 毯子从屏幕左上**带一条弧线**飞入，正放在屏幕中央（默认占屏宽 52%×高 62%）——app 双击、`./run.sh` 都是如此 |
| **在毯子上点右键**（或 Ctrl+左键） | 弹出上下文菜单：掀开毯子（收起）／**调整尺寸**（小·中·大 预设、拖拽调整）／抚平折痕／重置／设置…／退出 |
| 点 Dock 图标 | 毯子没铺就铺上、铺着就掀开（最可发现的开关）；⌘Q 直接退出 |
| 菜单「铺上 ⌘\\」 | 手工再铺一次（掀开之后想铺回来也行） |
| 在毯子上**按住拖动** | 抓住毯子任意位置拖：整块跟手、尾部起皱、边角滞后 |
| **快拖**（快速甩动鼠标） | 抓点被**拎离桌面**（最高 170pt），毯子半悬着**垂挂出大折**——v6.3 新增，对齐参考效果 |
| **甩出去后松手** | 毯子**腾空**飞出（抓点附近抬得高、远端滞后 → 空中垂挂），落地非弹性停下、缓缓平息 |
| 毯子上**单击** | 毯子边缘 ≈40pt 以内算「毯上」：点住即可抓住（光标变手型） |
| 毯子**以外**的桌面 | 完全穿透：单击 / 双击 / 框选图标照常，图标不会失效；右键也是桌面自己的菜单 |
| 菜单「掀开 ⌘\\」 | 毯子被提起扔出屏幕；菜单随后变成 **「摆放回来」**——原样放回你调好的位置/大小/角度 |
| 菜单「调整毯子（大小/角度）⌘E」 | **编辑模式**：毯子四角出现白色锚点（拖动=等比缩放，中心不动）+ 顶边外一个蓝色方块（拖动=旋转）；再按一次 ⌘E 完成 |
| 「折起来」 | 把**毯子一角拖过毯身**（行程过中线）→ 布料被推成**真实折痕/翻折**：折脊是**有体积的圆角**（不是剪纸式锐折）、折层压在自己身上并**露出变暗的背面**，落定后保留 |
| 菜单「抚平折痕（摊平毯子）⌘F」 | 一键把毯子**摊平**回规整矩形（中心/尺寸不变；折返发生在平面内，只清高度不够） |
| 菜单「重置位置/大小/角度 ⌘⇧R」 | 回到默认摆放（屏中心 + 默认尺寸 + 默认角度）；也可用右键菜单「调整尺寸 → 小/中/大」一键换尺寸 |
| 菜单「设置… ⌘,」 | 打开设置窗：换地毯贴图 / 默认大小 / 默认角度 / 图标隆起开关与高度 / 接触阴影 |
| 菜单「地毯款式」 | 快速切换 `assets/rugs/rug-*.png`（与设置窗同一份清单） |
| 菜单「退出 ⌘Q」 | 彻底退出，桌面 100% 原样 |

> 设置保存位置：源码运行 = 项目内 `settings.json`；app 运行 = `~/Library/Application Support/DesktopRug/settings.json`（删掉即回默认；不写系统偏好，符合 §4 红线）。

毯子空闲几秒后会**休眠**（物理停步、渲染只做空检查：空闲 CPU 实测 ≈0.5% 单核），任何交互/桌面变化都会把它唤醒。

### 权限（可选，只有一个）

为了让毯子下面的图标顶起隆包，需要一个**只读**权限：**系统设置 → 隐私与安全性 → 自动化 → 允许「桌面毛毯」（源码运行时是终端/你的启动程序）控制「访达」**。
首次运行若弹出授权窗，允许即可；**拒绝也没关系**——程序自动进入「无隆起模式」（拖甩、波浪、褶皱全保留，只是图标不鼓包），且**不会再弹第二次**。菜单里会标注原因。从 `.app` 启动时弹窗归属于「桌面毛毯」本身（Mach-O 启动器 + 包内运行时保证 TCC 归属正确），不再取决于你从哪个终端启动它。

---

## 验证与自检

```bash
# 1) 单元测试（144 个：布料物理（3D/翻折/自碰撞/休眠/折痕圆角/拖动不塌陷/折层遮蔽）/ 隆起场 / 图标解析 / 渲染层 hitTest+材质+贴图留白检测+**厚度几何（滚边/接触阴影/浮雕/灯光重配）** / 穿透闸门 / 菜单栏模板图标与交接报告 / AppleScript 编译回归）
.venv/bin/python -m pytest tests/ -q

# 2) 无人值守自检：铺上毯子 → 校验窗口层级 → 自动退出（屏幕上会出现 5 秒毯子，无残留）
./run.sh --selftest        # 期望：SELFTEST-LEVEL -2147483595 与 SELFTEST-OK，exit 0

# 3) 安全自验：运行前后两次快照逐项 diff，证明桌面环境零改动
.venv/bin/python scripts/safety_check.py snapshot > /tmp/before.json
./run.sh --selftest
.venv/bin/python scripts/safety_check.py snapshot > /tmp/after.json
.venv/bin/python scripts/safety_check.py compare /tmp/before.json /tmp/after.json   # 期望 SAFETY-OK

# 4) 点击路由机制验证（WindowServer 路由查询：负层级窗口可命中 / ignores 真穿透 / hitTest 不参与路由）
.venv/bin/python scripts/verify_routing.py            # 期望 ROUTE-SELFTEST PASS
.venv/bin/python scripts/verify_routing.py --live     # 期望 ROUTE-LIVE PASS（RECEIVE 方向可能需要屏幕上有可见桌面区）
```

> `scripts/verify_routing.py --live` 依赖一个**验证专用**环境变量 `RUG_CLICKTHROUGH_FORCE=ignore|receive`（由脚本自己注入，把穿透闸门钉死到指定状态，从而不依赖光标当前位置）。生产使用不要设置它。

```bash
# 5) 线上体检（对运行中的实例，任何屏幕状态下都能用）：渲染像素 / 遮挡覆盖率 / 点击路由 / 图标覆盖数
.venv/bin/python scripts/live_probe.py                 # 期望 LIVE-PROBE-RENDER OK（窗口表面不透明 ≥15%）
.venv/bin/python scripts/live_probe.py --save --icons  # 抓图存 /tmp/rug_window_surface.png；附图标覆盖数

# 6) 品牌图标（源 SVG → 应用图标 + 菜单栏模板图标）；--check 只渲染并在终端打 ASCII 预览
.venv/bin/python scripts/make_logo_assets.py --check
.venv/bin/python scripts/make_logo_assets.py           # 重建 icon.icns / assets/logo/menubar*.png

# 7) 菜单栏项排障（macOS 26 的 LS 启动坑，见开发文档 §12.20）
.venv/bin/python scripts/probe_status_item.py 12       # 造几个探针项，另开终端用 AX 量它们的坐标
```

### 还需要你亲手做的验收（10 秒）

自动化能验证的都已验证（见开发文档 §10 与 §12.11：层级、路由、40pt 判定、穿透闸门、安全快照、**真实鼠标抓拖投递**）——剩下的只有「肉眼观感判定」。开始前先保证**屏幕上有可见的桌面区域**，并**把毯子拖到图标上**（隆包只在毯子盖住图标时出现）：

1. 打开 app（或 `./run.sh`）→ 毯子应当**自动铺上**（已在跑就直接用）；挪开窗口 / F11 显示桌面能看到毯子（位置可用 `scripts/live_probe.py` 查）；
2. 把毯子拖到**桌面图标上方**盖住它们 → 看隆包；在毯子上**按住拖动**、再**甩一下**松手 → 看起皱/波浪/飞行是否自然；
3. 在毯子**外面**的桌面**双击**一个图标 → 应正常打开（穿透 ✓）；
4. 在毯子上**单击**再拖动 → 应能抓住（手型光标 ✓）；
5. 菜单「退出」→ 看毯子消失、桌面原样。

---

## 安全保证（硬红线，写进架构）

| # | 红线 | 实现 |
| --- | --- | --- |
| S1 | **壁纸零改动** | 代码不调任何壁纸 API、不碰 `desktoppicture.db` / `com.apple.wallpaper`；毯子只是「漂浮在图标之上的一层窗口」，壁纸原样可见 |
| S2 | **桌面文件零改动** | 对 `~/Desktop` 只有只读访问（`os.stat`/`os.scandir` + osascript **只读**查询名/位置）；不移动/重命名/删除/隐藏任何文件；连 `.DS_Store` 都不 stat |
| S3 | **无系统状态残留** | 所有视觉发生在自建 overlay 窗口内；退出 = 窗口销毁 = 一切消失；不写 LaunchAgent / 登录项 / 全局偏好 |
| S4 | **零网络** | 运行时不发起任何网络请求（贴图是本地文件） |
| S5 | **权限最小化** | 全程只需「自动化 → Finder」一个只读权限；拒绝即降级，不二次弹窗；不需要辅助功能 / 屏幕录制 |
| S6 | **一键彻底退出** | 菜单「退出」或 `Ctrl-C` 即结束进程，无后台驻留 |

`scripts/safety_check.py` 是上面的**可执行证明**：快照覆盖新旧两代壁纸存储（`desktoppicture.db` 与 `com.apple.wallpaper/Store` 递归 md5）、`~/Desktop` 文件清单（名字/大小/mtime）、以及（可选 `--with-finder`）Finder 图标位置；运行前后 diff 为空即通过。

---

## 项目结构

```
desktop-blanket/
├── LICENSE                 # MIT
├── requirements.txt        # 运行时 + 开发依赖（numpy / pyobjc-* / pillow / pytest）
├── build_app.sh            # 打包「桌面毛毯.app」+ DMG + 安装到 /Applications（唯一打包入口）
├── packaging/launcher.c    # app 的 Mach-O 启动器（execv 到包内运行时；见下「打包设计」）
├── run.sh                  # 源码运行（.venv/bin/python rug.py）
├── rug.py                  # 入口：菜单栏应用 + 总控状态机（默认启动即铺上；--start-hidden / --selftest）
├── overlay.py              # 全屏 NSPanel + SCNView 渲染 + 逐像素 hitTest + 点击穿透闸门
├── cloth.py                # ClothSim：numpy 向量化 **真 3D 布料**（3D PBD + 自碰撞 + 屈曲生折痕/翻折 + 隆起 + 抛掷 + 休眠）
├── bumps.py                # BumpField：图标位置/大小 → 静态高度场（"taller stacks, bigger bumps"）
├── icons.py                # IconSensor：osascript 只读查询 + FSEvents/轮询 + 坐标标定
├── menubar.py              # 菜单栏项：模板图标 / 落位判定 / macOS 26「LS 启动拿不到项」的交接
├── contracts/              # 接口契约（interfaces.py 为签名与语义唯一来源）
├── tests/                  # pytest（布料 / 隆起 / 图标 / 渲染层 / 闸门 / 菜单栏）
├── scripts/
│   ├── prep_texture.py     # 把素材图处理成引擎贴图（旋转/缩放/裁边）
│   ├── make_logo_assets.py # 品牌 SVG → icon.icns + 菜单栏模板图标（--check 打 ASCII 预览）
│   ├── probe_status_item.py # 菜单栏项探针（配合 AX 量坐标，排「图标看不见」用）
│   ├── safety_check.py     # 安全自验（快照 + diff，见上）
│   ├── verify_routing.py   # 点击路由机制验证（WindowServer 查询）
│   ├── live_probe.py       # 运行中实例的线上体检（渲染像素/遮挡/路由/图标覆盖）
│   ├── fold_demo.py        # v4 翻折离线验证：驱动「拖一角横过毯身」→ 数值 + 目验 PNG
│   ├── look_demo.py        # v5 观感评测台：rest/drag/fold/corner 四场景离线渲染 + 数值判据（覆盖率/破洞/白边/高度）
│   └── make_back_texture.py # 由正面贴图生成背面贴图（镜像+压暗+模糊，翻折露底用）
├── spike/m0_overlay_spike.py  # M0 层级/性能预验证脚本（留档可复跑）
├── assets/rugs/            # 地毯贴图投放处（rug-01.png 为引擎贴图；<名字>-back.png = 背面）
├── assets/logo/            # 品牌资产：applogo.svg 是唯一来源，其余由 make_logo_assets.py 生成
├── archive/                # 历史产物留档（旧的空壳 app 打包脚本与过时说明，见 archive/README.md）
└── 开发文档.md              # 设计/技术/里程碑/验收 全文档
```

> 代码规模：核心模块 ≈5.7k 行、接口契约 ≈730 行、测试 ≈2.7k 行（144 例）、辅助脚本 ≈2.1k 行。

### 打包设计（为什么不是 py2app / PyInstaller）

- 本项目的 venv 基于 **uv 的 python-build-standalone**（`zlib` 等模块静态内建），**py2app 0.28 会直接崩**（`zlib.__file__` AttributeError，2026-10-09 实测）；`setup.py`（py2app 配置）已归档到 `archive/`。
- 改用的方案零外部打包器依赖：**复制可重定位的 standalone 运行时进包 + `uv pip install --target` 把依赖装进包内 + `packaging/launcher.c` 编译出的 Mach-O 启动器 execv 到包内 `rug.py`**。
- 三个必须踩对的坑（都已写进构建脚本 / 启动器）：① 主可执行文件必须是 **Mach-O**，否则 `codesign` 无法封存 bundle（`codesign -v` 报 “a sealed resource is missing or invalid”→ 别人机器上可能表现为「应用已损坏」且右键救不回来）；② 运行时禁止写 `__pycache__`（`PYTHONDONTWRITEBYTECODE=1` + 构建时预编译），否则运行一次就把签名封存改坏；③ 构建 staging 必须放在 Spotlight 不索引的临时目录，且自检后立刻 `lsregister -u` 注销——否则「启动台/聚焦里莫名多出一个应用」。

### 技术要点（详版见开发文档 §7）

- **层级魔法**：`NSPanel` 放到 `kCGDesktopIconWindowLevel + 8`（本机实测 -2147483595）——在 Finder 桌面图标层之上、任何普通窗口（level 0）之下。普通窗口永远盖住毯子。
- **点击穿透**：macOS 没有逐像素穿透 API。实测结论：view 的 `hitTest` 返回 `None` **不会**把点击交给下层（只吞掉），而 `setIgnoresMouseEvents_(True)` 是**真正有效**的穿透机制。做法：光标距毯子 > 40pt 时把面板设为「忽略鼠标」（点击直达桌面），≤ 40pt 时恢复接收并由 `hitTest` 做 40pt 精确判定；由 30Hz 定时器 + 全局鼠标移动监视器 + 窗口内鼠标事件共同驱动（鼠标类全局监视器无需任何权限）。
- **厚度观感（v6「不像薄纸片」，2026-10-08）**：① **滚边**——沿网格边界环向外+向下卷出 7pt 圆角厚边（顶点色带受光→背光梯度）→ 顶视下毯子四周有真实的材料厚度带，不再是剃刀边；② **接触阴影裙**——沿边界环外扩的渐隐投影（贴边最暗、约 25pt 淡出），**被抬起的布影子更大更淡、偏向光反侧**（`lean=1/tan 仰角`）→ 毯子是真放在桌面上；③ **静止微浮雕**——网格空间静态低频起伏（只扰法线、不动深度）→ 平铺区有柔和明暗（厚织物软塌感），抬起/翻折处自动衰减；④ **折层遮蔽**——被上层布压住的粒子按顶点色压暗 → 翻折两层之间有接触阴影；⑤ **灯光重配**（环境/平行，平铺总亮不变、明暗斜率 +44%）+ 折面不再「全翻正」（折脊两侧有迎光/背光差）。离线验收：`scripts/look_demo.py` 的「厚度」A/B 指标（逐特征关掉重抓表面取差），四场景 `edge_px 22~23`（纸片边 1~2）、`shadow_dalp ~60`、`relief_dlum ~2.6`、折缝 `ao_dlum 4~6`。
- **布料（v5 真 3D，观感对齐参考视频）**：3D 位置 PBD + **自碰撞** + **折痕圆角下限**（(i,i+2) 单侧硬约束：曲率半径 ≥ ~15pt，禁止剪纸式锐折 → 折脊有体积、顶视不开裂）+ 低频屈曲种子（压缩量先按网格低通再转高度 → 长出百 pt 量级的大波浪）+ 图标隆起并入地板高度；**拖动 = 手钉点张力 + 地面摩擦，整块跟手只驱动质心（不做形状记忆）** → 手上的位移只能靠折叠消化，折痕才会出现；准静态自适应收尾 + 空闲休眠（空载 CPU ≈0.5%）。渲染：逐帧网格法线 + **顶/背两张贴图按面朝向分渲染**（翻折露底处显示背面）；**贴图加载期自动裁掉纯白/透明留白**（`_detect_texture_margin`，正/背图同一 rect）——「毯子周围一圈白框」的根因。
- **图标感知**：`osascript` 只读查 Finder 桌面项位置，本地 `os.stat` 补大小 → 高斯隆包叠加；FSEvents 只读监听桌面变化（1.5s 轮询降级），新建/删除文件 2s 内更新隆包。

## 实测性能（本机 2560×1440 / Apple Silicon）

| 项 | 实测（v4 真 3D，2026-10-08 真机） |
| --- | --- |
| 空闲 CPU（毯子入睡后） | ≈ **0.5% 单核**（`step()` 早退；v3 为 1.0~1.25%） |
| 内存 RSS | 启动 ≈75MB；铺上后 ≈**148MB**（SceneKit/Metal 资源常驻，掀开不复还） |
| 入睡耗时 | 铺上落定 ≈**2.7s**（含图标隆起，v3 为 5.7~6.9s）；拖拽后 ≈5.9s |
| 拖动/收尾单帧 | ≈**5.6ms**（100 步实测 557ms；30fps 预算 33ms） |
| v6 厚度几何开销 | ≈**+0.5ms/帧**（滚边 288×4 + 阴影裙 288×4 顶点；`step`+重建实测 5.12ms/帧） |

> 旧的「空闲 CPU≈0% / 内存<100MB」目标仍未完全达到，取舍与可选的后续优化见 [开发文档.md](开发文档.md) §10/§12.16。

## 已知限制与排查

- **v1 只支持主显示器**（`NSScreen.mainScreen()`）；多屏是「每屏一个覆盖窗口实例」的自然扩展，未做。
- 不做「真的把图标扫到毯下」（不移动你的文件布局）——明确排除。
- 分发包是 **ad-hoc 签名 + 未公证**的（见「安装」一节）：自己编自己用没问题，别人要用得右键打开或走系统设置放行；原生 Mac 应用的完整体验（双击即开、无警告）需要 Apple 开发者账号签名 + 公证。
- **要能看见/点到毯子，屏幕上得有可见的桌面区域**：毯子在所有普通窗口之下（这是设计），如果你把窗口铺满了整个屏幕，毯子会被盖住、点不到——把窗口挪开露出桌面即可。
- 终端打印 `图标隆起不可用，进入降级模式`？看括号里的原因对症处理（降级下毯子照常好玩，不会二次弹窗）：
  - **`-1743` / 权限**：系统设置 → 隐私与安全性 → 自动化 → 找到你的终端/启动器 → 勾选「访达」；
  - **AppleScript 语法错误（`-2741` / `-1728`，如 “expected \",\" but found plural class name”）**：说明 macOS 又改了 Finder 词典（2026-10 已在 macOS 26 上踩过一次并修复）。跑 `.venv/bin/python -m pytest tests/test_icons.py -q` 会立刻定位到哪个脚本编不过，把报错贴给开发者即可；
  - **`-1728` 标定失败**：会自动回退 (0, 24)，隆起仍可用。
- 点毯子没反应？先点一下菜单栏的**毛毯图标**（让本应用成为前台）再试；另外确认没有把毯子拖到了别的显示器空间。
- **看不见毯子？** 毯子在所有普通窗口之下是设计（否则会挡住你的 App）——窗口铺满屏幕时就完全看不到；挪开窗口 / F11 显示桌面即可。**看不到隆包？** 隆包只在毯子盖到图标时出现——把毯子拖到图标上。任意时刻可用 `.venv/bin/python scripts/live_probe.py` 体检：毯子画出来没有 / 被谁遮了多少 / 点击会打在谁身上 / 盖住了几个图标。
- **关不掉 / 菜单栏图标找不到了？** 一般用菜单栏毛毯图标 →「退出 ⌘Q」或终端 Ctrl-C 即可（2026-10-08 修复过一次 Ctrl-C 失效问题，见开发文档 §12.12）；兜底：`pkill -f rug.py` 立即彻底退出，桌面即刻复原。
- **双击 app 后「没反应 / 没毯子」？** 先看日志 `~/Library/Logs/DesktopRug.log`（双击启动没有终端，所有 `RUG-*` 行都写在这里；终端运行时直接打在屏幕上）。常见情况：① 已经有实例在跑——重复启动只会打一行「已有实例在运行」然后退出（菜单栏那个毛毯图标就是它）；② 毯子铺上了但被满屏的窗口盖住（设计如此，F11/挪开窗口可见）；③ Finder 自动化权限被拒 → 日志里有 `RUG-WARN`，毯子照常铺、只是图标不鼓包。
- **`/Applications` 里出现两个同名 app / 启动台里多一个？** 说明有旧版本残留。跑一次 `./build_app.sh` 会清理历史空壳并刷新 LaunchServices 注册；仍有多余条目就 `lsregister -dump | grep -B7 com.desktop.rug` 看路径，再 `lsregister -u <多余路径>` 注销。
- **想看到「翻折」**：拖住毯子一个角**横着划过毯身**（行程要过中线），松手后折层会留在毯上（露底处是变暗的背面、折脊是圆角）；来回拖会把折层拖平——这是真实行为，不是 bug；「抚平折痕 ⌘F」一键摊平。观感评测/出图：`.venv/bin/python scripts/look_demo.py`（四场景 → /tmp/look_*.png + 覆盖率/白边/破洞数值）；单折验证：`scripts/fold_demo.py`。
- 观感依赖你的贴图质量；`assets/rugs/README.md` 有规格（正俯视、均匀光照、边缘带透明通道最佳）。

---

## 许可（License）

**MIT License** —— 见 [LICENSE](LICENSE)。可自由使用、修改、分发（含商用），保留版权声明与许可声明即可。

Copyright (c) 2026 nexsjournal

素材与第三方内容的归属另算，与本项目代码的 MIT 授权无关：

| 内容 | 说明 |
| --- | --- |
| `assets/rugs/rug-01.png`、`rug-01-back.png` | 由项目内的演示素材（`blanket.webp`）经 `scripts/prep_texture.py` 处理而来，仅作示例；换成你自己的地毯图即可（规格见 `assets/rugs/README.md`） |
| `assets/logo/**` | 本项目品牌标记，随代码一同按 MIT 授权 |
| `refer/` | 参考视频来自 [X 原帖 @terkelg](https://x.com/terkelg/status/2107540718461886464)，**非本项目版权、不随仓库分发**（已在 `.gitignore` 中忽略） |
| 灵感来源 | [desktop.cleaning](https://desktop.cleaning) 的 "Rugs"；本项目为从零自行设计实现，未使用其代码或素材 |
