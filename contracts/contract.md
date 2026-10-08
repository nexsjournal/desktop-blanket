# 契约 v1（architect 冻结 · 2026-10-07）

签名与语义唯一来源：`contracts/interfaces.py`（docstring 已写单位/错误行为）。
改契约必须回报 architect，不许自行发明字段。桥接姿势看 `spike/m0_overlay_spike.py`。

## 0. 全局约定
- **坐标**：契约层数据一律「主屏本地屏幕点 pt」：主屏左上为 (0,0)、x 右、y 下、z 上（凸出屏幕）；1 场景单位=1pt。`IconInfo` 例外：Finder 桌面视图坐标（桌面左上原点、y 向下、未标定），`screen = finder + (dx,dy)`（来自 `calibration()`）。AppKit(bottom-left) 与 SceneKit 场景轴的换算只在 overlay/rug 调用点做。
- **线程**：主线程单线程；AppKit 事件 + NSTimer 驱动 `sim.step()`；所有回调主线程；禁止加线程/锁。
- **安全红线（§4）**：~/Desktop 只读；不碰壁纸 API / desktoppicture.db / `defaults write` / killall Finder；零网络；退出无残留。写进实现，不靠自觉。

## 1. 所有权表（一个路径一个写者）
| 路径 | 写者 | 说明 |
| --- | --- | --- |
| `cloth.py`、`bumps.py`、`tests/test_cloth.py`、`tests/test_bumps.py` | A（布料/纯逻辑） | ClothSim 全量 + BumpField |
| `icons.py`、`tests/test_icons.py` | B（传感） | IconSensor：osascript 只读 + FSEvents/轮询 + 标定 |
| `overlay.py` | C（渲染） | RugOverlay：NSPanel+SCNView+hitTest+伪阴影+光照 |
| `rug.py` | D（主控） | RugApp+parse_args/main：菜单/状态机/selftest/降级 |
| `contracts/**`、`requirements.txt`、`run.sh`、`scripts/`、`开发文档.md`、`assets/` | **integrator/主 Agent 专有** | 其他人只读；改需求（如加 `pyobjc-framework-FSEvents`）报 integrator，不许直接改 |

## 2. 跨模块数据流
`IconSensor.query()` → `list[IconInfo]`（未标定）→ `calibration()` 得 (dx,dy) → `BumpField(icons, w, h, calibration=(dx,dy))` → `overlay.sim.set_bumps(field)` + `wake()`。
RugApp 持有 IconSensor/RugOverlay；overlay 经 `sim_factory` 新建 sim（闭包注入当前 BumpField）；`on_grab/on_release` 回调驱动状态机。

## 3. 模块验收命令（integrator 收口前各自先过）
- A：`.venv/bin/python -m pytest tests/test_cloth.py tests/test_bumps.py -q` 全绿。必测：vertices 形状 (5040,3) float32、indices 复用同一 buffer、grab→drag_to 跟随、release 后 kinetic_energy 收敛→is_asleep、图标格 z>0、nearest_distance 阈值布尔正确、throw_in 非法 corner 抛 ValueError。
- B：`.venv/bin/python -m pytest tests/test_icons.py -q` 全绿（mock osascript 子进程，含 -1743 权限拒绝→IconSensorError、stat 缺文件→size 0、轮询降级触发 on_change）；真机手测 `IconSensor().query()` 非空 + `calibration()` 返回有限值（需授「自动化→Finder」）。
- C：`.venv/bin/python -c "import overlay"` 过；真机 10s 交互：面板 level=图标层+8（CGWindowList 探针）、毯上可抓拖、毯外双击桌面图标生效（穿透）。
- D：`.venv/bin/python rug.py --selftest` → 2s 打印 `SELFTEST-LEVEL <int>`、5s `SELFTEST-OK`、exit 0；手测菜单铺/掀/款式切换/退出后进程消失、桌面无残留。

## 4. PyObjC 桥接备忘（spike 实录 · C/D 必读）
- 类工厂缺失（如 `SCNGeometryElement.geometryElement…`）→ 一律 `alloc().initWithData:…` 系构造。
- `SCNGeometrySource.alloc().initWithData_semantic_colorSpace_vectorCount_floatComponents_componentsPerVector_bytesPerComponent_dataOffset_dataStride_`；numpy→NSData：`NSData.dataWithBytes_length_(a.tobytes(), a.nbytes)`，数组必须 C-contiguous float32。
- `SCNMaterial.diffuse()` 是属性容器：`mat.diffuse().setContents_(NSColor…)`；透明三件套：panel/view `setOpaque_(False)` + `setBackgroundColor_(NSColor.clearColor())` + `setHasShadow_(False)`。
- NSTimer：selector 必须写完整 ObjC 形式 `"tick:"`（不是 `"tick_"`）；`NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_`。
- 层级：`int(CGWindowLevelForKey(kCGDesktopIconWindowLevel)) + 8`，动态读取；Quartz import 失败回退 `-2147483603 + 8`；`setCollectionBehavior_(CanJoinAllSpaces|Stationary|IgnoresCycle)`；`setIgnoresMouseEvents_(False)`；`orderFrontRegardless()`。
- 退出：NSTimer 探针计数 + `os._exit(0)` 兜底（`app.run()` 不可靠返回）。
- **渲染映射**：sim 顶点(top-left,y-down,z-up) → 场景 x→x、sim 高度 z→y(朝相机)、sim y→z；正交相机俯视、1 单位=1pt、视野=屏宽高。uv 的 v 轴在 SceneKit 朝上：喂贴图前 1-v。索引环绕同 spike（a,a+1,a+cols+1, a,a+cols+1,a+cols），材质建议双面。

## 5. 错误与日志约定
- stdout 单行日志：`RUG-INFO|WARN|ERROR <msg>`；禁止逐帧打印；模块内异常不得静默吞掉——可降级优先于崩溃（WARN 继续跑，ERROR 报因后继续或干净退出）。
- icons：AppleEvent -1743 / 超时 / 解析失败 → `IconSensorError`；RugApp 捕获 → degraded 模式（无隆起、物理照常、菜单标注），**不二次弹权限**；FSEvents 绑定缺失 → 1~2s 轮询（主线程 NSTimer），docstring 已标注为允许降级。
- selftest 探针行固定：`SELFTEST-LEVEL <int>` / `SELFTEST-OK`（成功，exit 0）；失败 `SELFTEST-LEVEL -1` / `SELFTEST-FAIL`（exit 1）。无 RUG- 前缀，供脚本 grep。
