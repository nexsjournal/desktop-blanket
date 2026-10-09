#!/bin/bash
# build_app.sh — 打包「桌面毛毯.app」（自包含：内置 Python 运行时 + 依赖 + 贴图）。
#
# 用法：
#   ./build_app.sh              构建 → 自检 → 生成 DMG → 安装到 /Applications → 启动
#   ./build_app.sh --keep-stage 保留临时组装目录（排障用）
#
# 产物：
#   /Applications/桌面毛毯.app        双击即用（内置运行时，与项目目录无关）
#   dist/桌面毛毯-<版本>.dmg          分发给别人：拖进 Applications 即装
#
# 自检门禁（构建必须通过）：
#   1. 包内运行时能 import numpy/objc/AppKit/Quartz/SceneKit；
#   2. 用包内运行时跑 rug.py --selftest，必须打印 SELFTEST-OK
#      （证明「点开 app 后屏幕上真的会出现毯子窗口」）。
#
# 设计说明（为什么不用 py2app）：本项目 venv 基于 uv 的 python-build-standalone，
# zlib 等模块是静态内建，py2app 0.28 会直接崩（zlib.__file__ AttributeError，见
# 2026-10-09 排障记录）。standalone 运行时本身可重定位（复制后 sys.prefix 跟着走），
# 因此这里直接「复制运行时 + 装依赖到包内 + 脚本启动器」，零外部打包器依赖。
set -euo pipefail
cd "$(dirname "$0")"

VERSION="6.9.0"
APP_NAME="桌面毛毯"
BUNDLE_ID="com.desktop.rug"
APP="$APP_NAME.app"
# 分发包文件名：两处约束决定了这个名字
#   ① 必须 ASCII —— 中文名在 GitHub Release / HTTP 下载链路上会被掐掉非 ASCII 字符
#      （实测上传后变成 "-6.9.0.dmg"）；
#   ② 用「稳定名」（不带版本号）—— README 里的 GitHub 直链形如
#      /releases/latest/download/DesktopRug-arm64.dmg，换版本时链接不用改；
#      版本号由 Release 的 tag 承载。
DIST_ARCH="$(uname -m)"
DIST_DMG="dist/DesktopRug-${DIST_ARCH}.dmg"
LOG_FILE="$HOME/Library/Logs/DesktopRug.log"
LOCK_FILE="$HOME/Library/Application Support/DesktopRug/instance.lock"
LSREGISTER="/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister"
LEGACY_APPS=(
  "/Applications/${APP_NAME}.app"
  "/Applications/Desktop Rug.app"
  "$PWD/dist/${APP_NAME}.app"
  "$PWD/dist/Desktop Rug.app"
  "$PWD/dist/rug.app"
)

KEEP_STAGE=0
[[ "${1:-}" == "--keep-stage" ]] && KEEP_STAGE=1

step() { printf '\n\033[1m▸ %s\033[0m\n' "$*"; }
ok()   { printf '\033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '\033[33m!\033[0m %s\n' "$*"; }
die()  { printf '\033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

# ─────────────────────────────────────────────── 0. 预检
step "0/7 预检"
[[ "$(uname -s)" == "Darwin" ]] || die "只能在 macOS 上打包"
[[ -x .venv/bin/python ]] || die "缺少 .venv/bin/python（先：uv venv .venv && uv pip install -r requirements.txt）"
for f in assets/rugs/rug-01.png assets/rugs/rug-01-back.png rug.py settings_ui.py \
         menubar.py assets/logo/applogo.svg scripts/make_logo_assets.py; do
  [[ -e "$f" ]] || die "缺少 $f"
done
command -v uv >/dev/null || die "需要 uv 把依赖装进包内运行时（brew install uv）"

# 品牌图标资产：icon.icns 与菜单栏模板图标都是**生成物**（不进仓库），缺文件或源 SVG 更新就重建
if [[ ! -e icon.icns || assets/logo/applogo.svg -nt icon.icns \
      || ! -e assets/logo/menubar.png || ! -e assets/logo/menubar@2x.png \
      || ! -e assets/logo/menubar@3x.png ]]; then
  .venv/bin/python scripts/make_logo_assets.py >/dev/null || die "图标资产生成失败"
  ok "图标资产已重建（源 SVG → icon.icns + assets/logo/menubar*.png）"
else
  ok "图标资产比源 SVG 新，跳过重建"
fi

BASE_PY="$(.venv/bin/python -c 'import sys; print(sys.base_prefix)')"
[[ -x "$BASE_PY/bin/python3.11" ]] || die "找不到基座 Python：$BASE_PY"
HOST_ARCH="$(uname -m)"
ok "基座运行时：${BASE_PY}（${HOST_ARCH}）"

# ─────────────────────────────────────────────── 1. 组装骨架
step "1/7 组装 App 包骨架（临时目录，避开 Spotlight 索引）"
STAGE="$(mktemp -d "${TMPDIR:-/tmp}/rug-build.XXXXXX")"
cleanup() {
  if [[ "$KEEP_STAGE" == "1" ]]; then
    warn "保留组装目录：$STAGE"
  else
    rm -rf "$STAGE"
  fi
}
trap cleanup EXIT
APP_DIR="$STAGE/$APP"
mkdir -p "$APP_DIR/Contents/MacOS" "$APP_DIR/Contents/Resources/app/assets" \
         "$APP_DIR/Contents/Resources/runtime"
ok "组装目录：$STAGE"

# ─────────────────────────────────────────────── 2. 内置 Python 运行时
step "2/7 复制 Python 运行时（约 60MB，剔掉测试/IDLE/Tk 等无关模块）"
rsync -a \
  --exclude 'lib/python3.11/test/' \
  --exclude 'lib/python3.11/idlelib/' \
  --exclude 'lib/python3.11/tkinter/' \
  --exclude 'lib/python3.11/ensurepip/' \
  --exclude 'lib/python3.11/lib2to3/' \
  --exclude 'lib/python3.11/site-packages/' \
  --exclude 'lib/python3.11/config-*/' \
  --exclude 'include/' \
  --exclude 'share/' \
  --exclude 'bin/*-config' \
  --exclude 'bin/2to3*' \
  --exclude 'bin/idle3*' \
  --exclude 'bin/pip*' \
  --exclude 'bin/wheel*' \
  "$BASE_PY/" "$APP_DIR/Contents/Resources/runtime/"

# ─────────────────────────────────────────────── 3. 依赖装进包内运行时
step "3/7 安装依赖到包内运行时"
RUNTIME_PY="$APP_DIR/Contents/Resources/runtime/bin/python3.11"
mkdir -p "$APP_DIR/Contents/Resources/runtime/lib/python3.11/site-packages"
uv pip install --quiet \
  --python "$RUNTIME_PY" \
  --target "$APP_DIR/Contents/Resources/runtime/lib/python3.11/site-packages" \
  numpy pyobjc-framework-Cocoa pyobjc-framework-Quartz \
  pyobjc-framework-SceneKit pyobjc-framework-FSEvents || die "依赖安装失败（检查网络/uv）"
"$RUNTIME_PY" -m compileall -q "$APP_DIR/Contents/Resources/runtime/lib/python3.11" >/dev/null 2>&1 || true
"$RUNTIME_PY" - <<'PY' || die "包内运行时依赖自检失败"
import sys
import numpy, objc, AppKit, Quartz, SceneKit  # noqa: F401
from Foundation import NSObject  # noqa: F401
print(f"  包内运行时就绪：Python {sys.version.split()[0]} / numpy {numpy.__version__} / pyobjc {objc.__version__}")
PY

# ─────────────────────────────────────────────── 4. 应用源码 + 素材
step "4/7 复制应用源码与贴图"
rsync -a rug.py cloth.py overlay.py bumps.py icons.py menubar.py settings_ui.py \
      "$APP_DIR/Contents/Resources/app/"
rsync -a --exclude '__pycache__' contracts "$APP_DIR/Contents/Resources/app/"
rsync -a --exclude '*.webp' --exclude '.DS_Store' assets/rugs/ \
      "$APP_DIR/Contents/Resources/app/assets/rugs/"
# 品牌图标：菜单栏模板图标（menubar*.png）运行时要用；SVG 一并带上便于溯源
rsync -a --exclude '.DS_Store' assets/logo/ \
      "$APP_DIR/Contents/Resources/app/assets/logo/"
# 出厂设置（首次运行播种到 ~/Library/Application Support/DesktopRug/settings.json）
cp settings.json "$APP_DIR/Contents/Resources/app/settings.json"
cp icon.icns "$APP_DIR/Contents/Resources/AppIcon.icns"
printf 'APPL????' > "$APP_DIR/Contents/PkgInfo"
# 预编译 .pyc：运行时不许再写（会破坏签名封存），所以构建时编好随包签名
"$APP_DIR/Contents/Resources/runtime/bin/python3.11" -m compileall -q \
  "$APP_DIR/Contents/Resources/app" >/dev/null 2>&1 || true
ok "源码 + 贴图 + 图标已就位（含预编译 .pyc）"

# ─────────────────────────────────────────────── 5. Info.plist + 启动器
step "5/7 写 Info.plist 与启动器"
cat > "$APP_DIR/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>launcher</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundleIdentifier</key>
    <string>${BUNDLE_ID}</string>
    <key>CFBundleName</key>
    <string>${APP_NAME}</string>
    <key>CFBundleDisplayName</key>
    <string>${APP_NAME}</string>
    <key>CFBundleVersion</key>
    <string>${VERSION}</string>
    <key>CFBundleShortVersionString</key>
    <string>${VERSION}</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>LSMinimumSystemVersion</key>
    <string>11.0</string>
    <!-- 常驻状态栏应用（Clash 那种）：平时不占 Dock。设置窗打开时由 rug.py 在运行时
         临时切到 Regular（Dock 图标出现），关窗后切回 Accessory（Dock 图标消失）。 -->
    <key>LSUIElement</key>
    <true/>
    <key>LSMultipleInstancesProhibited</key>
    <true/>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSAppleEventsUsageDescription</key>
    <string>读取桌面图标位置，让毯子盖住图标并在图标处隆起（只读，不修改任何文件）。</string>
    <key>NSDesktopFolderUsageDescription</key>
    <string>读取桌面文件大小用于隆起造型（只读）。</string>
</dict>
</plist>
PLIST

cat > "$APP_DIR/Contents/MacOS/_launcher_fallback.sh" <<'LAUNCHER'
#!/bin/bash
# 桌面毛毯启动器（无 clang 时的回退版）：只用包内自带的 Python 运行时。
set -u
CONTENTS="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"
export PYTHONDONTWRITEBYTECODE=1   # 不许往 app 包里写 .pyc（会破坏签名封存）
exec "$CONTENTS/Resources/runtime/bin/python3.11" "$CONTENTS/Resources/app/rug.py" "$@"
LAUNCHER
chmod +x "$APP_DIR/Contents/MacOS/_launcher_fallback.sh"

# 主可执行文件用 Mach-O（packaging/launcher.c）：脚本型主可执行文件会让 codesign
# 无法封存 bundle（"a sealed resource is missing or invalid" → 别人机器上可能报
# 「应用已损坏」且右键救不回来）。没装 clang 时回退脚本启动器并跳过 bundle 签名。
LAUNCHER_MACHO=0
if command -v clang >/dev/null 2>&1; then
  clang -O2 -o "$APP_DIR/Contents/MacOS/launcher" packaging/launcher.c \
    || die "launcher.c 编译失败"
  LAUNCHER_MACHO=1
  rm -f "$APP_DIR/Contents/MacOS/_launcher_fallback.sh"
  ok "主可执行文件：Mach-O（packaging/launcher.c 编译）"
else
  mv "$APP_DIR/Contents/MacOS/_launcher_fallback.sh" "$APP_DIR/Contents/MacOS/launcher"
  warn "没有 clang：主可执行文件退回 shell 脚本（bundle 签名将跳过；分发时别人可能需要"
  warn "  右键打开或在 系统设置→隐私与安全性 里放行——装 Xcode 命令行工具即可用 Mach-O 版）"
fi
plutil -lint "$APP_DIR/Contents/Info.plist" >/dev/null || die "Info.plist 格式错误"
ok "Info.plist（${APP_NAME} ${VERSION} / ${BUNDLE_ID} / LSUIElement）"

# ─────────────────────────────────────────────── 6. 签名 + 自检门禁
step "6/7 ad-hoc 签名 + 自检门禁"
# arm64 上内核只执行「有签名」的二进制：先把包内所有 Mach-O 逐个 ad-hoc 签，
# 再签整个 bundle（--force 覆盖 wheel/运行时自带的浅签名）。
SIGNED=0
while IFS= read -r -d '' f; do
  if file -b "$f" | grep -q "Mach-O"; then
    if codesign --force --sign - "$f" >/dev/null 2>&1; then
      SIGNED=$((SIGNED + 1))
    fi
  fi
done < <(find "$APP_DIR" -type f -print0)
if [[ "$LAUNCHER_MACHO" == "1" ]]; then
  if codesign --force --sign - "$APP_DIR" >/dev/null 2>&1 \
     && codesign --verify --strict "$APP_DIR" >/dev/null 2>&1; then
    ok "ad-hoc 签名完成并通过校验（包内 ${SIGNED} 个二进制 + bundle）"
  else
    warn "bundle 签名校验未通过（包内 ${SIGNED} 个二进制已签名）——本机运行不受影响"
  fi
else
  warn "脚本型主可执行文件：跳过 bundle 签名（包内 ${SIGNED} 个二进制已签名）"
fi

# RUG_NO_FILE_LOG=1：构建脚本的 stdin/stdout 都不是终端，不关日志的话 selftest 的
# 输出会被「双击启动落盘」逻辑写进 ~/Library/Logs/DesktopRug.log，管道里读不到。
SELFTEST_OUT="$(RUG_NO_FILE_LOG=1 "$APP_DIR/Contents/MacOS/launcher" --selftest 2>&1 || true)"
printf '%s\n' "$SELFTEST_OUT" | grep -E "SELFTEST-(LEVEL|OK|FAIL)" | sed 's/^/   /' || true
if ! printf '%s\n' "$SELFTEST_OUT" | grep -q '^SELFTEST-OK$'; then
  echo "   ── selftest 完整输出（末 25 行）──"
  printf '%s\n' "$SELFTEST_OUT" | tail -25 | sed 's/^/   /'
  die "自检未通过：覆盖窗口没有按预期出现（上面是完整输出）"
fi
ok "自检通过：覆盖窗口 level 正确、毯子上屏（SELFTEST-OK）"
# 自检跑过的 staging 副本会被 LaunchServices 登记（幽灵条目，出现在 app 列表里就是
# 「莫名多出一个应用」的来源之一）——用完立刻注销，只留 /Applications 里那一个。
"$LSREGISTER" -u "$APP_DIR" >/dev/null 2>&1 || true

# ─────────────────────────────────────────────── 7. DMG + 安装
step "7/7 生成 DMG 并安装到 /Applications"
mkdir -p dist build
# 清掉历史遗留的「空壳 app」（create_app.sh / create_final_app.sh 产物：写死项目绝对
# 路径的 4 文件壳，发给别人打不开，还会在启动台里重复出现）
for legacy in "${LEGACY_APPS[@]}"; do
  if [[ -e "$legacy" ]]; then
    rm -rf "$legacy"
    warn "已清理旧壳：$legacy"
  fi
done
mkdir -p "$STAGE/dmg"
ditto "$APP_DIR" "$STAGE/dmg/$APP"
ln -s /Applications "$STAGE/dmg/Applications"
rm -f dist/DesktopRug-*.dmg dist/"${APP_NAME}"-*.dmg   # 旧版本分发包不再保留
hdiutil create -quiet -volname "$APP_NAME" -srcfolder "$STAGE/dmg" -ov -format UDZO "$DIST_DMG"
ok "分发包：$DIST_DMG"

# 旧实例先退出（否则新代码要等下次启动生效）
if [[ -f "$LOCK_FILE" ]]; then
  OLD_PID="$(cat "$LOCK_FILE" 2>/dev/null || true)"
  if [[ -n "${OLD_PID:-}" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    kill "$OLD_PID" 2>/dev/null || true
    sleep 1
    warn "已退出旧实例（pid ${OLD_PID}）"
  fi
fi
# 早于单实例锁的历史实例（旧壳 app / .venv 启动的 rug.py）也一并退出。
# 注意用「python… rug.py」这种含解释器的模式：旧壳启动的进程命令行是相对路径
# （.venv/bin/python rug.py），只按绝对路径匹配会漏掉（2026-10-09 实测踩坑）。
if pkill -f "python.*rug\.py" 2>/dev/null; then
  warn "已退出旧 rug 进程"
fi
sleep 1
if [[ "$(pgrep -f 'rug\.py' | wc -l | tr -d ' ')" -gt 0 ]]; then
  warn "仍有 rug 进程在跑：$(pgrep -f 'rug\.py' | tr '\n' ' ')"
fi
rm -rf "/Applications/$APP"
ditto "$APP_DIR" "/Applications/$APP"
"$LSREGISTER" -f "/Applications/$APP" >/dev/null 2>&1 || true
ok "已安装：/Applications/$APP"

echo
echo "  启动应用（屏幕上应立刻出现毯子，菜单栏出现毛毯图标）…"
open "/Applications/$APP"
sleep 4
if pgrep -f "/Applications/$APP/Contents/Resources/app/rug.py" >/dev/null; then
  ok "应用已启动并在运行"
else
  warn "进程没起来——日志尾部："
  tail -20 "$LOG_FILE" 2>/dev/null || echo "   （日志为空）"
fi

cat <<EOF

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 构建完成
   应用    /Applications/$APP          （双击即用，可加进登录项开机自启）
   分发包  $DIST_DMG     （发给别人：打开后把 app 拖进 Applications）
   日志    $LOG_FILE
   设置    ~/Library/Application Support/DesktopRug/settings.json

 退出应用：菜单栏毛毯图标 → 退出（或 pkill -f 'Resources/app/rug.py'）

 别人首次打开（未签名+公证的通用情况）：
   系统会提示「无法验证开发者」→ 右键点应用 →「打开」→ 再点「打开」即可；
   macOS 15+ 需到 系统设置 → 隐私与安全性 → 找到拦截提示点「仍要打开」。
   要做到双击即开不弹窗，需要 Apple 开发者账号（\$99/年）做 Developer ID 签名 + 公证。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EOF
