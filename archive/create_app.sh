#!/bin/bash
# create_automator_app.sh - 创建 Automator 应用包装器

set -euo pipefail
cd "$(dirname "$0")"

APP_NAME="Desktop Rug"
APP_PATH="dist/${APP_NAME}.app"
SCRIPT_PATH="$(pwd)/run.sh"

echo "🚀 创建 macOS 应用包装器..."
echo ""

# 创建 dist 目录
mkdir -p dist

# 创建应用包结构
echo "📦 创建应用包结构..."
mkdir -p "${APP_PATH}/Contents/MacOS"
mkdir -p "${APP_PATH}/Contents/Resources"

# 创建 Info.plist
echo "📝 创建 Info.plist..."
cat > "${APP_PATH}/Contents/Info.plist" << 'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>Desktop Rug</string>
    <key>CFBundleIdentifier</key>
    <string>com.desktop.rug</string>
    <key>CFBundleName</key>
    <string>Desktop Rug</string>
    <key>CFBundleDisplayName</key>
    <string>Desktop Rug</string>
    <key>CFBundleVersion</key>
    <string>6.1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>6.1</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>LSUIElement</key>
    <true/>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>LSMinimumSystemVersion</key>
    <string>11.0</string>
    <key>NSAppleEventsUsageDescription</key>
    <string>需要访问Finder以读取桌面图标位置</string>
</dict>
</plist>
PLIST

# 创建启动脚本
echo "🔨 创建启动脚本..."
cat > "${APP_PATH}/Contents/MacOS/${APP_NAME}" << 'LAUNCHER'
#!/bin/bash
# 获取应用所在目录（不是 Contents/MacOS）
APP_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$APP_DIR"
exec ./run.sh
LAUNCHER

chmod +x "${APP_PATH}/Contents/MacOS/${APP_NAME}"

# 创建 PkgInfo
echo "APPL????" > "${APP_PATH}/Contents/PkgInfo"

echo ""
echo "✅ 创建成功！"
echo ""
echo "📍 应用位置: $(pwd)/${APP_PATH}"
echo ""
echo "🎯 下一步："
echo "   1. 移动到应用程序文件夹:"
echo "      mv \"${APP_PATH}\" /Applications/"
echo ""
echo "   2. 或者直接运行:"
echo "      open \"${APP_PATH}\""
echo ""
echo "⚠️  首次运行:"
echo "   - 右键点击应用 → 打开（绕过未签名警告）"
echo "   - 授予自动化权限（读取Finder图标位置）"
echo ""
