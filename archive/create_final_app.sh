#!/bin/bash
# create_final_app.sh - 创建带图标的完整应用

set -euo pipefail
cd "$(dirname "$0")"

APP_NAME="桌面毛毯"
APP_PATH="dist/${APP_NAME}.app"
PROJECT_DIR="$(pwd)"

echo "🚀 创建桌面毛毯应用..."
echo ""

# 创建应用包结构
mkdir -p "${APP_PATH}/Contents/MacOS"
mkdir -p "${APP_PATH}/Contents/Resources"

# 创建 Info.plist
cat > "${APP_PATH}/Contents/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>launcher</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundleIdentifier</key>
    <string>com.desktop.rug</string>
    <key>CFBundleName</key>
    <string>桌面毛毯</string>
    <key>CFBundleDisplayName</key>
    <string>桌面毛毯</string>
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
cat > "${APP_PATH}/Contents/MacOS/launcher" << LAUNCHER
#!/bin/bash
# 定位到项目目录
cd "${PROJECT_DIR}"
# 启动应用
exec .venv/bin/python rug.py
LAUNCHER

chmod +x "${APP_PATH}/Contents/MacOS/launcher"

# 复制图标
if [ -f "icon.icns" ]; then
    cp icon.icns "${APP_PATH}/Contents/Resources/AppIcon.icns"
    echo "✅ 图标已添加"
else
    echo "⚠️  未找到icon.icns，使用默认图标"
fi

# 创建 PkgInfo
echo "APPL????" > "${APP_PATH}/Contents/PkgInfo"

echo ""
echo "✅ 应用创建成功！"
echo ""
echo "📍 应用位置: ${APP_PATH}"
echo ""
