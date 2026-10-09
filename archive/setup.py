"""
setup.py - 桌面毛毯 (Desktop Rug) 打包配置
使用 py2app 将 Python 项目打包成 macOS 应用
"""
from setuptools import setup

APP = ['rug.py']
DATA_FILES = [
    ('assets/rugs', [
        'assets/rugs/rug-01.png',
        'assets/rugs/rug-01-back.png',
    ]),
]

OPTIONS = {
    'argv_emulation': False,
    'iconfile': None,
    'plist': {
        'CFBundleName': 'Desktop Rug',
        'CFBundleDisplayName': 'Desktop Rug',
        'CFBundleIdentifier': 'com.desktop.rug',
        'CFBundleVersion': '6.1.0',
        'CFBundleShortVersionString': '6.1',
        'LSUIElement': True,
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '11.0',
        'NSAppleEventsUsageDescription': 'Read desktop icon positions',
        'NSDesktopFolderUsageDescription': 'Read desktop files for bump display',
    },
    'packages': ['numpy', 'AppKit', 'Foundation', 'Quartz', 'SceneKit'],
    'includes': ['cloth', 'overlay', 'bumps', 'icons', 'settings_ui', 'contracts.interfaces'],
    'excludes': ['matplotlib', 'scipy', 'pandas', 'PIL'],
    'resources': ['assets/rugs/'],
    'optimize': 0,
}

setup(
    name='DesktopRug',
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
