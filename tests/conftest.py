"""tests/conftest.py — 把项目根目录加入 sys.path，保证 cloth/bumps/icons/contracts 可导入。

（pytest prepend 导入模式只会把无 __init__.py 的 tests/ 目录加入 sys.path，
不会自动加项目根；此处显式补充。仅改 sys.path，无其他副作用。）
"""
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
