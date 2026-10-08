#!/bin/bash
# run.sh — 桌面毛毯启动脚本（integrator 专有）。
# 用法：./run.sh [--selftest]
set -euo pipefail
cd "$(dirname "$0")"
exec .venv/bin/python rug.py "$@"
