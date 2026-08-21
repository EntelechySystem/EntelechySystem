#!/usr/bin/env bash
# =============================================================================
# EntelechySystem —— uv 环境迁移与清理脚本（幂等）
#
# 功能：检测漂移 → 备份旧环境 → 彻底清理 → 用 uv (Python 3.14) 重建 → 验证
# 用法：
#   bash scripts/uv_migrate.sh            # 常规执行（清华镜像回退可选）
#   bash scripts/uv_migrate.sh --mirror   # 强制使用清华 PyPI 镜像
#
# 说明：
#   - 幂等：可重复执行；每次执行前都会先备份当前 frozen 快照。
#   - 本地依赖（RECS / scene-kit）已通过 pyproject.toml 的
#     [tool.uv.sources] 以 editable 方式声明，uv sync 会自动处理。
#   - 默认不安装 torch（可选依赖），需要时：uv sync --extra torch
# =============================================================================
set -euo pipefail

# ── 常量 ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_VERSION="3.14"
MIRROR_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
BACKUP_FILE="${PROJECT_ROOT}/old_requirements.txt"
USE_MIRROR=false
[[ "${1:-}" == "--mirror" ]] && USE_MIRROR=true

cd "${PROJECT_ROOT}"

echo "================================================================"
echo "  EntelechySystem uv 环境迁移"
echo "  项目根: ${PROJECT_ROOT}"
echo "  Python: ${PY_VERSION}  镜像: ${USE_MIRROR}"
echo "================================================================"

# ── 0. 前置检查 ──────────────────────────────────────────────────────────────
if ! command -v uv >/dev/null 2>&1; then
    echo "[✗] 未找到 uv，请先安装：curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
uv --version

# ── 1. 检测漂移 ──────────────────────────────────────────────────────────────
echo ""
echo "[1/5] 检测当前环境..."
echo "  pyproject.toml: $([[ -f pyproject.toml ]] && echo '✓' || echo '✗')"
echo "  uv.lock:        $([[ -f uv.lock ]] && echo '✓' || echo '✗')"
echo "  .venv:          $([[ -d .venv ]] && echo '✓' || echo '✗')"

# ── 2. 备份旧环境快照 ────────────────────────────────────────────────────────
echo ""
echo "[2/5] 备份旧环境 frozen 快照..."
if [[ -d .venv ]]; then
    if uv pip freeze >"${BACKUP_FILE}" 2>/dev/null && [[ -s "${BACKUP_FILE}" ]]; then
        echo "  ✓ 已备份 $(wc -l <"${BACKUP_FILE}" | tr -d ' ') 个包到 ${BACKUP_FILE}"
    else
        echo "  ⚠  freeze 失败，尝试跳过备份（不影响重建）"
        rm -f "${BACKUP_FILE}"
    fi
else
    echo "  - 无 .venv，跳过备份"
fi

# ── 3. 彻底清理 ──────────────────────────────────────────────────────────────
echo ""
echo "[3/5] 彻底清理旧环境..."
rm -rf .venv
rm -f uv.lock
echo "  ✓ 已删除 .venv 与 uv.lock"

# ── 4. 重建环境 ──────────────────────────────────────────────────────────────
echo ""
echo "[4/5] 用 uv (Python ${PY_VERSION}) 重建..."
uv python pin "${PY_VERSION}"
echo "  ✓ uv python pin ${PY_VERSION}"

# 需要镜像时设置环境变量（仅本次执行生效）
if ${USE_MIRROR}; then
    export UV_DEFAULT_INDEX="${MIRROR_URL}"
    echo "  ✓ 使用镜像: ${MIRROR_URL}"
fi

# uv sync：自动解析 pyproject.toml 的 dependencies 与 [tool.uv.sources] 本地依赖
if uv sync; then
    echo "  ✓ uv sync 完成"
else
    echo "  ⚠ uv sync 失败（可能是网络问题）"
    if ! ${USE_MIRROR}; then
        echo "  → 尝试使用清华镜像重试..."
        export UV_DEFAULT_INDEX="${MIRROR_URL}"
        uv sync
        echo "  ✓ 镜像 uv sync 完成"
    else
        echo "  ✗ 镜像方式也失败，请检查网络后重试"
        exit 1
    fi
fi

# ── 5. 验证 ──────────────────────────────────────────────────────────────────
echo ""
echo "[5/5] 验证环境..."
echo ""
echo "── 依赖树（前 2 层）──"
uv tree --depth 2 || true

echo ""
echo "── 导入验证 ──"
check_import() {
    local pkg="$1"
    if uv run python -c "import ${pkg}" 2>/dev/null; then
        echo "  ✓ import ${pkg}"
    else
        echo "  ✗ import ${pkg} 失败"
    fi
}
check_import chardet
check_import numpy
check_import openpyxl
check_import pandas
check_import recs
check_import scene_kit

echo ""
echo "── ES 引擎导入验证 ──"
if uv run python -c "
import sys
sys.path.insert(0, '.')
from EntelechySystem.engine.tools.Tools import Tools
from EntelechySystem.engine.core.define_engineGlobalVariables import gb
print('  ✓ ES 引擎工具与全局变量导入成功，version =', gb['engine_version'])
"; then
    :
else
    echo "  ⚠ ES 引擎导入失败（可能是运行时路径问题，见 docs/开发者指南.md）"
fi

echo ""
echo "================================================================"
echo "  迁移完成！"
echo "  环境: ${PROJECT_ROOT}/.venv (Python ${PY_VERSION})"
echo "  备份: ${BACKUP_FILE}"
echo "  常用命令:"
echo "    uv run python EntelechySystem/Experiments/EXP_test_010.py   # 运行示例实验"
echo "    uv sync --extra torch                                       # 可选：启用 PyTorch"
echo "================================================================"
