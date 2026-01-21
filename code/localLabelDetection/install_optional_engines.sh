#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-labels}"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "❌ Venv not found at $VENV_DIR (run run_local_venv.sh first)"
  exit 1
fi

echo "📦 Installing optional engines (best-effort)"

echo "   • Trying detectron2 (pip)"
"$VENV_DIR/bin/pip" install detectron2 >/dev/null 2>&1 || true

if ! "$VENV_DIR/bin/python" -c 'import detectron2' >/dev/null 2>&1; then
  if [[ "${DETECTRON2_INSTALL_FROM_SOURCE:-1}" == "1" ]]; then
    echo "   • detectron2 wheel not available; trying source install from GitHub"
    "$VENV_DIR/bin/pip" install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git' || true
  fi
fi

echo "   • Trying MMDetection (pip)"
"$VENV_DIR/bin/pip" install 'mmcv>=2.0.0rc4,<2.2.0' >/dev/null 2>&1 || true
"$VENV_DIR/bin/pip" install mmdet >/dev/null 2>&1 || true

# On macOS, pip installs of mmcv often lack compiled extensions (mmcv._ext), which breaks inference.
if ! "$VENV_DIR/bin/python" -c 'import mmcv._ext' >/dev/null 2>&1; then
  if [[ "${MMCV_BUILD_FROM_SOURCE:-1}" == "1" ]]; then
    echo "   • mmcv._ext missing; trying to build MMCV from source (MMCV_WITH_OPS=1)"
    MMCV_WITH_OPS=1 "$VENV_DIR/bin/pip" install -U --no-build-isolation 'git+https://github.com/open-mmlab/mmcv.git@v2.1.0' || true
  fi
fi

if ! "$VENV_DIR/bin/python" -c 'import mmdet' >/dev/null 2>&1; then
  if [[ "${MMDET_TRY_OPENMIM:-0}" == "1" ]]; then
    echo "   • mmdet not installed; trying openmim + mmcv"
    "$VENV_DIR/bin/pip" install -U openmim || true
    "$VENV_DIR/bin/python" -m mim install mmcv || true
    "$VENV_DIR/bin/pip" install mmdet || true
  fi
fi

if "$VENV_DIR/bin/python" -c 'import mmdet, mmcv._ext' >/dev/null 2>&1; then
  echo "   • mmdet + mmcv._ext: OK"
else
  echo "   • mmdet/mmcv ops still not usable (mmcv._ext missing)."
fi

echo "✅ Optional engine install attempt finished"
