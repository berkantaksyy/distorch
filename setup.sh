#!/usr/bin/env bash
# distorch kurulumu:  bash setup.sh
set -e
cd "$(dirname "$0")"
PY=python3
if $PY -m venv .venv 2>/dev/null; then PY=.venv/bin/python; fi
$PY -m pip install -q -U pip || true
$PY -m pip install -q -r requirements.txt
$PY -c "import cv2,numpy,scipy;print('cv2',cv2.__version__,'numpy',numpy.__version__,'scipy',scipy.__version__)"
$PY -c "import torch;print('torch',torch.__version__,'CPU-only' if not torch.version.cuda else 'CUDA')" \
  || echo "UYARI: torch yok -> bekci kapisi devre disi, sistem yine calisir"
echo "kurulum tamam"
