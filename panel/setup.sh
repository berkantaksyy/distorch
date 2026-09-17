#!/usr/bin/env bash
# Test paneli kurulumu. Linux. macOS'a ozgu hicbir sey yok.
#   ./setup.sh          kur
#   ./setup.sh --cpu    torch'u acikca CPU tekerlekleriyle kur (varsayilan da bu)
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
VENV=.venv

echo "== python"
$PY -c 'import sys; print(sys.version)'

# Tkinter stdlib ama Debian/Ubuntu ayri paket olarak veriyor
if ! $PY -c 'import tkinter' 2>/dev/null; then
  echo
  echo "HATA: tkinter yok. Once sunu kur:"
  echo "  Debian/Ubuntu : sudo apt install -y python3-tk"
  echo "  Fedora/RHEL   : sudo dnf install -y python3-tkinter"
  echo "  Arch          : sudo pacman -S tk"
  exit 1
fi

echo "== sanal ortam: $VENV"
[ -d "$VENV" ] || $PY -m venv --system-site-packages "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --quiet --upgrade pip wheel

echo "== paketler"
pip install --quiet -r requirements.txt

echo "== kontrol"
python - <<'EOF'
import importlib, sys
for m in ("cv2", "numpy", "PIL", "tkinter"):
    importlib.import_module(m); print(f"  {m:10s} ok")
for m in ("torch", "ultralytics"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:10s} ok  {getattr(mod,'__version__','')}")
    except Exception as e:
        print(f"  {m:10s} YOK ({e.__class__.__name__}) - o bolum calismaz")
import cv2
print("  v4l2 backend:", "var" if hasattr(cv2, "CAP_V4L2") else "YOK")
EOF

echo "== kamera aygitlari"
ls -1 /dev/video* 2>/dev/null || echo "  /dev/video* yok - kamera takili mi?"
if command -v v4l2-ctl >/dev/null; then
  for d in /dev/video*; do
    echo "  -- $d"
    v4l2-ctl -d "$d" --list-formats-ext 2>/dev/null | grep -E "\[|1920x1080" | head -8 || true
  done
else
  echo "  (v4l2-ctl yok: sudo apt install v4l-utils  -> formatlari gorebilirsin)"
fi

cat <<'EOF'

== hazir
  source .venv/bin/activate
  python panel.py

  python panel.py --device /dev/video2 --out ./cikti --shots 3

Kullanici video grubunda degilse kamera acilmaz:
  sudo usermod -aG video "$USER"   # sonra oturumu kapat/ac
EOF
