#!/usr/bin/env bash
# Paneli ac. Kurulum eksikse once onu yapar.
#
#   ./run.sh
#   ./run.sh --device /dev/video2 --fourcc MJPG
#
# panel.py'ye verilen butun argumanlar oldugu gibi gecer.

cd "$(dirname "$0")" || exit 1
VENV=.venv

hazir() { [ -x "$1" ] && "$1" -c 'import cv2, numpy, PIL, tkinter' 2>/dev/null; }

PY="$VENV/bin/python"
if ! hazir "$PY"; then
  echo "kurulum eksik ya da yarim -> ./setup.sh calistiriliyor"
  echo
  ./setup.sh || { echo "kurulum tamamlanamadi - yukaridaki ozete bak"; exit 1; }
  echo
fi

if   hazir "$VENV/bin/python"; then PY="$VENV/bin/python"
elif hazir python3;            then PY=python3
else
  echo "HATA: calisir bir python bulunamadi. Tani icin:  ./setup.sh --diag"
  exit 1
fi

exec "$PY" panel.py "$@"
