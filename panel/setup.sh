#!/usr/bin/env bash
# TEK KURULUM. Sistem paketleri + sanal ortam + opencv/numpy/pillow +
# torch(CPU) + ultralytics. Hicbir sey elle yapilmiyor.
#
#   ./setup.sh            kur
#   ./setup.sh --core     sadece cekirdek (torch/ultralytics yok, ~1 dk)
#   ./setup.sh --diag     sadece tani (hicbir sey kurmaz)
#   ./setup.sh --force    sanal ortami sifirdan kur
#
# KATMANLI: once panelin ACILMASI icin gereken cekirdek kurulur, sonra agir
# katman (torch, ultralytics). Agir katman patlarsa panel yine acilir, sadece
# .pt model / YOLO bolumu calismaz. "Her sey ya da hicbir sey" degil.
#
# set -e YOK: bir adim patlarsa ekrani kirmiziya bogup kacmak yerine hatayi
# yakalayip sonda tek satirlik ozet veriyoruz.

cd "$(dirname "$0")" || exit 1
VENV=.venv
PY=${PYTHON:-python3}
LOG=/tmp
FAILED=()
UYARI=()
DIAG=0; FORCE=0; CORE=0

for a in "$@"; do
  case "$a" in
    --diag)  DIAG=1 ;;
    --force) FORCE=1 ;;
    --core)  CORE=1 ;;
    -h|--help)
      sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "bilinmeyen secenek: $a  (--diag --force --core --help)"; exit 1 ;;
  esac
done

if [ -t 1 ]; then R=$'\e[31m'; G=$'\e[32m'; Y=$'\e[33m'; B=$'\e[1m'; N=$'\e[0m'
else R=; G=; Y=; B=; N=; fi
ok()   { printf '  %sOK%s    %s\n'  "$G" "$N" "$1"; }
warn() { printf '  %sUYARI%s %s\n' "$Y" "$N" "$1"; UYARI+=("$1"); }
bad()  { printf '  %sHATA%s  %s\n' "$R" "$N" "$1"; FAILED+=("$1"); }
head_() { printf '\n%s== %s%s\n' "$B" "$1" "$N"; }
son()  { tail -"${2:-3}" "$1" 2>/dev/null | sed 's/^/        /'; }

# ------------------------------------------------------------------ sistem
head_ "sistem"
. /etc/os-release 2>/dev/null
ARCH=$(uname -m)
echo "  ${PRETTY_NAME:-$(uname -s)}  |  $ARCH  |  $($PY -V 2>&1)"

if ! command -v "$PY" >/dev/null 2>&1; then
  bad "python3 yok - once onu kur (apt install python3)"
fi

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  if command -v sudo >/dev/null 2>&1; then SUDO="sudo"; else
    warn "root degilsin ve sudo yok - sistem paketleri atlanacak"
  fi
fi

# libgl1/glib2: ultralytics bagimlilik olarak GUI'li opencv'yi cekebiliyor,
# o da bu kitapliklari ariyor. Kurulu olmalari hicbir sey bozmuyor.
if   command -v apt-get >/dev/null 2>&1; then
  MGR="apt";    PKGS="python3-venv python3-tk python3-dev v4l-utils libgl1 libglib2.0-0"
elif command -v dnf >/dev/null 2>&1; then
  MGR="dnf";    PKGS="python3-tkinter python3-devel v4l-utils mesa-libGL glib2"
elif command -v pacman >/dev/null 2>&1; then
  MGR="pacman"; PKGS="tk v4l-utils libglvnd glib2"
elif command -v zypper >/dev/null 2>&1; then
  MGR="zypper"; PKGS="python3-tk python3-devel v4l-utils Mesa-libGL1 glib2"
else
  MGR="";       PKGS=""; warn "paket yoneticisi taninmadi - sistem paketlerini atliyorum"
fi

sistem_kur() {
  echo "  eksik sistem paketi var, kuruluyor ($MGR): $PKGS"
  case "$MGR" in
    apt)    $SUDO apt-get update -qq              >/dev/null 2>&1
            $SUDO apt-get install -y -qq $PKGS    >"$LOG/panel_sys.log" 2>&1 ;;
    dnf)    $SUDO dnf install -y -q $PKGS         >"$LOG/panel_sys.log" 2>&1 ;;
    pacman) $SUDO pacman -S --noconfirm --needed $PKGS >"$LOG/panel_sys.log" 2>&1 ;;
    zypper) $SUDO zypper -n install $PKGS         >"$LOG/panel_sys.log" 2>&1 ;;
  esac
}

need_sys=0
$PY -c 'import tkinter'  2>/dev/null || need_sys=1
$PY -c 'import ensurepip' 2>/dev/null || need_sys=1

if [ $DIAG -eq 0 ] && [ -n "$MGR" ] && [ $need_sys -eq 1 ]; then
  if sistem_kur; then ok "sistem paketleri"; else
    bad "sistem paketleri kurulamadi"
    son "$LOG/panel_sys.log"
    printf '\n  %sSu komutu bir kez kendin calistir, sonra ./setup.sh tekrar:%s\n' "$Y" "$N"
    case "$MGR" in
      apt)    echo "      sudo apt-get install -y $PKGS" ;;
      dnf)    echo "      sudo dnf install -y $PKGS" ;;
      pacman) echo "      sudo pacman -S --needed $PKGS" ;;
      zypper) echo "      sudo zypper install $PKGS" ;;
    esac
    echo
  fi
elif [ $need_sys -eq 0 ]; then
  ok "sistem paketleri zaten tam"
fi

$PY -c 'import tkinter' 2>/dev/null && ok "tkinter" || bad "tkinter hala yok"

# ------------------------------------------------------------------ venv
head_ "sanal ortam"
if [ $DIAG -eq 0 ]; then
  [ $FORCE -eq 1 ] && rm -rf "$VENV"
  if [ ! -d "$VENV" ]; then
    if ! $PY -m venv "$VENV" >"$LOG/panel_venv.log" 2>&1; then
      # Ubuntu/Debian klasigi: python3-venv kurulu degilse "ensurepip is not
      # available" diye bir duvar basar. Once onu kurmayi dene, sonra virtualenv.
      if grep -qi "ensurepip" "$LOG/panel_venv.log" 2>/dev/null; then
        warn "python3-venv eksik (ensurepip yok), kuruluyor"
        [ "$MGR" = "apt" ] && $SUDO apt-get install -y -qq python3-venv >>"$LOG/panel_sys.log" 2>&1
        rm -rf "$VENV"
        $PY -m venv "$VENV" >>"$LOG/panel_venv.log" 2>&1
      fi
    fi
    if [ ! -x "$VENV/bin/python" ]; then
      warn "python -m venv basarisiz, virtualenv deneniyor"
      $PY -m pip install --user --quiet virtualenv >/dev/null 2>&1
      $PY -m virtualenv "$VENV" >>"$LOG/panel_venv.log" 2>&1 || \
        bad "sanal ortam kurulamadi - ayrinti: $LOG/panel_venv.log"
    fi
  fi
fi

if [ -x "$VENV/bin/python" ]; then
  VPY="$PWD/$VENV/bin/python"; ok "$VENV  ($("$VPY" -V 2>&1))"
else
  VPY="$PY"; warn "sanal ortam yok, sistem python'u kullanilacak"
fi
"$VPY" -c 'import tkinter' 2>/dev/null && ok "tkinter (sanal ortam icinden)" \
  || bad "sanal ortam tkinter goremiyor"

# PEP 668 (Debian 12+, Ubuntu 24.04): venv kurulamadiysa sistem python'u pip'i
# reddeder. O durumda tek cikis bu bayrak.
PIPX=(--disable-pip-version-check --no-input --timeout 60 --retries 3)
if [ "$VPY" = "$PY" ]; then PIPX+=(--break-system-packages); fi
pipi() { "$VPY" -m pip install --quiet "${PIPX[@]}" "$@"; }

# ------------------------------------------------------------------ cekirdek
if [ $DIAG -eq 0 ]; then
  head_ "cekirdek paketler  (panelin acilmasi icin)"
  "$VPY" -m pip install --quiet "${PIPX[@]}" --upgrade pip wheel >/dev/null 2>&1
  if pipi -r requirements.txt >"$LOG/panel_core.log" 2>&1; then
    ok "numpy + opencv-headless + pillow"
  else
    bad "cekirdek paketler kurulamadi - ayrinti: $LOG/panel_core.log"
    son "$LOG/panel_core.log" 5
  fi
fi

# ------------------------------------------------------------------ agir
if [ $DIAG -eq 0 ] && [ $CORE -eq 0 ]; then
  head_ "agir paketler  (.pt modeli ve YOLO icin)"

  if "$VPY" -c 'import torch, torchvision' 2>/dev/null; then
    ok "torch zaten var ($("$VPY" -c 'import torch;print(torch.__version__)' 2>/dev/null))"
  else
    : >"$LOG/panel_torch.log"
    echo "  torch (CPU) indiriliyor, birkac dakika surebilir..."
    tamam=1
    # x86_64: resmi CPU deposu. PyPI tekerlegi CUDA tasiyor (~2.5 GB) ve bu
    # makinede hicbir ise yaramiyor.
    # aarch64: CPU deposunda aarch64 tekerlegi yok, PyPI'daki zaten CPU.
    if [ "$ARCH" = "x86_64" ]; then
      SIRA=("--index-url https://download.pytorch.org/whl/cpu" "")
    else
      SIRA=("" "--index-url https://download.pytorch.org/whl/cpu")
    fi
    for ek in "${SIRA[@]}"; do
      # shellcheck disable=SC2086
      if pipi torch torchvision $ek >>"$LOG/panel_torch.log" 2>&1; then
        tamam=0; break
      fi
    done
    if [ $tamam -eq 0 ] && "$VPY" -c 'import torch, torchvision' 2>/dev/null; then
      ok "torch $("$VPY" -c 'import torch;print(torch.__version__)' 2>/dev/null)"
    else
      warn "torch kurulamadi -> '.pt modeli' secenegi calismaz (panel yine acilir)"
      warn "ayrinti: $LOG/panel_torch.log"
      son "$LOG/panel_torch.log"
    fi
  fi

  if pipi -r requirements-extra.txt >"$LOG/panel_extra.log" 2>&1; then
    ok "ultralytics + scipy"
  else
    warn "ultralytics/scipy kurulamadi -> YOLO bolumu calismaz (panel yine acilir)"
    warn "ayrinti: $LOG/panel_extra.log"
    son "$LOG/panel_extra.log" 5
  fi

  # ultralytics bagimlilik olarak GUI'li opencv-python'i cekiyor; headless'in
  # ustune binince Linux'ta klasik "Could not load the Qt platform plugin xcb"
  # hatasi cikiyor. Ikisi ayni cv2 klasorune yaziyor, biri gitmeli.
  if "$VPY" -m pip show opencv-python >/dev/null 2>&1 || \
     "$VPY" -m pip show opencv-contrib-python >/dev/null 2>&1; then
    "$VPY" -m pip uninstall -y -q opencv-python opencv-contrib-python \
        >"$LOG/panel_cv.log" 2>&1
    pipi --force-reinstall "opencv-python-headless>=4.8,<5" >>"$LOG/panel_cv.log" 2>&1
    if "$VPY" -c 'import cv2' 2>/dev/null; then
      ok "opencv cakismasi temizlendi (GUI'li surum kaldirildi, headless kaldi)"
    else
      bad "opencv bozuldu - ayrinti: $LOG/panel_cv.log"
      son "$LOG/panel_cv.log" 5
    fi
  fi
fi

# ------------------------------------------------------------------ kamera
head_ "kamera"
if ls /dev/video* >/dev/null 2>&1; then
  for d in /dev/video*; do
    if command -v v4l2-ctl >/dev/null 2>&1; then
      nm=$(v4l2-ctl -d "$d" --info 2>/dev/null | sed -n 's/.*Card type *: *//p' | head -1)
      yuy=$(v4l2-ctl -d "$d" --list-formats 2>/dev/null | grep -c "YUYV")
      echo "  $d  ${nm:-?}  $( [ "${yuy:-0}" -gt 0 ] && echo 'YUYV var' || echo 'YUYV yok' )"
    else
      echo "  $d"
    fi
  done
  if id -nG 2>/dev/null | tr ' ' '\n' | grep -qx video; then
    ok "kullanici 'video' grubunda"
  else
    if [ $DIAG -eq 0 ] && [ -n "$SUDO" ]; then
      $SUDO usermod -aG video "$USER" 2>/dev/null \
        && warn "'video' grubuna eklendin - OTURUMU KAPATIP AC, yoksa kamera acilmaz" \
        || warn "'video' grubunda degilsin: sudo usermod -aG video $USER"
    else
      warn "'video' grubunda degilsin: sudo usermod -aG video $USER"
    fi
  fi
else
  warn "/dev/video* yok - kamera takili degil (panel yine de acilir)"
fi

# ------------------------------------------------------------------ kontrol
head_ "kontrol  (panel.py --check)"
if "$VPY" panel.py --check; then
  KONTROL=0
else
  KONTROL=$?
  bad "panel.py --check gecmedi (yukariya bak)"
fi

# ------------------------------------------------------------------ ozet
head_ "ozet"
if [ ${#FAILED[@]} -eq 0 ]; then
  printf '  %spanel calismaya hazir%s\n' "$G" "$N"
  [ ${#UYARI[@]} -gt 0 ] && printf '  (%d uyari var, yukariya bak)\n' "${#UYARI[@]}"
  echo
  echo "    ./run.sh                    <- tek komut, paneli acar"
  echo
  echo "  ya da elle:"
  echo "    source $VENV/bin/activate"
  echo "    python panel.py"
  echo
  echo "  ayni ortam distorch icin de kullanilabilir:"
  echo "    cd .. && panel/$VENV/bin/python -m distorch.calibrate kare.jpg"
else
  printf '  %s%d sorun:%s\n' "$R" "${#FAILED[@]}" "$N"
  for f in "${FAILED[@]}"; do echo "    - $f"; done
  echo
  echo "  tani icin:  ./setup.sh --diag"
  echo "  sifirdan :  ./setup.sh --force"
  echo "  loglar   :  $LOG/panel_*.log"
fi
echo
exit ${#FAILED[@]}
