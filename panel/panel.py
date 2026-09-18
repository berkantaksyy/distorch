#!/usr/bin/env python3
"""Test paneli - USB kamera, distorsiyon duzeltme, kesme, YOLO. Servis degil.

Tkinter penceresi. Linux/V4L2. macOS'a ozgu hicbir sey yok.
Hicbir dosyani degistirmez. Matematik (Brown k1,k2 + capa donusumu) burada
kopya olarak duruyor; distorch paketi SADECE "bilezik + CNN" modunda, tembel
sekilde import ediliyor - o mod secilmezse panel distorch'suz da calisir.

    ./run.sh                   kur (gerekiyorsa) ve ac
    python3 panel.py
    python3 panel.py --device /dev/video2 --out ./cikti
    python3 panel.py --check   pencere acmadan ortami sina (setup.sh bunu cagirir)
"""
import argparse
import glob
import importlib
import json
import os
import platform
import sys
import threading
import time
from pathlib import Path

BURASI = Path(__file__).resolve().parent

# ultralytics calisma aninda eksik paket gorurse kendi kendine pip install
# dener: agsiz makinede dakikalarca takilir, sonunda yine hata verir.
# Kurulum setup.sh'in isi, calisma aninin degil.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("YOLO_VERBOSE", "false")


def _zorunlu(ad, paket):
    """Zorunlu bir paket yoksa 30 satirlik traceback yerine tek satir sonuc."""
    sys.stderr.write(
        f"\nHATA: '{ad}' bulunamadi (paket: {paket}).\n\n"
        f"  Kurulum tek komut:\n"
        f"      cd {BURASI}\n"
        f"      ./setup.sh\n\n"
        f"  Sonra:  ./run.sh\n\n")
    raise SystemExit(2)


try:
    import numpy as np
except ImportError:
    _zorunlu("numpy", "numpy")
try:
    import cv2
except ImportError:
    _zorunlu("cv2", "opencv-python-headless")
try:
    from PIL import Image, ImageTk
except ImportError:
    _zorunlu("PIL", "pillow")

# tkinter stdlib ama Debian/Ubuntu ayri paket olarak veriyor (python3-tk).
# Yoksa programi burada oldurmuyoruz, isaretliyoruz: --check yine calissin diye.
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    TK_HATA = ""
    _TkTaban = tk.Tk
except Exception as _e:                      # ImportError veya TclError
    TK_HATA = f"{type(_e).__name__}: {_e}"
    tk = ttk = filedialog = messagebox = None
    _TkTaban = object                        # sinif tanimlanabilsin; GUI acilmaz

CANON_F = 1920.0
ANCHORS = (0.30, 0.60)


# --------------------------------------------------------------- geometri

def radial_gain(r, k1, k2):
    r2 = np.asarray(r, np.float64) ** 2
    return 1.0 + k1 * r2 + k2 * r2 * r2


def radial_inverse(rd, k1, k2, n_fixed=8, n_newton=6):
    """f^-1(rd). Sabit nokta + Newton; Newton olmadan kosede ~7 px hata kaliyor."""
    rd = np.asarray(rd, np.float64)
    r = rd.copy()
    for _ in range(n_fixed):
        r = rd / np.maximum(radial_gain(r, k1, k2), 1e-9)
    for _ in range(n_newton):
        g = radial_gain(r, k1, k2)
        f = r * g - rd
        df = g + r * (2 * k1 * r + 4 * k2 * r ** 3)
        r = r - f / np.where(np.abs(df) < 1e-12, 1e-12, df)
    return r


def distort_points(pts, th):
    p = np.asarray(pts, np.float64).reshape(-1, 2)
    f, cx, cy = th.get("f", CANON_F), th["cx"], th["cy"]
    x, y = (p[:, 0] - cx) / f, (p[:, 1] - cy) / f
    g = radial_gain(np.hypot(x, y), th["k1"], th["k2"])
    return np.column_stack([x * g * f + cx, y * g * f + cy])


def undistort_points(pts, th):
    p = np.asarray(pts, np.float64).reshape(-1, 2)
    f, cx, cy = th.get("f", CANON_F), th["cx"], th["cy"]
    xd, yd = (p[:, 0] - cx) / f, (p[:, 1] - cy) / f
    rd = np.hypot(xd, yd)
    r = radial_inverse(rd, th["k1"], th["k2"])
    s = np.where(rd < 1e-12, 1.0, r / np.where(rd < 1e-12, 1.0, rd))
    return np.column_stack([xd * s * f + cx, yd * s * f + cy])


def kk_from_mags(m1, m2, anchors=ANCHORS):
    """(m1,m2) -> (k1,k2). g(r_i) = 1/m_i, r_i = m_i*capa_i."""
    a1, a2 = anchors
    r1, r2 = m1 * a1, m2 * a2
    A = np.array([[r1 ** 2, r1 ** 4], [r2 ** 2, r2 ** 4]], float)
    b = np.array([1.0 / m1 - 1.0, 1.0 / m2 - 1.0], float)
    k1, k2 = np.linalg.solve(A, b)
    return float(k1), float(k2)


def rotation(deg, size):
    cx, cy = size[0] / 2.0, size[1] / 2.0
    a = np.radians(-deg)
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, -sa, cx - ca * cx + sa * cy],
                     [sa, ca, cy - sa * cx - ca * cy]], float)


def fit_scale(th, size, n=200):
    """Duzeltilmis karenin TAMAMI tuvale sigsin diye olcek + kaydirma.

    Duzeltmede en cok tasan yer koseler degil KENAR ORTALARI: sadece 4 koseye
    bakan hesap 1920x1080'de ustte ~60 px disarida birakiyor (olculdu). O yuzden
    kenar boyunca ornekliyoruz.

    Tuval 1920x1080 kaliyor; degisen tek sey cikis odagi. distorch'un kendi
    notu: "f is fixed: straightness cannot observe focal length, and the k
    coefficients absorb the choice" - yani f bir olcum degil, secim.
    """
    w, h = size
    t = np.linspace(0.0, 1.0, n)
    z, bir = np.zeros_like(t), np.ones_like(t)
    border = np.vstack([np.column_stack([t * (w - 1), z]),
                        np.column_stack([t * (w - 1), bir * (h - 1)]),
                        np.column_stack([z, t * (h - 1)]),
                        np.column_stack([bir * (w - 1), t * (h - 1)])])
    u = undistort_points(border, th)
    lo, hi = u.min(0), u.max(0)
    k = float(min(w / max(hi[0] - lo[0], 1e-9), h / max(hi[1] - lo[1], 1e-9)))
    off = np.array([w, h], float) / 2.0 - (lo + hi) / 2.0 * k
    return k, off


def build_maps(th, size, roll_deg=0.0, fit=None):
    """Tek remap: duzeltme (+ istege bagli kadraj ve duzlestirme).

    fit=(k, off) verilirse hicbir sey kesilmez: duzeltilmis kadrajin tamami
    tuvalin icine sigdirilir, kenarlarda siyah yaylar kalir. fit=None eski
    davranis - tuval dolar ama ham karenin dis %32'si disarida kalir.
    """
    w, h = size
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float64)
    p = np.stack([gx.ravel(), gy.ravel()], 1)
    if fit is not None:
        k, off = fit
        p = (p - off) / k
    if roll_deg:
        m = rotation(-roll_deg, size)
        p = p @ m[:, :2].T + m[:, 2]
    src = distort_points(p, th)
    return (src[:, 0].reshape(h, w).astype(np.float32),
            src[:, 1].reshape(h, w).astype(np.float32))


def scale_theta(th, s):
    return {"k1": th["k1"], "k2": th["k2"], "f": th.get("f", CANON_F) * s,
            "cx": (th["cx"] + 0.5) * s - 0.5, "cy": (th["cy"] + 0.5) * s - 0.5}


# --------------------------------------------------------------- profil

def theta_from_profile(path):
    """distorch / OpenCV semasindaki profil JSON -> (theta, roll_deg)."""
    d = json.loads(Path(path).read_text())
    if "camera_matrix" in d:
        K, dc = d["camera_matrix"], d["dist_coeffs"][0]
        th = {"k1": float(dc[0]), "k2": float(dc[1]),
              "cx": float(K[0][2]), "cy": float(K[1][2]), "f": float(K[0][0])}
    else:
        th = {k: float(d[k]) for k in ("k1", "k2", "cx", "cy")}
        th["f"] = float(d.get("f", CANON_F))
    return th, float(d.get("roll_deg") or 0.0)


# --------------------------------------------------------------- distort CNN

class DistortNet:
    """distort_v2 / v3 vb. cikarim. Agirlik + meta dosyasi disaridan verilir."""

    INPUT = (512, 288)
    MID = (960, 540)
    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, weights, meta=None, bias=None):
        try:
            import torch, torchvision
            import torch.nn as nn
        except ImportError as e:
            raise RuntimeError(
                f"torch/torchvision yok ({e}). Kurmak icin: ./setup.sh") from e
        self.torch = torch
        wp = Path(weights)
        mp = Path(meta) if meta else Path(str(wp).replace(".pt", "_meta.json"))
        bp = Path(bias) if bias else Path(str(wp).replace(".pt", "_bias.json"))
        self.meta = json.loads(mp.read_text()) if mp.exists() else {}
        self.bias = (json.loads(bp.read_text()).get("sapma_medyan")
                     if bp.exists() else None) or {"m1": 0, "m2": 0, "cx": 0, "cy": 0}

        class Net(nn.Module):
            def __init__(s, hid=256, ch=64):
                super().__init__()
                m = torchvision.models.resnet18(weights=None)
                s.body = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool,
                                       m.layer1, m.layer2, m.layer3, m.layer4)
                fw, fh = DistortNet.INPUT[0] // 32, DistortNet.INPUT[1] // 32
                ys = torch.linspace(-1, 1, fh).view(fh, 1).expand(fh, fw)
                xs = torch.linspace(-1, 1, fw).view(1, fw).expand(fh, fw)
                s.register_buffer("coords",
                                  torch.stack([xs, ys, xs * xs + ys * ys], 0).unsqueeze(0),
                                  persistent=False)
                s.reduce = nn.Sequential(nn.Conv2d(515, ch, 1), nn.BatchNorm2d(ch),
                                         nn.ReLU(inplace=True))
                s.head = nn.Sequential(nn.Linear(ch * fh * fw, hid), nn.ReLU(inplace=True),
                                       nn.Dropout(0.1), nn.Linear(hid, 4))

            def forward(s, x):
                z = s.body(x)
                z = s.reduce(torch.cat([z, s.coords.expand(z.shape[0], -1, -1, -1)], 1))
                return s.head(z.flatten(1))

        self.net = Net()
        self.net.load_state_dict(torch.load(str(wp), map_location="cpu", weights_only=True))
        self.net.eval()
        self.name = wp.name

    def theta(self, frame, correct_bias=True):
        g = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if (g.shape[1], g.shape[0]) != self.MID:
            g = cv2.resize(g, self.MID, interpolation=cv2.INTER_AREA)
        g = cv2.resize(g, self.INPUT, interpolation=cv2.INTER_AREA)
        x = (np.stack([g] * 3, -1).astype(np.float32) / 255.0 - self.MEAN) / self.STD
        t = self.torch.from_numpy(x.transpose(2, 0, 1).copy()).unsqueeze(0)
        with self.torch.no_grad():
            o = self.net(t)[0].numpy().astype(np.float64)
        nm = self.meta.get("norm")
        if nm:
            o = o * np.asarray(nm["std"]) + np.asarray(nm["mean"])
        m1, m2, cx, cy = (float(v) for v in o)
        if correct_bias:
            b = self.bias
            m1 -= b["m1"]; m2 -= b["m2"]; cx -= b["cx"]; cy -= b["cy"]
        k1, k2 = kk_from_mags(m1, m2, tuple(self.meta.get("anchors", ANCHORS)))
        return {"k1": k1, "k2": k2, "cx": cx, "cy": cy, "f": CANON_F,
                "m1": m1, "m2": m2}


def distorch_sistem(frame, use_rings=True):
    """distorch'un tam sistemi: CNN + delik + kenar + solve (+ bilezik).

    Panel normalde distorch'u IMPORT ETMEZ, matematigi kendi icinde kopya tutar.
    Bu mod tek istisna ve import tembel: sadece bu mod secilince oluyor, distorch
    yoksa panel calismaya devam eder.

    CNN karar vermiyor, baslangic degeri veriyor; cikti geometrik cozumdur.
    use_rings=False -> sadece asama 1 (delik + kenar), bilezik yok.
    """
    kok = str(BURASI.parent)
    if kok not in sys.path:
        sys.path.insert(0, kok)
    from distorch import calibrate as C
    r = C.calibrate(frame, use_net=True, use_rings=use_rings)   # asla raise etmez
    if "theta" not in r:
        raise RuntimeError(r.get("reason", "cozulemedi"))
    if r.get("verdict") == "reject":
        # DIKKAT: reject verdiginde de bir theta donuyor, ama cozucu sinira
        # dayanmis olabiliyor (bos karede k1=3.0 cikti, olculdu). Onizlemeyi
        # sessizce mahvetmesin diye burada duruyoruz.
        ned = "; ".join(c["message"] for c in r.get("checks", [])
                        if c.get("level") == "reject")
        raise RuntimeError(f"reddedildi - {ned or 'kalite kapilari gecilmedi'}")
    return r


def sniff_pt(path):
    """.pt hangi aile? -> 'distorch' | 'yolo' | 'bilinmiyor'

    Ikisi de .pt uzantili ama tamamen ayri seyler:
      distorch agirligi -> ResNet-18 govde + (m1, m2, cx, cy) basi, duz state_dict
      yolo agirligi     -> ultralytics checkpoint, icinde pickle'lanmis model
    Yanlis yuvaya konani anlasilmaz bir torch hatasi yerine acik mesajla reddetmek
    icin arsivin pickle'ina bakiyoruz; modeli yuklemeye gerek yok.
    """
    import zipfile
    try:
        with zipfile.ZipFile(path) as z:
            pk = [n for n in z.namelist() if n.endswith("data.pkl")]
            if not pk:
                return "bilinmiyor"
            raw = z.read(pk[0])
            if b"ultralytics" in raw:
                return "yolo"
            if b"reduce.0.weight" in raw or b"body.0.weight" in raw:
                return "distorch"
    except Exception:
        pass
    return "bilinmiyor"


# --------------------------------------------------------------- kamera

def list_devices():
    out = sorted(glob.glob("/dev/video*"),
                 key=lambda p: int("".join(c for c in p if c.isdigit()) or 0))
    return out or ["/dev/video0"]


def fourcc_code(s):
    """VideoWriter_fourcc OpenCV 4.10'da kullanimdan kalkti, yenisi VideoWriter.fourcc.
    Ikisinden hangisi varsa onu kullan; surum yuzunden panel acilmasin diye."""
    fn = getattr(cv2, "VideoWriter_fourcc", None)
    if fn is None:
        fn = cv2.VideoWriter.fourcc
    return int(fn(*s))


class Camera:
    """V4L2, YUY2, sabit cozunurluk. Arka planda okur, son kareyi tutar."""

    def __init__(self, device, width=1920, height=1080, fps=30, fourcc="YUYV"):
        self.device, self.size, self.fps, self.fourcc = device, (width, height), fps, fourcc
        self.cap = None
        self.frame = None
        self.lock = threading.Lock()
        self.run = False
        self.err = ""
        self.real = (0, 0)
        self.measured_fps = 0.0
        self.thread = None

    def open(self):
        idx = self.device
        if isinstance(idx, str):
            # "/dev/video2" de olur, duz "2" de: ikisi de 2 numarali aygit
            if idx.startswith("/dev/video") or idx.strip().isdigit():
                idx = int("".join(c for c in idx if c.isdigit()) or 0)
        try:
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        except Exception as e:
            self.err = f"acilamadi: {self.device} ({type(e).__name__}: {e})"
            return False
        if not cap.isOpened():
            self.err = (f"acilamadi: {self.device} (v4l2)\n\n"
                        f"- aygit takili mi:  ls -l /dev/video*\n"
                        f"- izin var mi:      sudo usermod -aG video $USER  (sonra oturumu kapat/ac)\n"
                        f"- baska bir program kullaniyor olabilir")
            return False
        try:
            cap.set(cv2.CAP_PROP_FOURCC, fourcc_code(self.fourcc))
        except Exception:
            pass                                  # format tutmazsa surucu kendi secer
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, f = cap.read()
        if not ok or f is None:
            cap.release()
            self.err = ("kare okunamadi - format/cozunurluk desteklenmiyor olabilir\n\n"
                        f"destekledigi formatlar:  v4l2-ctl -d {self.device} --list-formats-ext")
            return False
        self.real = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.cap = cap
        self.frame = f
        self.run = True
        self.err = ""
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return True

    def _loop(self):
        t0, n = time.time(), 0
        while self.run:
            try:
                ok, f = self.cap.read()
            except Exception:
                break
            if not ok or f is None:
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = f
            n += 1
            if n >= 15:
                self.measured_fps = n / (time.time() - t0)
                t0, n = time.time(), 0

    def grab(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def close(self):
        """Okuma is parcacigi bitmeden release ETME.

        read() YUYV 1080p'de 200 ms bloklayabiliyor; sabit bir sleep yetmiyor,
        release okuma sirasinda calisirsa V4L2 kilitleniyor ya da cokuyor.
        """
        self.run = False
        if self.thread is not None:
            self.thread.join(timeout=3.0)
            self.thread = None
        if self.cap:
            self.cap.release()
        self.cap = None


# --------------------------------------------------------------- YOLO

class Yolo:
    def __init__(self, weights):
        try:
            from ultralytics import YOLO as _Y
        except ImportError as e:
            raise RuntimeError(f"ultralytics yok ({e}). Kurmak icin: ./setup.sh") from e
        self.m = _Y(str(weights))
        self.name = Path(weights).name
        self.task = getattr(self.m, "task", "?")

    def run(self, img, conf=0.25, retina=True):
        r = self.m.predict(img, conf=conf, retina_masks=retina, verbose=False)[0]
        out = []
        n = 0 if r.boxes is None else len(r.boxes)
        for i in range(n):
            x1, y1, x2, y2 = [float(v) for v in r.boxes.xyxy[i].cpu().numpy()]
            d = {"box": (x1, y1, x2, y2), "conf": float(r.boxes.conf[i]),
                 "cls": int(r.boxes.cls[i]), "name": r.names.get(int(r.boxes.cls[i]), "?"),
                 "rot": None}
            if r.masks is not None and i < len(r.masks.data):
                m = (r.masks.data[i].cpu().numpy() > 0.5).astype(np.uint8)
                if m.shape[:2] != img.shape[:2]:
                    m = cv2.resize(m, (img.shape[1], img.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
                cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if cnts:
                    c = max(cnts, key=cv2.contourArea)
                    (cx, cy), (w, h), ang = cv2.minAreaRect(c)
                    lo, sh = (w, h) if w >= h else (h, w)
                    if w < h:
                        ang += 90.0
                    d["rot"] = {"c": (cx, cy), "uzun": lo, "kisa": sh,
                                "aci": (ang + 90.0) % 180.0 - 90.0, "kontur": c}
            out.append(d)
        return out


# --------------------------------------------------------------- panel

def _sayi(var, yedek, tur=int):
    """Tk kutusuna sacma bir sey yazilmissa TclError firlatma, yedege don."""
    try:
        return tur(var.get())
    except Exception:
        return yedek


class Panel(_TkTaban):

    def __init__(self, args):
        super().__init__()
        self.title("Test paneli")
        self.cam = None
        self.net = None
        self.yolo = None
        self.theta = None
        self.roll = 0.0
        self.maps = None
        self.maps_key = None
        self.shot = 0
        self.out = self._out_hazirla(Path(args.out))

        # 1366x768 panel PC'de sag sutun ekran disina tasmasin
        self.preview_w = max(480, min(1040, self.winfo_screenwidth() - 420))
        self.preview_h = max(320, min(640, self.winfo_screenheight() - 260))

        self.v_dev = tk.StringVar(value=args.device or list_devices()[0])
        self.v_w = tk.IntVar(value=args.width)
        self.v_h = tk.IntVar(value=args.height)
        self.v_fps = tk.IntVar(value=args.fps)
        self.v_fourcc = tk.StringVar(value=args.fourcc)

        self.v_mode = tk.StringVar(value="kapali")  # kapali|profil|model|sistem
        self.v_kadraj = tk.StringVar(value="sigdir")   # sigdir (kesme yok) | tam
        self.v_prof = tk.StringVar(value="")
        self.v_wts = tk.StringVar(value="")
        self.v_bias = tk.BooleanVar(value=True)
        self.v_level = tk.BooleanVar(value=False)
        self.v_rolltxt = tk.StringVar(value="0.000")

        self.v_cut = {k: tk.DoubleVar(value=0.0) for k in ("ust", "alt", "sol", "sag")}

        self.v_yolo_on = tk.BooleanVar(value=False)
        self.v_yolo_w = tk.StringVar(value="")
        self.v_conf = tk.DoubleVar(value=0.25)
        self.v_retina = tk.BooleanVar(value=True)
        self.v_mmpx = tk.StringVar(value="")

        self.v_n = tk.IntVar(value=args.shots)
        self.v_gap = tk.IntVar(value=args.gap_ms)
        self.v_tag = tk.StringVar(value="test")

        self.rapor = None           # distorch tam sistem raporu (sistem modu)
        self.sistem_ozet = ""       # durum cubugunda kalici kalsin diye
        self.fit_k = None           # kadraj olcegi; None = kesme var (tam)
        self._det = []              # onizlemede yeniden kullanilan son tespit
        self._det_next = 0.0        # bir sonraki YOLO kosusunun en erken zamani

        self.status = tk.StringVar(value="hazir")
        self._build()
        self.protocol("WM_DELETE_WINDOW", self._quit)
        self.after(40, self._tick)

    # ---------------------------------------------------------- arayuz
    def _build(self):
        root = ttk.Frame(self, padding=8)
        root.pack(fill="both", expand=True)
        left = ttk.Frame(root)
        left.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(root, width=340)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        self.canvas = tk.Label(left, background="#111")
        self.canvas.pack(fill="both", expand=True)
        ttk.Label(left, textvariable=self.status, wraplength=self.preview_w).pack(
            anchor="w", pady=(6, 0))

        def box(t):
            f = ttk.LabelFrame(right, text=t, padding=6)
            f.pack(fill="x", pady=4)
            return f

        # kamera
        c = box("Kamera")
        r = ttk.Frame(c); r.pack(fill="x")
        ttk.Label(r, text="aygit").pack(side="left")
        self.cb_dev = ttk.Combobox(r, textvariable=self.v_dev, values=list_devices(), width=14)
        self.cb_dev.pack(side="left", padx=4)
        ttk.Button(r, text="tara", width=5,
                   command=lambda: self.cb_dev.configure(values=list_devices())).pack(side="left")
        r = ttk.Frame(c); r.pack(fill="x", pady=2)
        for lbl, var, w in (("en", self.v_w, 6), ("boy", self.v_h, 6),
                            ("fps", self.v_fps, 4)):
            ttk.Label(r, text=lbl).pack(side="left")
            ttk.Entry(r, textvariable=var, width=w).pack(side="left", padx=(2, 6))
        ttk.Label(r, text="format").pack(side="left")
        ttk.Combobox(r, textvariable=self.v_fourcc, values=["YUYV", "MJPG", "GREY"],
                     width=6).pack(side="left", padx=2)
        r = ttk.Frame(c); r.pack(fill="x", pady=2)
        self.btn_cam = ttk.Button(r, text="KAMERAYI AC", command=self._toggle_cam)
        self.btn_cam.pack(side="left", fill="x", expand=True)

        # duzeltme
        c = box("1) DISTORSIYON DUZELTME   (distorch agirligi / profil)")
        for t, v in (("kapali (ham)", "kapali"),
                     ("profil JSON", "profil"),
                     ("1) sadece CNN   (.pt agirligi)", "model"),
                     ("2) bilezik + CNN   (distorch tam sistem)", "sistem")):
            ttk.Radiobutton(c, text=t, value=v, variable=self.v_mode,
                            command=self._reset_theta).pack(anchor="w")
        r = ttk.Frame(c); r.pack(fill="x", pady=2)
        ttk.Entry(r, textvariable=self.v_prof).pack(side="left", fill="x", expand=True)
        ttk.Button(r, text="...", width=3,
                   command=lambda: self._pick(self.v_prof, [("profil", "*.json")])).pack(side="left")
        r = ttk.Frame(c); r.pack(fill="x", pady=2)
        ttk.Entry(r, textvariable=self.v_wts).pack(side="left", fill="x", expand=True)
        ttk.Button(r, text="...", width=3,
                   command=lambda: self._pick(self.v_wts, [("distorch agirligi", "*.pt")],
                                              want="distorch")).pack(side="left")
        ttk.Checkbutton(c, text="sapma duzeltmesi (bias json varsa)",
                        variable=self.v_bias, command=self._reset_theta).pack(anchor="w")
        r = ttk.Frame(c); r.pack(fill="x")
        ttk.Checkbutton(r, text="roll'u sifirla", variable=self.v_level,
                        command=self._reset_maps).pack(side="left")
        ttk.Entry(r, textvariable=self.v_rolltxt, width=8).pack(side="left", padx=4)
        ttk.Label(r, text="derece").pack(side="left")
        r = ttk.Frame(c); r.pack(fill="x", pady=(4, 0))
        ttk.Label(r, text="kadraj").pack(side="left")
        for t, v in (("kesme yok", "sigdir"), ("tam (kenar kesilir)", "tam")):
            ttk.Radiobutton(r, text=t, value=v, variable=self.v_kadraj,
                            command=self._reset_maps).pack(side="left", padx=(6, 0))

        # kesme
        c = box("Kesme  (duzeltmeden SONRA, % olarak)")
        for k in ("ust", "alt", "sol", "sag"):
            r = ttk.Frame(c); r.pack(fill="x")
            ttk.Label(r, text=k, width=5).pack(side="left")
            lab = ttk.Label(r, text="0", width=4, anchor="e")
            ttk.Scale(r, from_=0, to=40, variable=self.v_cut[k], orient="horizontal",
                      command=lambda v, l=lab: l.configure(text=f"{float(v):.0f}")
                      ).pack(side="left", fill="x", expand=True)
            lab.pack(side="left")
        ttk.Button(c, text="sifirla", command=self._cut_reset).pack(anchor="e")

        # yolo
        c = box("2) NESNE TESPITI   (YOLO agirligi - ayri model)")
        ttk.Checkbutton(c, text="calistir", variable=self.v_yolo_on).pack(anchor="w")
        r = ttk.Frame(c); r.pack(fill="x", pady=2)
        ttk.Entry(r, textvariable=self.v_yolo_w).pack(side="left", fill="x", expand=True)
        ttk.Button(r, text="...", width=3,
                   command=lambda: self._pick(self.v_yolo_w, [("yolo agirligi", "*.pt")],
                                              reset_yolo=True, want="yolo")).pack(side="left")
        r = ttk.Frame(c); r.pack(fill="x")
        ttk.Label(r, text="conf").pack(side="left")
        lab = ttk.Label(r, text="0.25", width=5, anchor="e")
        ttk.Scale(r, from_=0.01, to=0.9, variable=self.v_conf, orient="horizontal",
                  command=lambda v, l=lab: l.configure(text=f"{float(v):.2f}")
                  ).pack(side="left", fill="x", expand=True)
        lab.pack(side="left")
        ttk.Checkbutton(c, text="retina_masks (tam cozunurluk maske)",
                        variable=self.v_retina).pack(anchor="w")
        r = ttk.Frame(c); r.pack(fill="x")
        ttk.Label(r, text="mm/px").pack(side="left")
        ttk.Entry(r, textvariable=self.v_mmpx, width=10).pack(side="left", padx=4)

        # kayit
        c = box("Kayit")
        r = ttk.Frame(c); r.pack(fill="x")
        ttk.Label(r, text="etiket").pack(side="left")
        ttk.Entry(r, textvariable=self.v_tag, width=12).pack(side="left", padx=4)
        ttk.Label(r, text="adet").pack(side="left")
        ttk.Spinbox(r, from_=1, to=20, textvariable=self.v_n, width=4).pack(side="left", padx=2)
        ttk.Label(r, text="ms").pack(side="left")
        ttk.Spinbox(r, from_=0, to=2000, increment=50, textvariable=self.v_gap,
                    width=6).pack(side="left", padx=2)
        ttk.Label(c, text="her basista: N adet duzeltilmis kare + tek ozet gorsel + tek json",
                  foreground="#555", wraplength=310).pack(anchor="w")
        ttk.Button(c, text="KARE AL", command=self._shoot).pack(fill="x", pady=3)
        ttk.Button(c, text="cikti klasorunu sec", command=self._pick_out).pack(fill="x")
        self.lbl_out = ttk.Label(c, text=str(self.out), wraplength=310, foreground="#555")
        self.lbl_out.pack(anchor="w")

    def _out_hazirla(self, p):
        """Cikti klasoru yazilamiyorsa panel acilmadan cokmesin, /tmp'ye dus."""
        for aday in (p, Path("/tmp") / "panel_cikti"):
            try:
                aday.mkdir(parents=True, exist_ok=True)
                t = aday / ".yazma_denemesi"
                t.write_text("x"); t.unlink()
                if aday != p:
                    sys.stderr.write(f"UYARI: {p} yazilamadi, cikti -> {aday}\n")
                return aday
            except Exception:
                continue
        return p

    def _cut_reset(self):
        for v in self.v_cut.values():
            v.set(0.0)

    def _pick(self, var, types, reset_yolo=False, want=None):
        agirlik = BURASI.parent / "weights"
        p = filedialog.askopenfilename(
            filetypes=types + [("hepsi", "*.*")],
            initialdir=str(agirlik if agirlik.is_dir() else BURASI))
        if not p:
            return
        if want:
            got = sniff_pt(p)
            if got != want and got != "bilinmiyor":
                other = {"distorch": "YOLO", "yolo": "distorch"}[got]
                messagebox.showerror(
                    "yanlis model",
                    f"Bu bir {other} agirligi.\n\n{Path(p).name}\n\n"
                    f"Bu alan {want} agirligi bekliyor. Ikisi ayri seyler:\n"
                    f"  distorch .pt -> lens bozulmasini cozer (k1, k2, cx, cy)\n"
                    f"  yolo .pt     -> siseyi bulur (kutu / maske)")
                return
        var.set(p)
        self._reset_theta()
        if reset_yolo:
            self.yolo = None
            self._det = []

    def _pick_out(self):
        p = filedialog.askdirectory(initialdir=str(self.out.parent))
        if p:
            self.out = Path(p)
            self.out.mkdir(parents=True, exist_ok=True)
            self.lbl_out.configure(text=str(self.out))

    def _reset_theta(self):
        self.theta = None
        self.net = None
        self.rapor = None
        self.sistem_ozet = ""
        self._reset_maps()

    def _reset_maps(self):
        self.maps = None
        self.maps_key = None

    # ---------------------------------------------------------- kamera
    def _toggle_cam(self):
        if self.cam:
            self.cam.close()
            self.cam = None
            self.btn_cam.configure(text="KAMERAYI AC")
            self.status.set("kamera kapali")
            return
        w = _sayi(self.v_w, 1920); h = _sayi(self.v_h, 1080)
        c = Camera(self.v_dev.get(), w, h, _sayi(self.v_fps, 30), self.v_fourcc.get())
        self.status.set(f"kamera aciliyor: {self.v_dev.get()} ...")
        self.update_idletasks()
        if not c.open():
            self.status.set("kamera acilamadi")
            messagebox.showerror("kamera", c.err)
            return
        self.cam = c
        self.btn_cam.configure(text="KAMERAYI KAPAT")
        if c.real != (w, h):
            self.status.set(f"DIKKAT: istenen {w}x{h}, alinan {c.real[0]}x{c.real[1]}")

    # ---------------------------------------------------------- isleme
    def _get_theta(self, frame):
        mode = self.v_mode.get()
        if mode == "kapali":
            return None
        if mode == "profil":
            if self.theta is None and self.v_prof.get():
                try:
                    self.theta, self.roll = theta_from_profile(self.v_prof.get())
                    self.v_rolltxt.set(f"{self.roll:.3f}")
                except Exception as e:
                    self.status.set(f"profil okunamadi: {e}")
                    self.v_mode.set("kapali")
            return self.theta
        if mode == "sistem":
            if self.theta is None:
                try:
                    self.status.set("distorch tam sistem calisiyor (bilezik dahil)...")
                    self.update_idletasks()
                    r = distorch_sistem(frame, use_rings=True)
                    self.theta, self.rapor = r["theta"], r
                    s2 = r.get("stage2") or {}
                    self.sistem_ozet = (
                        f"   distorch {r.get('verdict')}"
                        f"  bilezik {s2.get('n_rings', 0)}"
                        f"{'+' if s2.get('used') else '-'}"
                        f"  kose {r.get('quality', {}).get('corner_px')} px")
                    self.status.set(self.sistem_ozet.strip())
                except Exception as e:
                    self.sistem_ozet = ""
                    self.status.set(f"distorch calismadi: {type(e).__name__}: {e}")
                    self.v_mode.set("kapali")
                    return None
            return self.theta
        if self.net is None and self.v_wts.get():
            try:
                self.status.set("model yukleniyor...")
                self.update_idletasks()
                self.net = DistortNet(self.v_wts.get())
                self.status.set(f"model: {self.net.name}")
            except Exception as e:
                self.status.set(f"model yuklenemedi: {e}")
                self.v_mode.set("kapali")
                return None
        if self.net is None:
            return None
        if self.theta is None:                       # ilk karede bir kez cozulur
            self.theta = self.net.theta(frame, self.v_bias.get())
        return self.theta

    def correct(self, frame):
        th = self._get_theta(frame)
        if th is None:
            return frame, None
        h, w = frame.shape[:2]
        t = th if (w, h) == (1920, 1080) else scale_theta(th, w / 1920.0)
        roll = _sayi(self.v_rolltxt, 0.0, float) if self.v_level.get() else 0.0
        sigdir = self.v_kadraj.get() == "sigdir"
        key = (round(t["k1"], 6), round(t["k2"], 6), round(t["cx"], 2),
               round(t["cy"], 2), w, h, round(roll, 4), sigdir)
        if self.maps is None or self.maps_key != key:
            # 1920x1080'de 1-2 sn suruyor; donma sanilmasin diye haber ver
            self.status.set("haritalar hazirlaniyor...")
            self.update_idletasks()
            fit = fit_scale(t, (w, h)) if sigdir else None
            self.fit_k = fit[0] if fit else None
            self.maps = build_maps(t, (w, h), roll, fit)
            self.maps_key = key
        return cv2.remap(frame, self.maps[0], self.maps[1], cv2.INTER_LINEAR), th

    def cut(self, img):
        h, w = img.shape[:2]
        y0 = int(h * self.v_cut["ust"].get() / 100.0)
        y1 = h - int(h * self.v_cut["alt"].get() / 100.0)
        x0 = int(w * self.v_cut["sol"].get() / 100.0)
        x1 = w - int(w * self.v_cut["sag"].get() / 100.0)
        if y1 - y0 < 16 or x1 - x0 < 16:
            return img
        return img[y0:y1, x0:x1]

    def _yolo_hazir(self):
        if not self.v_yolo_on.get() or not self.v_yolo_w.get():
            return False
        if self.yolo is None:
            try:
                self.status.set("yolo yukleniyor...")
                self.update_idletasks()
                self.yolo = Yolo(self.v_yolo_w.get())
                self.status.set(f"yolo: {self.yolo.name} ({self.yolo.task})")
            except Exception as e:
                self.status.set(f"yolo yuklenemedi: {e}")
                self.v_yolo_on.set(False)
                return False
        return True

    def detect(self, img, throttle=False):
        """throttle=True: onizleme. CPU'da bir kosu 1-2 sn; her karede calistirirsak
        arayuz hic nefes alamiyor. Kosu suresi kadar bekleyip son tespiti yeniden
        cizeriz. KARE AL'da throttle yok, olcum karesi kendi tespitiyle kaydedilir."""
        if not self._yolo_hazir():
            self._det = []
            return img, []
        if throttle and time.time() < self._det_next:
            return self._draw(img, self._det), self._det
        try:
            t0 = time.time()
            det = self.yolo.run(img, self.v_conf.get(), self.v_retina.get())
            self._det_next = time.time() + max(0.25, time.time() - t0)
        except Exception as e:
            self.status.set(f"yolo hatasi: {e}")
            self._det_next = time.time() + 1.0
            return img, []
        self._det = det
        return self._draw(img, det), det

    def _draw(self, img, det):
        if not det:
            return img
        mm = _sayi(self.v_mmpx, None, float) if self.v_mmpx.get().strip() else None
        vis = img.copy()
        F = cv2.FONT_HERSHEY_SIMPLEX
        for d in det:
            x1, y1, x2, y2 = [int(v) for v in d["box"]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (60, 60, 255), 2)
            txt = f'{d["name"]} {d["conf"]:.2f}'
            if d["rot"]:
                rt = d["rot"]
                cv2.drawContours(vis, [rt["kontur"]], -1, (255, 200, 60), 1)
                pts = cv2.boxPoints((rt["c"], (rt["uzun"], rt["kisa"]),
                                     rt["aci"])).astype(int)
                cv2.polylines(vis, [pts], True, (80, 230, 80), 2)
                d["en_boy"] = rt["kisa"] / max(rt["uzun"], 1e-6)
                txt += f'  aci {rt["aci"]:+.1f}  en/boy {d["en_boy"]:.3f}'
                if mm:
                    d["uzun_mm"] = rt["uzun"] * mm
                    d["kisa_mm"] = rt["kisa"] * mm
                    txt += f'  {rt["uzun"]*mm:.0f}x{rt["kisa"]*mm:.0f} mm'
            cv2.putText(vis, txt, (x1, max(16, y1 - 6)), F, 0.55, (80, 230, 80), 2)
        return vis

    # ---------------------------------------------------------- dongu
    def _tick(self):
        if self.cam:
            f = self.cam.grab()
            if f is not None:
                try:
                    cor, th = self.correct(f)
                    img = self.cut(cor)
                    img, det = self.detect(img, throttle=True)
                    self._show(img)
                    s = (f"{self.cam.real[0]}x{self.cam.real[1]} @ "
                         f"{self.cam.measured_fps:.1f} fps   kesim sonrasi "
                         f"{img.shape[1]}x{img.shape[0]}")
                    if th:
                        s += f"   k1 {th['k1']:+.4f} k2 {th['k2']:+.4f} cx {th['cx']:.0f} cy {th['cy']:.0f}"
                        s += (f"   kadraj sigdir k={self.fit_k:.3f}" if self.fit_k
                              else "   kadraj tam (kenar kesiliyor)")
                        s += self.sistem_ozet
                    if det:
                        s += f"   {len(det)} tespit"
                    self.status.set(s)
                except Exception as e:
                    self.status.set(f"hata: {type(e).__name__}: {e}")
        self.after(40, self._tick)

    def _show(self, img):
        h, w = img.shape[:2]
        if h < 1 or w < 1:
            return
        s = min(self.preview_w / w, self.preview_h / h, 1.0)
        v = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))),
                       interpolation=cv2.INTER_AREA)
        im = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(v, cv2.COLOR_BGR2RGB)))
        self.canvas.configure(image=im)
        self.canvas.image = im

    # ---------------------------------------------------------- cekim
    def _shoot(self):
        if not self.cam:
            messagebox.showwarning("kamera", "once kamerayi ac")
            return
        try:
            self._shoot_gercek()
        except Exception as e:                # disk dolu, izin yok, model patladi
            self.status.set(f"kayit basarisiz: {type(e).__name__}: {e}")
            messagebox.showerror("kayit", f"{type(e).__name__}: {e}\n\nklasor: {self.out}")

    def _shoot_gercek(self):
        """N kare: her biri icin SADECE duzeltilmis PNG, hepsi icin tek ozet
        gorsel (distorch + YOLO cizimli, alt alta) ve tek kayit.json."""
        self.shot += 1
        tag = self.v_tag.get().strip() or "test"
        stamp = time.strftime("%Y%m%d_%H%M%S")
        d = self.out / f"{tag}_{stamp}"
        d.mkdir(parents=True, exist_ok=True)

        meta = {"etiket": tag, "zaman": stamp, "mod": self.v_mode.get(),
                "kadraj": self.v_kadraj.get(),
                "kadraj_olcek": None if self.fit_k is None else round(self.fit_k, 4),
                "sapma_duzeltmesi": bool(self.v_bias.get()),
                "kesme": {k: round(v.get(), 2) for k, v in self.v_cut.items()},
                "kamera": {"aygit": self.v_dev.get(), "format": self.v_fourcc.get(),
                           "cozunurluk": list(self.cam.real)},
                "kareler": []}
        if self.v_mode.get() == "sistem" and self.rapor:
            r = self.rapor
            meta["distorch"] = {"verdict": r.get("verdict"),
                                "bilezik": (r.get("stage2") or {}).get("n_rings"),
                                "bilezik_kullanildi": (r.get("stage2") or {}).get("used"),
                                "kalite": r.get("quality"),
                                "mm_per_px_panel": r.get("mm_per_px_panel")}

        gap = _sayi(self.v_gap, 150)
        ozet = []
        for i in range(max(1, _sayi(self.v_n, 3))):
            f = self.cam.grab()
            if f is None:
                continue
            cor, th = self.correct(f)
            img = self.cut(cor)
            vis, det = self.detect(img)                 # olcum karesi: throttle yok
            cv2.imwrite(str(d / f"{i+1:02d}_distorch.png"), img)
            ozet.append(vis if det else img)
            meta["kareler"].append({
                "no": i + 1,
                "dosya": f"{i+1:02d}_distorch.png",
                "theta": None if th is None else {k: round(th[k], 6)
                                                  for k in ("k1", "k2", "cx", "cy")},
                "tespit": [{k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in x.items()
                            if k in ("name", "conf", "en_boy", "uzun_mm", "kisa_mm")}
                           for x in det]})
            if gap:
                self.update()
                time.sleep(gap / 1000.0)

        if ozet:
            w = min(x.shape[1] for x in ozet)
            satir = [x if x.shape[1] == w else
                     cv2.resize(x, (w, int(x.shape[0] * w / x.shape[1])),
                                interpolation=cv2.INTER_AREA) for x in ozet]
            kare = np.vstack(satir)
            if kare.shape[1] > 1280:                    # ozet gorsel, tam cozunurluk gerekmez
                k = 1280.0 / kare.shape[1]
                kare = cv2.resize(kare, (1280, int(kare.shape[0] * k)),
                                  interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(d / "ozet_yolo.jpg"), kare, [cv2.IMWRITE_JPEG_QUALITY, 90])
            meta["ozet"] = "ozet_yolo.jpg"

        (d / "kayit.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        self.status.set(f"{len(meta['kareler'])} kare + ozet -> {d}")

    def _quit(self):
        if self.cam:
            self.cam.close()
        self.destroy()


# --------------------------------------------------------------- kontrol

def _geometri_sinavi():
    """Matematik dogru mu: distort -> undistort geri doniyor mu, harita cikiyor mu."""
    th = {"k1": -0.35, "k2": 0.12, "cx": 962.0, "cy": 541.0, "f": CANON_F}
    rng = np.random.default_rng(0)
    p = np.column_stack([rng.uniform(0, 1920, 500), rng.uniform(0, 1080, 500)])
    hata = float(np.abs(undistort_points(distort_points(p, th), th) - p).max())
    k1, k2 = kk_from_mags(1.04, 1.11)
    mx, my = build_maps(dict(th, k1=k1, k2=k2), (64, 48), roll_deg=3.0)
    harita = (mx.shape == (48, 64) and bool(np.isfinite(mx).all())
              and bool(np.isfinite(my).all()))
    return hata, harita


def kontrol():
    """Pencere acmadan ortami sinar. setup.sh bunu son adim olarak cagiriyor.
    Cikis kodu 0 -> panel acilir. 1 -> acilmaz, sebebi yukarida yazar."""
    G, R, Y, N = "\033[32m", "\033[31m", "\033[33m", "\033[0m"
    if not sys.stdout.isatty():
        G = R = Y = N = ""
    kotu = []

    print("== ortam")
    print(f"  python      {sys.version.split()[0]}   {sys.executable}")
    print(f"  sistem      {platform.system()} {platform.release()}  {platform.machine()}")
    print(f"  numpy       {np.__version__}")
    print(f"  opencv      {cv2.__version__}")
    if TK_HATA:
        print(f"  {R}tkinter     YOK -> {TK_HATA}{N}")
        kotu.append("tkinter (Debian/Ubuntu: sudo apt install -y python3-tk)")
    else:
        print(f"  tkinter     {tk.TkVersion}")

    print("\n== istege bagli paketler  (yoksa panel yine acilir)")
    for mod, ne in (("torch", "distorch .pt modeli"), ("torchvision", "distorch .pt modeli"),
                    ("ultralytics", "YOLO tespiti"), ("scipy", "distorch cozucusu")):
        try:
            m = importlib.import_module(mod)
            print(f"  {G}var{N}  {mod:14s} {getattr(m, '__version__', '')}  ({ne})")
        except Exception as e:
            print(f"  {Y}yok{N}  {mod:14s} {type(e).__name__}  -> {ne} calismaz")

    print("\n== matematik")
    try:
        hata, harita = _geometri_sinavi()
        iyi = hata < 1e-3 and harita
        print(f"  {(G+'OK'+N) if iyi else (R+'HATA'+N)}  distort/undistort geri donus "
              f"{hata:.2e} px, harita {'cikti' if harita else 'CIKMADI'}")
        if not iyi:
            kotu.append("geometri sinavi")
    except Exception as e:
        print(f"  {R}HATA{N}  {type(e).__name__}: {e}")
        kotu.append("geometri sinavi")

    print("\n== kamera")
    v4l2 = hasattr(cv2, "CAP_V4L2")
    print(f"  v4l2 backend  {'var' if v4l2 else 'YOK'}")
    ayg = sorted(glob.glob("/dev/video*"))
    if ayg:
        for d in ayg:
            print(f"  {d}")
    else:
        print(f"  {Y}/dev/video* yok{N} - kamera takili degil (panel yine acilir)")

    print("\n== ekran")
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        print(f"  DISPLAY={os.environ.get('DISPLAY', '')} "
              f"WAYLAND_DISPLAY={os.environ.get('WAYLAND_DISPLAY', '')}")
    elif platform.system() == "Linux":
        print(f"  {Y}DISPLAY yok{N} - SSH ile baglandiysan 'ssh -X' gerekir, "
              f"panel masaustunde acilmali")

    print()
    if kotu:
        print(f"{R}panel ACILMAZ:{N}")
        for k in kotu:
            print(f"  - {k}")
        return 1
    print(f"{G}panel acilabilir.{N}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="distorsiyon/YOLO test paneli")
    ap.add_argument("--device", default=None)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--fourcc", default="YUYV")
    ap.add_argument("--shots", type=int, default=3)
    ap.add_argument("--gap-ms", type=int, default=150)
    ap.add_argument("--out", default=None,
                    help="cikti klasoru (varsayilan: panel.py'nin yanindaki cikti/)")
    ap.add_argument("--check", action="store_true",
                    help="pencere acmadan ortami sina, cikis kodu dondur")
    args = ap.parse_args()

    if args.check:
        raise SystemExit(kontrol())

    if args.out is None:
        args.out = str(BURASI / "cikti")

    if TK_HATA:
        sys.stderr.write(
            f"\nHATA: tkinter yuklenemedi -> {TK_HATA}\n\n"
            f"  Debian/Ubuntu : sudo apt install -y python3-tk\n"
            f"  Fedora/RHEL   : sudo dnf install -y python3-tkinter\n"
            f"  Arch          : sudo pacman -S tk\n\n"
            f"  Sonra:  ./setup.sh --force\n\n")
        raise SystemExit(2)

    try:
        p = Panel(args)
    except tk.TclError as e:
        sys.stderr.write(
            f"\nHATA: pencere acilamadi -> {e}\n\n"
            f"  Panelin bir masaustu oturumuna ihtiyaci var.\n"
            f"  SSH ile baglandiysan:   ssh -X kullanici@makine\n"
            f"  Ortami gormek icin:     python3 panel.py --check\n\n")
        raise SystemExit(2)
    p.mainloop()


if __name__ == "__main__":
    main()
