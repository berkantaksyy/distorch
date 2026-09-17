#!/usr/bin/env python3
"""Test paneli - USB kamera, distorsiyon duzeltme, kesme, YOLO. Servis degil.

Tkinter penceresi. Linux/V4L2. macOS'a ozgu hicbir sey yok.
Kendi kendine yeter: distorch paketini IMPORT ETMEZ, hicbir dosyani degistirmez.
Matematik (Brown k1,k2 + capa donusumu) burada kopya olarak duruyor.

    python3 panel.py
    python3 panel.py --device 2 --out ./cikti
"""
import argparse
import glob
import json
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

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


def build_maps(th, size, roll_deg=0.0):
    """Tek remap: duzeltme (+ istege bagli duzlestirme). Kare bir kez ornekleniyor."""
    w, h = size
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float64)
    p = np.stack([gx.ravel(), gy.ravel()], 1)
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
        import torch, torchvision
        import torch.nn as nn
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

    def open(self):
        idx = self.device
        if isinstance(idx, str) and idx.startswith("/dev/video"):
            idx = int("".join(c for c in idx if c.isdigit()))
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            self.err = f"acilamadi: {self.device} (v4l2)"
            return False
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, f = cap.read()
        if not ok:
            cap.release()
            self.err = "kare okunamadi - format/cozunurluk desteklenmiyor olabilir"
            return False
        self.real = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.cap = cap
        self.frame = f
        self.run = True
        self.err = ""
        threading.Thread(target=self._loop, daemon=True).start()
        return True

    def _loop(self):
        t0, n = time.time(), 0
        while self.run:
            ok, f = self.cap.read()
            if not ok:
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
        self.run = False
        time.sleep(0.08)
        if self.cap:
            self.cap.release()
        self.cap = None


# --------------------------------------------------------------- YOLO

class Yolo:
    def __init__(self, weights):
        from ultralytics import YOLO as _Y
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

class Panel(tk.Tk):
    PREVIEW_W = 1040

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
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)

        self.v_dev = tk.StringVar(value=args.device or list_devices()[0])
        self.v_w = tk.IntVar(value=args.width)
        self.v_h = tk.IntVar(value=args.height)
        self.v_fps = tk.IntVar(value=args.fps)
        self.v_fourcc = tk.StringVar(value=args.fourcc)

        self.v_mode = tk.StringVar(value="kapali")     # kapali | profil | model
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
        self.v_saveraw = tk.BooleanVar(value=True)

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
        ttk.Label(left, textvariable=self.status).pack(anchor="w", pady=(6, 0))

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
        for t, v in (("kapali (ham)", "kapali"), ("profil JSON", "profil"),
                     ("distorch modeli (.pt)", "model")):
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

        # kesme
        c = box("Kesme  (duzeltmeden SONRA, % olarak)")
        for k in ("ust", "alt", "sol", "sag"):
            r = ttk.Frame(c); r.pack(fill="x")
            ttk.Label(r, text=k, width=5).pack(side="left")
            ttk.Scale(r, from_=0, to=40, variable=self.v_cut[k],
                      orient="horizontal").pack(side="left", fill="x", expand=True)
            ttk.Label(r, textvariable=self.v_cut[k], width=5).pack(side="left")
        ttk.Button(c, text="sifirla",
                   command=lambda: [v.set(0.0) for v in self.v_cut.values()]).pack(anchor="e")

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
        ttk.Scale(r, from_=0.01, to=0.9, variable=self.v_conf,
                  orient="horizontal").pack(side="left", fill="x", expand=True)
        ttk.Label(r, textvariable=self.v_conf, width=5).pack(side="left")
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
        ttk.Checkbutton(c, text="ham kareyi de kaydet", variable=self.v_saveraw).pack(anchor="w")
        ttk.Button(c, text="KARE AL", command=self._shoot).pack(fill="x", pady=3)
        ttk.Button(c, text="cikti klasorunu sec", command=self._pick_out).pack(fill="x")
        self.lbl_out = ttk.Label(c, text=str(self.out), wraplength=310, foreground="#555")
        self.lbl_out.pack(anchor="w")

    def _pick(self, var, types, reset_yolo=False, want=None):
        p = filedialog.askopenfilename(filetypes=types + [("hepsi", "*.*")])
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

    def _pick_out(self):
        p = filedialog.askdirectory()
        if p:
            self.out = Path(p)
            self.out.mkdir(parents=True, exist_ok=True)
            self.lbl_out.configure(text=str(self.out))

    def _reset_theta(self):
        self.theta = None
        self.net = None
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
        c = Camera(self.v_dev.get(), self.v_w.get(), self.v_h.get(),
                   self.v_fps.get(), self.v_fourcc.get())
        if not c.open():
            messagebox.showerror("kamera", c.err)
            return
        self.cam = c
        self.btn_cam.configure(text="KAMERAYI KAPAT")
        if c.real != (self.v_w.get(), self.v_h.get()):
            self.status.set(f"DIKKAT: istenen {self.v_w.get()}x{self.v_h.get()}, "
                            f"alinan {c.real[0]}x{c.real[1]}")

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
        if self.net is None and self.v_wts.get():
            try:
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
        try:
            roll = float(self.v_rolltxt.get()) if self.v_level.get() else 0.0
        except ValueError:
            roll = 0.0
        key = (round(t["k1"], 6), round(t["k2"], 6), round(t["cx"], 2),
               round(t["cy"], 2), w, h, round(roll, 4))
        if self.maps is None or self.maps_key != key:
            self.maps = build_maps(t, (w, h), roll)
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

    def detect(self, img):
        if not self.v_yolo_on.get() or not self.v_yolo_w.get():
            return img, []
        if self.yolo is None:
            try:
                self.yolo = Yolo(self.v_yolo_w.get())
                self.status.set(f"yolo: {self.yolo.name} ({self.yolo.task})")
            except Exception as e:
                self.status.set(f"yolo yuklenemedi: {e}")
                self.v_yolo_on.set(False)
                return img, []
        try:
            det = self.yolo.run(img, self.v_conf.get(), self.v_retina.get())
        except Exception as e:
            self.status.set(f"yolo hatasi: {e}")
            return img, []
        try:
            mm = float(self.v_mmpx.get()) if self.v_mmpx.get().strip() else None
        except ValueError:
            mm = None
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
        return vis, det

    # ---------------------------------------------------------- dongu
    def _tick(self):
        if self.cam:
            f = self.cam.grab()
            if f is not None:
                try:
                    cor, th = self.correct(f)
                    img = self.cut(cor)
                    img, det = self.detect(img)
                    self._show(img)
                    s = (f"{self.cam.real[0]}x{self.cam.real[1]} @ "
                         f"{self.cam.measured_fps:.1f} fps   kesim sonrasi "
                         f"{img.shape[1]}x{img.shape[0]}")
                    if th:
                        s += f"   k1 {th['k1']:+.4f} k2 {th['k2']:+.4f} cx {th['cx']:.0f} cy {th['cy']:.0f}"
                    if det:
                        s += f"   {len(det)} tespit"
                    self.status.set(s)
                except Exception as e:
                    self.status.set(f"hata: {type(e).__name__}: {e}")
        self.after(40, self._tick)

    def _show(self, img):
        h, w = img.shape[:2]
        s = min(self.PREVIEW_W / w, 640.0 / h, 1.0)
        v = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        im = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(v, cv2.COLOR_BGR2RGB)))
        self.canvas.configure(image=im)
        self.canvas.image = im

    # ---------------------------------------------------------- cekim
    def _shoot(self):
        if not self.cam:
            messagebox.showwarning("kamera", "once kamerayi ac")
            return
        self.shot += 1
        tag = self.v_tag.get().strip() or "test"
        stamp = time.strftime("%Y%m%d_%H%M%S")
        d = self.out / f"{tag}_{stamp}"
        d.mkdir(parents=True, exist_ok=True)
        meta = {"etiket": tag, "zaman": stamp, "mod": self.v_mode.get(),
                "kesme": {k: v.get() for k, v in self.v_cut.items()},
                "kamera": {"aygit": self.v_dev.get(), "format": self.v_fourcc.get(),
                           "cozunurluk": list(self.cam.real)},
                "kareler": []}
        for i in range(self.v_n.get()):
            f = self.cam.grab()
            if f is None:
                continue
            cor, th = self.correct(f)
            img = self.cut(cor)
            vis, det = self.detect(img)
            if self.v_saveraw.get():
                cv2.imwrite(str(d / f"{i+1:02d}_ham.png"), f)
            cv2.imwrite(str(d / f"{i+1:02d}_duzeltilmis.png"), img)
            if det:
                cv2.imwrite(str(d / f"{i+1:02d}_tespit.jpg"), vis,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
            meta["kareler"].append({
                "no": i + 1,
                "theta": None if th is None else {k: round(th[k], 6)
                                                  for k in ("k1", "k2", "cx", "cy")},
                "tespit": [{k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in x.items()
                            if k in ("name", "conf", "en_boy", "uzun_mm", "kisa_mm")}
                           for x in det]})
            if self.v_gap.get():
                self.update()
                time.sleep(self.v_gap.get() / 1000.0)
        (d / "kayit.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        self.status.set(f"{len(meta['kareler'])} kare -> {d}")

    def _quit(self):
        if self.cam:
            self.cam.close()
        self.destroy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--fourcc", default="YUYV")
    ap.add_argument("--shots", type=int, default=3)
    ap.add_argument("--gap-ms", type=int, default=150)
    ap.add_argument("--out", default="./cikti")
    Panel(ap.parse_args()).mainloop()


if __name__ == "__main__":
    main()
