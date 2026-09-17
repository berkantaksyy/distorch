"""distort_v2 inference - the GUARD role.

This module does not produce the calibration result. It has three jobs:
  1. starting value for the solver
  2. guard: it looks at the whole frame, independently of the holes
  3. narrowing the ring prediction window

The input contract is fixed; the weights are locked to this chain:
    raw -> grey -> 960x540 INTER_AREA -> 512x288 INTER_AREA
        -> /255 -> replicate to 3 channels -> ImageNet normalise
The 960x540 step is required: training saw 960x540 frames, inference gets
1920x1080, and without it the resampling signature differs. 512x288 is exactly
16:9; an anisotropic scale would break radial symmetry.

Architecture: ResNet-18 body, NO pooling (pooling destroys position, so cx/cy
cannot be learned), NO cropping (the corners carry the centre information).
Coordinate channels [x, y, r^2] are concatenated at feature level, not at input.
"""
import json
from pathlib import Path

import numpy as np
import cv2

from distorch import geom as G, NET_WEIGHTS, NET_META, NET_BIAS

INPUT = (512, 288)
MID = (960, 540)
IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)
N_OUT = 4

_model = None
_meta = {}
_bias = None
_status = "not loaded"


def has_torch():
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def preprocess(frame):
    import torch
    img = frame
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if (img.shape[1], img.shape[0]) != MID:
        img = cv2.resize(img, MID, interpolation=cv2.INTER_AREA)
    img = cv2.resize(img, INPUT, interpolation=cv2.INTER_AREA)
    g = img.astype(np.float32) / 255.0
    rgb = (np.stack([g, g, g], -1) - IMNET_MEAN) / IMNET_STD
    return torch.from_numpy(rgb.transpose(2, 0, 1).copy())


def _build():
    import torch
    import torch.nn as nn
    import torchvision

    class DistortNet(nn.Module):
        def __init__(self, hidden=256, ch=64, p_drop=0.1):
            super().__init__()
            m = torchvision.models.resnet18(weights=None)
            self.body = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool,
                                      m.layer1, m.layer2, m.layer3, m.layer4)
            fw, fh = INPUT[0] // 32, INPUT[1] // 32
            ys = torch.linspace(-1, 1, fh).view(fh, 1).expand(fh, fw)
            xs = torch.linspace(-1, 1, fw).view(1, fw).expand(fh, fw)
            self.register_buffer(
                "coords", torch.stack([xs, ys, xs * xs + ys * ys], 0).unsqueeze(0),
                persistent=False)
            self.reduce = nn.Sequential(nn.Conv2d(512 + 3, ch, 1),
                                        nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
            self.head = nn.Sequential(nn.Linear(ch * fh * fw, hidden),
                                      nn.ReLU(inplace=True), nn.Dropout(p_drop),
                                      nn.Linear(hidden, N_OUT))

        def forward(self, x):
            z = self.body(x)
            z = self.reduce(torch.cat([z, self.coords.expand(z.shape[0], -1, -1, -1)], 1))
            return self.head(z.flatten(1))

    return DistortNet()


def load(path=None, meta_path=None):
    """-> (ok, status). Never raises: missing torch or weights returns False."""
    global _model, _meta, _status
    if _model is not None:
        return True, _status
    if not has_torch():
        _status = "no torch - guard disabled, geometric solve still runs"
        return False, _status
    import torch
    p = Path(path or NET_WEIGHTS)
    mp = Path(meta_path or NET_META)
    if not p.exists():
        _status = f"weights missing: {p.name}"
        return False, _status
    try:
        _meta = json.loads(mp.read_text()) if mp.exists() else {}
        got = tuple(_meta.get("input_size", INPUT))
        if got != tuple(INPUT):
            raise ValueError(f"input contract mismatch: meta {got} != code {INPUT}")
        m = _build()
        m.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
        m.eval()
        _model = m
        _status = f"{p.name} (cpu)"
        return True, _status
    except Exception as e:
        _model = None
        _status = f"load failed: {type(e).__name__}: {e}"
        return False, _status


def meta():
    return dict(_meta)


def status():
    return _status


def bias():
    """Systematic offset against the geometric solution.

    The CNN is trained on one reference calibration, so its difference from the
    geometric solve is systematic, not random, and the guard threshold only
    means anything once it is removed. distort_v2 over 104 frames (none of
    which it saw in training):
        raw difference        median 64.8 px
        offset                m1 -0.019  m2 +0.218  cx +5.0 px  cy +7.7 px
        in-pipeline gap       median 3.12 px, MAD 1.06 px  -> threshold 6.3
    """
    global _bias
    if _bias is None:
        try:
            _bias = json.loads(Path(NET_BIAS).read_text())["sapma_medyan"]
        except Exception:
            _bias = {"m1": 0.0, "m2": 0.0, "cx": 0.0, "cy": 0.0}
    return _bias


def theta_from_frame(frame, correct_bias=True):
    """-> theta, or None when the model is unavailable (callers must handle it).

    correct_bias=False returns the raw output, used by the reproduction test.
    """
    ok, _ = load()
    if not ok:
        return None
    import torch
    x = preprocess(frame).unsqueeze(0)
    with torch.no_grad():
        out = _model(x)[0].cpu().numpy().astype(np.float64)
    norm = _meta.get("norm")
    if norm:
        out = out * np.asarray(norm["std"]) + np.asarray(norm["mean"])
    m1, m2, cx, cy = (float(v) for v in out)
    m1_raw, m2_raw = m1, m2          # envelope_ok needs the untouched output
    if correct_bias:
        b = bias()
        m1 -= b["m1"]; m2 -= b["m2"]; cx -= b["cx"]; cy -= b["cy"]
    k1, k2 = G.kk_from_mags(m1, m2, tuple(_meta.get("anchors", G.ANCHORS)))
    return {"k1": k1, "k2": k2, "cx": cx, "cy": cy, "f": G.CANON_F,
            "m1": m1, "m2": m2, "m1_raw": m1_raw, "m2_raw": m2_raw,
            "bias_corrected": bool(correct_bias)}


def envelope_ok(theta):
    """Is the prediction inside the training envelope? -> (ok, note).

    Tested on the RAW output, not the bias-corrected one: the envelope describes
    the space the model was trained to emit. Correcting first shifted every
    frame outside it (v2 reported 0/104 in-envelope before this was fixed).

    Not clamped: clamping would make the model look safe but remove its ability
    to say "I do not know".
    """
    env = _meta.get("envelope")
    if not env or "m1" not in theta:
        return True, ""
    lo, hi = env.get("s", (0.0, 1.05))
    m1r, m2r = env.get("m_ref", (1.1267, 1.4510))
    s1 = (theta.get("m1_raw", theta["m1"]) - 1.0) / max(m1r - 1.0, 1e-9)
    s2 = (theta.get("m2_raw", theta["m2"]) - 1.0) / max(m2r - 1.0, 1e-9)
    pad = 0.10
    if (lo - pad <= s1 <= hi + pad) and (lo - pad <= s2 <= hi + pad):
        return True, ""
    return False, (f"outside envelope: s1={s1:.2f} s2={s2:.2f} "
                   f"(envelope [{lo:.2f},{hi:.2f}] +-{pad}) - not trained here")
