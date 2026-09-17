"""Ring measurement on the near panel.

No global search. It was tried and falls short: only 1-2 of the 5 rings are
found, because the conveyor bar crosses every ring at the same brightness and
takes out about 40% of the circle. Loosening the filters lets the bar's own
curve in as false circles.

Three steps instead:
  1. seed    find a few rings globally (1-3 is enough)
  2. predict compute where the rest must be, from theta - coordinates only,
             the image is never resampled
  3. measure fit an ellipse in a small window around each prediction

Rings are equally spaced too: the undistorted step measured 472.5 / 483.7 /
477.3 px on three devices. Bright-panel step is ~248 px, so the ratio is ~1.93,
the depth ratio of the two panels.

If nothing is found nothing happens: stage 2 is skipped and theta from stage 1
stands.
"""
import numpy as np
import cv2

from distorch import geom as G

TOP_PAD, BOTTOM_PAD = 160, 260
TOPHAT_SE = 17          # keeps thin bright structures, drops the thick bar
HORIZ_SE = 35           # removes long horizontal structures
MASK_THR = 12

HOUGH = dict(dp=1, minDist=120, param1=70, param2=20, minRadius=40, maxRadius=95)
N_RAYS = 120
SEED_RESID, SEED_COVERAGE = 0.08, 65.0
MEASURE_RESID, MEASURE_COVERAGE = 0.10, 45.0   # looser: we know where to look
WINDOW = 85
PREDICT_TOL = 50
SAME_RING = 90

# Ring step / bright-panel hole step. Measured on 38 frames: 1.929 +- 0.022
# (1.1%). A mechanical constant of the chamber. Used to predict from a single
# seed, and to reject ring sets that do not produce it.
STEP_RATIO = 1.929
RATIO_MIN, RATIO_MAX = 1.85, 2.01


def _band(gray, filled_mask):
    """-> (tophat, mask, y0) with the bright panel excluded."""
    g = gray.astype(np.float32) if gray.dtype != np.float32 else gray
    ys, _ = np.nonzero(filled_mask)
    if len(ys) == 0:
        return None, None, 0
    bottom = int(ys.max())
    y0 = max(0, bottom - TOP_PAD)
    y1 = min(g.shape[0], bottom + BOTTOM_PAD)
    band = g[y0:y1, :].copy()
    band[cv2.dilate(filled_mask[y0:y1, :], np.ones((9, 9), np.uint8)) > 0] = 0
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TOPHAT_SE, TOPHAT_SE))
    th = cv2.GaussianBlur(cv2.morphologyEx(band, cv2.MORPH_TOPHAT, se), (0, 0), 1.0)
    m = (th > MASK_THR).astype(np.uint8)
    horiz = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (HORIZ_SE, 1)))
    m = cv2.morphologyEx(cv2.subtract(m, horiz), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return th, m, y0


def _angular_coverage(pts, cx, cy):
    a = np.degrees(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx))
    h, _ = np.histogram(a, bins=36, range=(-180, 180))
    return float((h > 0).sum() / 36 * 100)


def _ellipse(th, cx, cy, r0):
    """Rays from the centre, tophat peak on each = the bright rim.

    All rays are sampled in ONE remap call; one call per ray made the 106-frame
    run 19 s/frame.
    """
    rs = np.arange(0.6 * r0, 1.45 * r0, 0.4)
    a = np.linspace(0, 2 * np.pi, N_RAYS, endpoint=False)
    xs = cx + np.outer(np.cos(a), rs)
    ys = cy + np.outer(np.sin(a), rs)
    inside = (xs >= 1) & (xs < th.shape[1] - 1) & (ys >= 1) & (ys < th.shape[0] - 1)
    prof = cv2.remap(th, np.clip(xs, 0, th.shape[1] - 1).astype(np.float32),
                     np.clip(ys, 0, th.shape[0] - 1).astype(np.float32),
                     cv2.INTER_LINEAR)
    prof = np.where(inside, prof, -1.0)
    i = np.argmax(prof, axis=1)
    rows = np.arange(len(a))
    valid = (prof[rows, i] >= 10) & (inside.sum(1) >= 8)
    if valid.sum() < 30:
        return None
    ip = np.clip(i, 1, prof.shape[1] - 2)
    left, mid, right = prof[rows, ip - 1], prof[rows, ip], prof[rows, ip + 1]
    den = left - 2 * mid + right
    t = np.where(np.abs(den) < 1e-9, 0.0,
                 0.5 * (left - right) / np.where(np.abs(den) < 1e-9, 1.0, den))
    t = np.where(i == ip, t, 0.0)
    r = rs[i] + t * 0.4
    pts = np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])[valid].astype(np.float32)
    coverage = _angular_coverage(pts, float(pts[:, 0].mean()), float(pts[:, 1].mean()))
    n0 = len(pts)
    for _ in range(3):
        (ex, ey), (major, minor), ang = cv2.fitEllipse(pts)
        rad = _norm_radius(pts, ex, ey, major, minor, ang)
        keep = np.abs(rad - 1) < 0.05
        if keep.sum() < 25:
            break
        pts = pts[keep]
    (ex, ey), (major, minor), ang = cv2.fitEllipse(pts)
    rad = _norm_radius(pts, ex, ey, major, minor, ang)
    return dict(cx=float(ex), cy=float(ey), major=float(major), minor=float(minor),
                angle=float(ang), n=int(len(pts)), n0=int(n0),
                resid=float(np.std(rad)), coverage=float(coverage))


def _norm_radius(pts, ex, ey, major, minor, ang):
    t = np.radians(ang)
    ca, sa = np.cos(t), np.sin(t)
    dx, dy = pts[:, 0] - ex, pts[:, 1] - ey
    u = (dx * ca + dy * sa) / (minor / 2)
    v = (-dx * sa + dy * ca) / (major / 2)
    return np.sqrt(u * u + v * v)


def _dedupe(found):
    found = sorted(found, key=lambda h: h["cx"])
    out = []
    for h in found:
        if out and abs(h["cx"] - out[-1]["cx"]) < SAME_RING:
            if h["resid"] < out[-1]["resid"]:
                out[-1] = h
            continue
        out.append(h)
    return out


def seeds(gray, filled_mask):
    """Global scan for a few rings. One to three is enough."""
    th, _, y0 = _band(gray, filled_mask)
    if th is None:
        return []
    t8 = np.clip(th * 4, 0, 255).astype(np.uint8)
    circles = cv2.HoughCircles(t8, cv2.HOUGH_GRADIENT, **HOUGH)
    if circles is None:
        return []
    out = []
    for x, y, r in circles[0]:
        e = _ellipse(th, float(x), float(y), float(r))
        if e is None or e["coverage"] < SEED_COVERAGE or e["resid"] > SEED_RESID:
            continue
        e["cy"] += y0
        e["source"] = "seed"
        out.append(e)
    return _dedupe(out)


def predict(theta, found, n_side=4, size=G.CANON_SIZE, hole_row=None):
    """Where the remaining rings must sit in the raw frame.

    With a single seed the step comes from the bright panel's hole step times
    STEP_RATIO. That only makes detection easier: every candidate still has to
    pass the ellipse test on its own.
    """
    if len(found) == 0:
        return []
    p = np.array([[h["cx"], h["cy"]] for h in found], float)
    u = G.undistort_points(p, theta)
    if len(found) == 1:
        if hole_row is None or len(hole_row) < 3:
            return []
        uh = G.undistort_points(np.asarray(hole_row, float), theta)
        direction = uh[-1] - uh[0]
        direction = direction / (np.linalg.norm(direction) + 1e-9)
        step = np.linalg.norm(uh[-1] - uh[0]) / (len(uh) - 1) * STEP_RATIO
        d = direction * step
        sx, sy = (d[0], u[0, 0]), (d[1], u[0, 1])
    else:
        i = np.arange(len(u))
        a = np.column_stack([i, np.ones(len(i))])
        sx, *_ = np.linalg.lstsq(a, u[:, 0], rcond=None)
        sy, *_ = np.linalg.lstsq(a, u[:, 1], rcond=None)
    w, h = size
    out = []
    for k in range(-n_side, len(u) + n_side):
        q = G.distort_points(np.array([[sx[0] * k + sx[1], sy[0] * k + sy[1]]]), theta)[0]
        if WINDOW < q[0] < w - WINDOW and WINDOW < q[1] < h - WINDOW:
            out.append((float(q[0]), float(q[1])))
    return out


def measure(gray, filled_mask, predictions, window=WINDOW):
    """Fit an ellipse in a small window around each prediction, on RAW pixels."""
    g = gray.astype(np.float32) if gray.dtype != np.float32 else gray
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TOPHAT_SE, TOPHAT_SE))
    out = []
    for px, py in predictions:
        x0, y0 = int(px - window), int(py - window)
        x1, y1 = int(px + window), int(py + window)
        if x0 < 0 or y0 < 0 or x1 > g.shape[1] or y1 > g.shape[0]:
            continue
        w = g[y0:y1, x0:x1].copy()
        w[cv2.dilate(filled_mask[y0:y1, x0:x1], np.ones((9, 9), np.uint8)) > 0] = 0
        th = cv2.GaussianBlur(cv2.morphologyEx(w, cv2.MORPH_TOPHAT, se), (0, 0), 1.0)
        best = None
        for r0 in (48, 56, 64, 72):
            e = _ellipse(th, float(window), float(window), float(r0))
            if e is None or e["resid"] > MEASURE_RESID or e["coverage"] < MEASURE_COVERAGE:
                continue
            if best is None or e["resid"] < best["resid"]:
                best = e
        if best is None:
            continue
        best["cx"] += x0
        best["cy"] += y0
        if np.hypot(best["cx"] - px, best["cy"] - py) > PREDICT_TOL:
            continue
        best["source"] = "predicted"
        out.append(best)
    return _dedupe(out)


def check_ratio(found, hole_row, theta):
    """Does ring step / hole step come out at the expected value?"""
    if len(found) < 3 or hole_row is None or len(hole_row) < 3:
        return None, False
    ur = G.undistort_points(to_array(found), theta)
    uh = G.undistort_points(np.asarray(hole_row, float), theta)
    ratio = float((np.linalg.norm(ur[-1] - ur[0]) / (len(ur) - 1)) /
                  (np.linalg.norm(uh[-1] - uh[0]) / (len(uh) - 1)))
    return ratio, (RATIO_MIN <= ratio <= RATIO_MAX)


def find(gray, filled_mask, theta, hole_row=None, rounds=2):
    """-> (rings, info). Rings are dropped when the ratio check fails: better to
    skip stage 2 than to corrupt theta with a false detection."""
    found = seeds(gray, filled_mask)
    info = {"n_seeds": len(found), "ratio": None, "ratio_ok": False}
    if not found:
        return [], info
    for _ in range(rounds):
        fresh = measure(gray, filled_mask, predict(theta, found, hole_row=hole_row))
        merged = _dedupe(found + fresh)
        if len(merged) <= len(found):
            found = merged
            break
        found = merged
    ratio, ok = check_ratio(found, hole_row, theta)
    info["ratio"], info["ratio_ok"] = ratio, ok
    if len(found) >= 3 and not ok:
        return [], info
    return found, info


def to_array(rings):
    return np.array([[h["cx"], h["cy"]] for h in rings], float)
