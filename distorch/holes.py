"""Hole detector for the backlit panel.

Finds the bottom row of N_HOLES holes and measures their centres to sub-pixel
accuracy. Every step below was needed on the 115-frame field set; the order
matters.
"""
import numpy as np
import cv2

from distorch import N_HOLES, HOLE_SPACING_MM

# Thresholding. A fixed threshold does not work: the panel median wanders
# between 199 and 247 across the fleet (exposure, gain, gamma).
THR_RATIO = 0.86
THR_MIN, THR_MAX = 110.0, 215.0

# Candidate blobs.
AREA_MIN, AREA_MAX = 25, 5000
ASPECT_MIN, ASPECT_MAX = 0.4, 2.5
FILL_MIN = 0.22
MERGE_DIST = 30            # split hole fragments closer than this are merged
AREA_RATIO_MIN, AREA_RATIO_MAX = 0.3, 3.0

# Row selection. The row is curved by distortion, so a line does not fit.
ROW_TOL = 60
RANSAC_ITERS, RANSAC_THR = 200, 8.0
NEIGHBOUR_MIN = 100        # of two candidates closer than this, keep the larger

GAP_RATIO = 1.45           # a gap this much above median means a hole is missing
MISSING_CONTRAST = 8.0     # grey levels needed to accept a recovered hole
N_RAYS = 48


def panel_mask(gray):
    """-> (panel, filled, threshold). filled = panel plus its interior holes."""
    g = gray.astype(np.float32) if gray.dtype != np.float32 else gray
    t0, _ = cv2.threshold(g.astype(np.uint8), 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lab, st, _ = cv2.connectedComponentsWithStats((g > t0).astype(np.uint8), 8)
    if n < 2:
        z = np.zeros(g.shape, np.uint8)
        return z, z, float(t0)
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    thr = float(np.clip(THR_RATIO * float(np.median(g[lab == k])), THR_MIN, THR_MAX))

    b = cv2.morphologyEx((g > thr).astype(np.uint8), cv2.MORPH_CLOSE,
                         np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(b, 8)
    if n < 2:
        z = np.zeros(g.shape, np.uint8)
        return z, z, thr
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    panel = (lab == k).astype(np.uint8)

    h, w = g.shape
    ff = panel.copy()
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    return panel, (panel | (1 - ff)).astype(np.uint8), thr


def _candidates(g, panel, filled):
    inner = (filled & (1 - panel)).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(inner, 8)
    raw = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < AREA_MIN or a > AREA_MAX or w < 4 or h < 4:
            continue
        if not (ASPECT_MIN <= w / h <= ASPECT_MAX) or a / (w * h) < FILL_MIN:
            continue
        raw.append((cen[i][0], cen[i][1], float(a)))
    if len(raw) < 3:
        return np.zeros((0, 3))
    arr = np.array(raw, float)

    used = np.zeros(len(arr), bool)
    merged = []
    for i in range(len(arr)):
        if used[i]:
            continue
        d = np.hypot(arr[:, 0] - arr[i, 0], arr[:, 1] - arr[i, 1])
        grp = np.where((d < MERGE_DIST) & ~used)[0]
        used[grp] = True
        w = arr[grp, 2]
        merged.append(((arr[grp, 0] * w).sum() / w.sum(),
                       (arr[grp, 1] * w).sum() / w.sum(), w.sum()))
    arr = np.array(merged)

    # Holes are similar in size. Without this one frame produced 16 false
    # candidates.
    if len(arr) >= 5:
        am = np.median(np.sort(arr[:, 2])[-N_HOLES:])
        arr = arr[(arr[:, 2] > AREA_RATIO_MIN * am) & (arr[:, 2] < AREA_RATIO_MAX * am)]
    return arr


def _select_row(arr):
    """Densest y cluster, then quadratic RANSAC (the row is curved)."""
    if len(arr) < 4:
        return arr
    ys = arr[:, 1]
    best = None
    for y0 in ys:
        s = np.abs(ys - y0) < ROW_TOL
        if best is None or s.sum() > best.sum():
            best = s
    idx = np.where(best)[0]
    if len(idx) < 4:
        return arr[idx]
    x, y = arr[idx, 0], arr[idx, 1]
    rng = np.random.default_rng(0)
    inliers = None
    for _ in range(RANSAC_ITERS):
        s = rng.choice(len(idx), 3, replace=False)
        try:
            c = np.polyfit(x[s], y[s], 2)
        except Exception:
            continue
        inl = np.abs(np.polyval(c, x) - y) < RANSAC_THR
        if inliers is None or inl.sum() > inliers.sum():
            inliers = inl
    row = arr[idx[inliers]]
    row = row[np.argsort(row[:, 0])]

    keep = np.ones(len(row), bool)
    for i in range(len(row) - 1):
        if row[i + 1, 0] - row[i, 0] < NEIGHBOUR_MIN:
            keep[i if row[i, 2] < row[i + 1, 2] else i + 1] = False
    return row[keep]


def _darkest_in_window(g, panel, dist_tf, px, py, r):
    """Darkest point near (px, py), kept away from the panel edge."""
    y0, y1 = int(py - 1.5 * r), int(py + 1.5 * r) + 1
    x0, x1 = int(px - 3 * r), int(px + 3 * r) + 1
    if x0 < 0 or y0 < 0 or x1 > g.shape[1] or y1 > g.shape[0]:
        return None
    win, pm, dm = g[y0:y1, x0:x1], panel[y0:y1, x0:x1], dist_tf[y0:y1, x0:x1]
    if pm.sum() < 50:
        return None
    bg = float(np.median(win[pm == 1]))
    smooth = cv2.GaussianBlur(win, (0, 0), 0.5 * r)
    score = (bg - smooth) * (dm > 0.8 * r)
    yy, xx = np.unravel_index(int(np.argmax(score)), score.shape)
    return float(score[yy, xx]), x0 + xx, y0 + yy


def _recover_missing(g, panel, filled, xs, ys, r):
    """Fill gaps and, if still short, extend the ends.

    The middle hole sits right in front of an LED and can drop to 9 grey levels
    of contrast, so it is regularly the one that goes missing.
    """
    dist_tf = cv2.distanceTransform(filled, cv2.DIST_L2, 5)
    xs, ys = list(xs), list(ys)
    if len(xs) >= 4:
        curve = np.polyfit(xs, ys, 2)
        med = float(np.median(np.diff(xs)))
        for i, gap in enumerate(np.diff(xs)):
            if gap > GAP_RATIO * med:
                mx = 0.5 * (xs[i] + xs[i + 1])
                hit = _darkest_in_window(g, panel, dist_tf, mx,
                                         float(np.polyval(curve, mx)), r)
                if hit and hit[0] > MISSING_CONTRAST:
                    xs.append(hit[1]); ys.append(hit[2])
        o = np.argsort(xs); xs = [xs[i] for i in o]; ys = [ys[i] for i in o]

    if len(xs) < N_HOLES and len(xs) >= 4:
        gaps = np.diff(xs)
        curve = np.polyfit(xs, ys, 2)
        found = [h for h in (_darkest_in_window(g, panel, dist_tf, px,
                                                float(np.polyval(curve, px)), r)
                             for px in (xs[0] - gaps[-1], xs[-1] + gaps[0])) if h]
        if found:
            best = max(found)
            if best[0] > MISSING_CONTRAST:
                xs.append(best[1]); ys.append(best[2])
                o = np.argsort(xs); xs = [xs[i] for i in o]; ys = [ys[i] for i in o]
    return xs, ys


def _weighted_centroid(g, filled, panel, cx, cy, r):
    """(bg - I) weighted centroid. Call twice so the window ends up centred."""
    y0, y1 = int(max(0, cy - 2 * r)), int(min(g.shape[0], cy + 2 * r + 1))
    x0, x1 = int(max(0, cx - 2 * r)), int(min(g.shape[1], cx + 2 * r + 1))
    win, fm, pm = g[y0:y1, x0:x1], filled[y0:y1, x0:x1], panel[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    disk = d2 <= (1.7 * r) ** 2
    ring = (pm == 1) & disk & (d2 >= (1.25 * r) ** 2)
    if ring.sum() > 15:
        bg = float(np.median(win[ring]))
    elif (pm == 1).sum():
        bg = float(np.median(win[pm == 1]))
    else:
        bg = 230.0
    w = np.clip(bg - win, 0, None) * (fm == 1) * disk
    m = w.sum()
    if m < 1:
        return float(cx), float(cy), bg
    return float((w * xx).sum() / m), float((w * yy).sum() / m), bg


def _edge_ellipse(g, cx, cy, r, bg):
    """Half-level crossing along rays, then an ellipse through those points."""
    pts = []
    for a in np.linspace(0, 2 * np.pi, N_RAYS, endpoint=False):
        rs = np.arange(0.2 * r, 2.0 * r, 0.25)
        xs, ys = cx + rs * np.cos(a), cy + rs * np.sin(a)
        ok = (xs >= 1) & (xs < g.shape[1] - 1) & (ys >= 1) & (ys < g.shape[0] - 1)
        if ok.sum() < 8:
            continue
        xs, ys, rr = xs[ok], ys[ok], rs[ok]
        prof = cv2.remap(g, xs.astype(np.float32)[None], ys.astype(np.float32)[None],
                         cv2.INTER_LINEAR)[0]
        inner = float(prof[:max(3, len(prof) // 5)].mean())
        half = 0.5 * (bg + inner)
        idx = np.where((prof[:-1] < half) & (prof[1:] >= half))[0]
        if len(idx) == 0:
            continue
        i = int(idx[0])
        t = (half - prof[i]) / (prof[i + 1] - prof[i] + 1e-9)
        rad = rr[i] + t * (rr[i + 1] - rr[i])
        pts.append((cx + rad * np.cos(a), cy + rad * np.sin(a)))
    if len(pts) < 16:
        return None
    (ex, ey), (ma_, mi_), ang = cv2.fitEllipse(np.array(pts, np.float32))
    return float(ex), float(ey), float(ma_), float(mi_), float(ang), len(pts)


def find(gray, n_expected=N_HOLES):
    """Detect holes, left to right. -> list of dicts."""
    g = gray.astype(np.float32) if gray.dtype != np.float32 else gray
    panel, filled, thr = panel_mask(g)
    arr = _candidates(g, panel, filled)
    if len(arr) < 3:
        return []
    row = _select_row(arr)
    if len(row) < 3:
        return []
    r = float(np.clip(np.sqrt(np.median(row[:, 2]) / np.pi), 8, 30))
    xs, ys = _recover_missing(g, panel, filled, row[:, 0], row[:, 1], r)

    out = []
    for cx, cy in zip(xs, ys):
        cx, cy, bg = _weighted_centroid(g, filled, panel, cx, cy, r)
        cx, cy, bg = _weighted_centroid(g, filled, panel, cx, cy, r)
        el = _edge_ellipse(g, cx, cy, r, bg)
        out.append(dict(cx=cx, cy=cy, bg=bg, r=r, thr=thr,
                        cx_el=el[0] if el else None, cy_el=el[1] if el else None,
                        major=el[2] if el else None, minor=el[3] if el else None,
                        angle=el[4] if el else None, n_edge=el[5] if el else 0))
    return out


def centres(holes, method="ellipse"):
    """Centres to feed the solver, (N,2).

    Ellipse by default: both methods were solved side by side on 106 frames.
        method     line rms  spread%  rms_spacing  corner px  max resid
        centroid      0.488    1.031      0.526      4.69       1.49
        ellipse       0.479    0.714      0.355      4.71       1.12
    Falls back to the centroid where no ellipse fits (never happened).
    """
    out = []
    for h in holes:
        if method == "ellipse" and h.get("cx_el") is not None:
            out.append([h["cx_el"], h["cy_el"]])
        else:
            out.append([h["cx"], h["cy"]])
    return np.array(out, float)


def quality(holes):
    """Internal consistency of a hole set."""
    if len(holes) < 2:
        return dict(n=len(holes), ok=False, spread_pct=None, y_span=None,
                    symmetry_px=None, step_px=None)
    x = np.array([h["cx"] for h in holes])
    y = np.array([h["cy"] for h in holes])
    gaps = np.diff(x)
    sym = None
    if len(gaps) >= 2:
        n = len(gaps) // 2
        sym = float(np.abs(gaps[:n] - gaps[::-1][:n]).sum())
    return dict(n=len(holes),
                spread_pct=float((gaps.max() - gaps.min()) / gaps.mean() * 100),
                y_span=float(y.max() - y.min()), symmetry_px=sym,
                step_px=float(gaps.mean()), ok=len(holes) >= 3)


def is_undistorted(holes, std_thr=6.0):
    """Already-corrected frame? Raw gap std is ~30 px, corrected is < 6."""
    if len(holes) < 4:
        return False
    return float(np.std(np.diff([h["cx"] for h in holes]))) < std_thr
