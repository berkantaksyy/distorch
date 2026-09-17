"""Sub-pixel panel edge extraction.

The edges give the plumb-line constraint: what is straight in the world must be
straight after correction.
"""
import numpy as np
import cv2

STEP = 6            # sample every Nth column / row
WINDOW = 14         # search +/- this many px around the mask boundary
BLUR_SIGMA = 1.2
SLOPE_MAX = 0.6     # above this the segment is a corner or a sharp bend
TRIM = 25           # points dropped from each end of the kept segment
MIN_POINTS = 8      # below this a line is not fed to the solver


def _subpixel(profile, i):
    """Parabolic peak interpolation."""
    if i <= 0 or i >= len(profile) - 1:
        return float(i)
    a, b, c = float(profile[i - 1]), float(profile[i]), float(profile[i + 1])
    d = a - 2 * b + c
    return float(i) + (0.0 if abs(d) < 1e-9 else 0.5 * (a - c) / d)


def _boundary_points(g, mask, axis, step=STEP, window=WINDOW):
    """axis 0: top/bottom boundary per column. axis 1: left/right per row."""
    gs = cv2.GaussianBlur(g, (0, 0), BLUR_SIGMA)
    h, w = g.shape
    first, last = [], []
    for u in (range(0, w, step) if axis == 0 else range(0, h, step)):
        line = mask[:, u] if axis == 0 else mask[u, :]
        idx = np.flatnonzero(line)
        if len(idx) < 40:
            continue
        for s, acc in ((idx[0], first), (idx[-1], last)):
            a = max(0, s - window)
            b = min(len(line) - 1, s + window)
            seg = gs[a:b, u] if axis == 0 else gs[u, a:b]
            if len(seg) < 5:
                continue
            grad = np.abs(np.gradient(seg.astype(np.float64)))
            v = a + _subpixel(grad, int(np.argmax(grad)))
            acc.append((u, v) if axis == 0 else (v, u))
    return np.array(first, float), np.array(last, float)


def straight_segment(pts, axis, slope_max=SLOPE_MAX, trim=TRIM):
    """Keep only the longest genuinely straight run.

    Without this the solve collapses: corners in the fit take the line RMS from
    0.21 px to 34 px (measured).
    """
    if len(pts) < 10:
        return pts
    order = np.argsort(pts[:, 0] if axis == 0 else pts[:, 1])
    pts = pts[order]
    u = pts[:, 0] if axis == 0 else pts[:, 1]
    v = pts[:, 1] if axis == 0 else pts[:, 0]
    if len(np.unique(u)) < 3:
        return pts
    ok = np.abs(np.gradient(v, u)) < slope_max
    best, i = (0, 0), 0
    while i < len(ok):
        if ok[i]:
            j = i
            while j < len(ok) and ok[j]:
                j += 1
            if j - i > best[1] - best[0]:
                best = (i, j)
            i = j
        else:
            i += 1
    a, b = best
    pts = pts[a:b]
    return pts[trim:-trim] if len(pts) > 2 * trim else pts


def extract(gray, filled_mask):
    """-> {"top","bottom","left","right"} sub-pixel point arrays.

    The mask must be the FILLED one (second output of holes.panel_mask). With
    holes left open the column scan latches onto a hole edge instead of the
    panel edge and the bottom line drops from 99 points to 1.
    """
    g = gray.astype(np.float32) if gray.dtype != np.float32 else gray
    top, bottom = _boundary_points(g, filled_mask, 0)
    left, right = _boundary_points(g, filled_mask, 1)
    return {"top": straight_segment(top, 0), "bottom": straight_segment(bottom, 0),
            "left": straight_segment(left, 1), "right": straight_segment(right, 1)}


def upper_band_mask(gray, threshold, min_area=20000):
    """Second largest bright component: the lit strip above the panel.

    It sits at y ~ 100-320, wider than the panel, and is the only measurement
    available in the top of the frame. Without it the fit extrapolates up there
    and the strip edge stays visibly curved after correction (measured: 1.1-1.7
    px residual, dropping to 0.3-1.3 px once it is in the fit).
    """
    g = gray.astype(np.float32) if gray.dtype != np.float32 else gray
    b = cv2.morphologyEx((g > threshold).astype(np.uint8), cv2.MORPH_CLOSE,
                         np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(b, 8)
    if n < 3:
        return None
    order = np.argsort(-st[1:, cv2.CC_STAT_AREA]) + 1
    k = order[1]
    if st[k, cv2.CC_STAT_AREA] < min_area:
        return None
    m = (lab == k).astype(np.uint8)
    ff = m.copy()
    cv2.floodFill(ff, np.zeros((m.shape[0] + 2, m.shape[1] + 2), np.uint8), (0, 0), 1)
    return (m | (1 - ff)).astype(np.uint8)


def band_lines(gray, band_mask):
    """Only the long horizontal edges of the strip.

    Its left and right edges are short and usually clipped by the frame, so
    they are not used.
    """
    if band_mask is None:
        return []
    e = extract(gray, band_mask)
    return [p for p in (e["top"], e["bottom"]) if len(p) >= MIN_POINTS]


def usable_lines(edges):
    return [p for p in edges.values() if len(p) >= MIN_POINTS]
