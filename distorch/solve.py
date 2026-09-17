"""Geometric solver: panel edges plus equally spaced point rows.

Two kinds of constraint:
  LINE  straight in the world, straight after correction. Residual = distance
        to the total-least-squares line.
  ROW   equally spaced in the world, equally spaced after correction. Residual
        = distance to the line plus deviation from constant spacing.

The second one carries the new information: straightness cannot see scale error
ALONG a line, equal spacing can. Edges alone leave cx scattered by 134.6 px
across the fleet; with the holes it drops to 31.4 px, which is the real
device-to-device difference.

Each row gets its OWN step: the bright panel is ~248 px, the ring panel ~477 px
because they sit at different depths. No common step is imposed.
"""
import numpy as np
from scipy.optimize import least_squares

from distorch import geom as G, CANON_SIZE, CANON_F, HOLE_SPACING_MM

THETA0_FIXED = {"k1": -1.16, "k2": 1.13, "cx": 972.0, "cy": 523.4, "f": CANON_F}
W_ROW = 3.0          # row residual weight (measured; 1.0 vs 3.0 moves k1 by 0.03)
MIN_POINTS = 8
MIN_ROW = 3          # an n-point row carries 2n-4 information; n=2 carries none


def _theta(p):
    return {"k1": float(p[0]), "k2": float(p[1]), "cx": float(p[2]),
            "cy": float(p[3]), "f": CANON_F}


def _line_residual(q):
    m = q.mean(0)
    c = q - m
    _, _, v = np.linalg.svd(c, full_matrices=False)
    return c @ v[1]


def _row_residual(q):
    """-> (perpendicular, spacing, step)."""
    m = q.mean(0)
    c = q - m
    _, _, v = np.linalg.svd(c, full_matrices=False)
    perp = c @ v[1]
    t = c @ v[0]
    i = np.arange(len(t))
    a = np.column_stack([i, np.ones(len(t))])
    sol, *_ = np.linalg.lstsq(a, t, rcond=None)
    return perp, t - a @ sol, abs(float(sol[0]))


def residual_vector(p, lines, rows, w_row=W_ROW, labelled=False, line_weights=None):
    theta = _theta(p)
    weights = [1.0] * len(lines) if line_weights is None else list(line_weights)
    parts, labels = [], []
    for j, line in enumerate(lines):
        if len(line) < MIN_POINTS:
            continue
        r = _line_residual(G.undistort_points(line, theta)) * weights[j]
        parts.append(r); labels.append(("line", j, len(r), weights[j]))
    for j, row in enumerate(rows):
        if len(row) < MIN_ROW:
            continue
        perp, spacing, _ = _row_residual(G.undistort_points(row, theta))
        parts.append(perp * w_row); labels.append(("row_perp", j, len(perp), w_row))
        parts.append(spacing * w_row); labels.append(("row_spacing", j, len(spacing), w_row))
    if not parts:
        return (np.zeros(0), labels) if labelled else np.zeros(0)
    v = np.concatenate(parts)
    return (v, labels) if labelled else v


def solve(lines, rows, theta0=None, w_row=W_ROW, size=CANON_SIZE, line_weights=None):
    """-> theta plus metrics. rows: list of (N,2) equally spaced point arrays.

    line_weights lets a line count for less. The top strip is added at 0.2:
    it carries ~420 points and at full weight it pulls the fit away from the
    hole row (measured over 101 frames, median hole spread 0.93% -> 1.37%).
    At 0.2 almost all of the benefit is already there:
        weight  hole spread  strip residual  corner error
        0.0        0.93%        1.13 px        4.08 px
        0.2        1.08%        0.97 px        0.86 px
        1.0        1.37%        0.36 px        0.94 px
    """
    w, h = size
    t0 = theta0 or THETA0_FIXED
    res = least_squares(residual_vector, [t0["k1"], t0["k2"], t0["cx"], t0["cy"]],
                        args=(lines, rows, w_row, False, line_weights),
                        bounds=([-3, -3, 0.2 * w, 0.2 * h], [3, 3, 0.8 * w, 0.8 * h]),
                        x_scale=[0.1, 0.1, 100.0, 100.0],
                        xtol=1e-12, ftol=1e-12, max_nfev=300)
    theta = _theta(res.x)

    line_rms = [float(np.sqrt((_line_residual(G.undistort_points(l, theta)) ** 2).mean()))
                for l in lines if len(l) >= MIN_POINTS]
    out = dict(theta=theta, line_rms=line_rms,
               line_rms_all=float(np.sqrt(np.mean(np.square(line_rms)))) if line_rms else None,
               n_lines=len(line_rms), nfev=int(res.nfev), success=bool(res.success))

    info = []
    for row in rows:
        if len(row) < MIN_ROW:
            continue
        q = G.undistort_points(row, theta)
        perp, spacing, step = _row_residual(q)
        gaps = np.diff(q[:, 0])
        info.append(dict(n=len(row), step_px=step,
                         rms_spacing=float(np.sqrt((spacing ** 2).mean())),
                         rms_perp=float(np.sqrt((perp ** 2).mean())),
                         spread_pct=float((gaps.max() - gaps.min()) / gaps.mean() * 100)))
    out["rows"] = info
    if info:
        out["mm_per_px_panel"] = HOLE_SPACING_MM / info[0]["step_px"]
        out["spread_pct"] = info[0]["spread_pct"]
        if len(info) > 1:
            out["depth_ratio"] = info[1]["step_px"] / info[0]["step_px"]
    return out
