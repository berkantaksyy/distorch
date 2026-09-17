"""Quality measure and acceptance gates.

MAIN MEASURE: predicted corner error.
    How much the uncertainty in the fit turns into position error at the WORST
    point of the frame. It is computed from the points found in THIS frame
    alone, without knowing the truth, which is why it works as a runtime gate.
        1. numerical Jacobian of the residual vector w.r.t. theta
        2. covariance C = (J^T J)^-1, residuals pre-divided by sigma
        3. sample thetas from C, look at the spread of the four corners

Counting holes would be the wrong gate. What matters is SPAN, measured:
    7 holes 5.2 px · middle one missing 7.0 · one end missing 7.7
    both ends missing 124 px · only the middle three: singular
One corner-error gate separates all of these by itself.
"""
import numpy as np

from distorch import geom as G, solve as S, CANON_SIZE

SIGMA_LINE = 0.5       # edge sub-pixel noise (px)
SIGMA_ROW = 0.3        # hole / ring centre noise (px)

# Thresholds measured on 106 field frames, with the top strip in the fit:
#   corner px   median 2.50  p90 3.39  p95 3.56  p99 4.39  max 4.40
#   spread %    median 1.05  p90 1.87  p95 2.02            max 2.73
#   line rms    median 0.72  p90 1.39  p99 1.77            max 2.13
#   max resid   median 1.12  p90 2.08  p99 6.04            max 6.48
#   symmetry px median 4.9   p90 8.0   p99 11.0            max 11.2
# Warn near p95-p99, reject at about twice the healthy maximum. A corruption
# test (one hole shifted 8 px) pushes spread to 6.65%, caught by the 3.0 reject.
# Line RMS sits higher than before because the strip is a harder line and is
# now counted; the strip is what made the corner error drop by half.
CORNER_REJECT, CORNER_WARN = 10.0, 6.0
SPREAD_REJECT, SPREAD_WARN = 3.0, 2.0
LINE_RMS_REJECT, LINE_RMS_WARN = 4.0, 2.0
MAX_RESID_REJECT, MAX_RESID_WARN = 10.0, 4.0
SYMMETRY_WARN = 12.0
CX_RANGE, CY_RANGE = (900.0, 1080.0), (450.0, 700.0)

# Guard threshold, from the scatter left after removing the systematic bias
# (106 frames): median 6.26 + 3*MAD 2.58 = 14.0 px.
NET_THRESHOLD = 6.3      # v2: pipeline CNN-geometry gap median 3.12 + 3*MAD 1.06
                         # (v3 was 14.0 and the worst frame already sat at 14.46)


def _jacobian(p, lines, rows, w_row=S.W_ROW, line_weights=None):
    """Jacobian of the sigma-whitened residual vector w.r.t. theta.

    The residual is already multiplied by the line weight, so dividing by sigma
    gives weight/sigma - exactly the whitening a down-weighted line deserves.
    """
    _, labels = S.residual_vector(p, lines, rows, w_row, labelled=True,
                                  line_weights=line_weights)
    if not labels:
        return np.zeros((0, 4))
    scale = np.concatenate([
        np.full(n, 1.0 / (SIGMA_LINE if kind == "line" else SIGMA_ROW * w_row))
        for kind, _, n, _w in labels])
    step = [1e-4, 1e-4, 1e-2, 1e-2]
    cols = []
    for k in range(4):
        d = np.zeros(4); d[k] = step[k]
        r1 = S.residual_vector(np.asarray(p) + d, lines, rows, w_row,
                               line_weights=line_weights)
        r0 = S.residual_vector(np.asarray(p) - d, lines, rows, w_row,
                               line_weights=line_weights)
        cols.append((r1 - r0) / (2 * step[k]) * scale)
    return np.array(cols).T


def corner_error(lines, rows, theta, n_samples=1000, size=CANON_SIZE, seed=0,
                 line_weights=None):
    """Predicted corner position error, px, 1 sigma. Large means untrustworthy."""
    p = np.array([theta["k1"], theta["k2"], theta["cx"], theta["cy"]], float)
    j = _jacobian(p, lines, rows, line_weights=line_weights)
    if j.size == 0 or j.shape[0] < 5:
        return float("inf")
    try:
        c = np.linalg.inv(j.T @ j)
        chol = np.linalg.cholesky(c + np.eye(4) * 1e-18)
    except np.linalg.LinAlgError:
        return float("inf")
    if not np.all(np.isfinite(c)) or np.any(np.diag(c) < 0):
        return float("inf")
    w, h = size
    corners = np.array([[0., 0.], [w, 0.], [0., h], [w, h]])
    base = G.distort_points(corners, theta)
    rng = np.random.default_rng(seed)
    err = np.empty(n_samples)
    for i in range(n_samples):
        q = p + chol @ rng.standard_normal(4)
        t = {"k1": q[0], "k2": q[1], "cx": q[2], "cy": q[3], "f": theta["f"]}
        err[i] = np.linalg.norm(G.distort_points(corners, t) - base, axis=1).max()
    return float(np.percentile(err, 68))


def point_residuals(rows, theta):
    """Per-point residual, hypot(perpendicular, spacing).

    This is what catches a misdetected point. Leave-one-out was tried first and
    is unusable: dropping any of 7 points already weakens the solve, so even a
    clean frame predicts the held-out point 43 px off. Measured:
        clean                 spread 2.13%  max resid 2.31 px  loo 43.7 px
        one hole  +8 px off   spread 6.65%  max resid 5.90 px  loo 47.3 px
        one hole +20 px off   spread 15.9%  max resid 13.7 px  loo 53.1 px
    Spread and max residual separate them, loo does not - and this needs no
    re-solve, so it is 100x cheaper.
    """
    out = []
    for row in rows:
        if len(row) < 3:
            out.append(np.zeros(0))
            continue
        perp, spacing, _ = S._row_residual(G.undistort_points(row, theta))
        out.append(np.hypot(perp, spacing))
    return out


def max_residual(rows, theta):
    v = [r.max() for r in point_residuals(rows, theta) if len(r)]
    return float(max(v)) if v else None


def symmetry(row):
    """Distortion is symmetric about the centre, so mirrored gaps should match."""
    if len(row) < 4:
        return None
    g = np.diff(np.asarray(row)[:, 0])
    n = len(g) // 2
    return float(np.abs(g[:n] - g[::-1][:n]).sum())


def gates(theta, corner_px, spread_pct, line_rms, max_resid=None, symmetry_px=None,
          mm_per_px=None, mm_fleet=None, undistorted=False,
          net_diff=None, net_threshold=None, stage2_helped=None):
    checks = []

    def add(level, name, message):
        checks.append({"level": level, "check": name, "message": message})

    if undistorted:
        add("reject", "undistorted", "frame is already corrected, not raw")
    if corner_px is None or not np.isfinite(corner_px):
        add("reject", "corner", "corner error not computable (singular solve)")
    elif corner_px > CORNER_REJECT:
        add("reject", "corner", f"corner error {corner_px:.1f} px > {CORNER_REJECT}")
    elif corner_px > CORNER_WARN:
        add("warn", "corner", f"corner error {corner_px:.1f} px > {CORNER_WARN}")
    if spread_pct is not None:
        if spread_pct > SPREAD_REJECT:
            add("reject", "spread", f"spacing spread {spread_pct:.2f}% > {SPREAD_REJECT}")
        elif spread_pct > SPREAD_WARN:
            add("warn", "spread", f"spacing spread {spread_pct:.2f}% > {SPREAD_WARN}")
    if line_rms is not None:
        if line_rms > LINE_RMS_REJECT:
            add("reject", "line", f"line RMS {line_rms:.2f} px > {LINE_RMS_REJECT}")
        elif line_rms > LINE_RMS_WARN:
            add("warn", "line", f"line RMS {line_rms:.2f} px > {LINE_RMS_WARN}")
    if max_resid is not None:
        if max_resid > MAX_RESID_REJECT:
            add("reject", "residual",
                f"largest point residual {max_resid:.2f} px > {MAX_RESID_REJECT}")
        elif max_resid > MAX_RESID_WARN:
            add("warn", "residual", f"largest point residual {max_resid:.2f} px")
    if symmetry_px is not None and symmetry_px > SYMMETRY_WARN:
        add("warn", "symmetry", f"symmetry deviation {symmetry_px:.1f} px")
    if not (CX_RANGE[0] < theta["cx"] < CX_RANGE[1]):
        add("warn", "cx", f"cx {theta['cx']:.0f} outside fleet range {CX_RANGE}")
    if not (CY_RANGE[0] < theta["cy"] < CY_RANGE[1]):
        add("warn", "cy", f"cy {theta['cy']:.0f} outside fleet range {CY_RANGE}")
    if mm_per_px and mm_fleet and abs(mm_per_px - mm_fleet) / mm_fleet > 0.05:
        add("warn", "scale", f"mm/px {mm_per_px:.4f} is >5% off the fleet median")
    ok, notes = G.is_valid(theta)
    if not ok:
        for n in notes:
            add("reject", "validity", n)
    if G.black_border(theta) > 0:
        add("warn", "black", "corrected image may show a black border")
    if net_diff is not None and net_threshold is not None and net_diff > net_threshold:
        add("warn", "net", f"CNN disagrees by {net_diff:.1f} px > {net_threshold:.1f}")
    # Dropping stage 2 is NOT a warning: it happens on half the frames and is
    # harmless, theta from stage 1 is already valid. Warning on it would drown
    # the real warnings.
    if stage2_helped is False:
        add("info", "stage2", "rings did not lower the corner error, dropped")

    verdict = ("reject" if any(c["level"] == "reject" for c in checks)
               else "warn" if any(c["level"] == "warn" for c in checks)
               else "accept")
    return {"verdict": verdict, "checks": checks}
