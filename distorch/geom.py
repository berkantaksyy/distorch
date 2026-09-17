"""Distortion geometry: Brown radial model with two coefficients.

    r2 = x^2 + y^2                      normalised: x = (X - cx) / f
    g  = 1 + k1*r2 + k2*r2^2
    distort   (straight -> seen):  Xd = x*g*f + cx
    undistort (seen -> straight):  invert g per ray
"""
import numpy as np

from distorch import CANON_SIZE, CANON_F

ANCHORS = (0.30, 0.60)

THETA_PLINEBIG = {"k1": -1.1157852656498009, "k2": 0.9844536002193723,
                  "cx": 1009.1741, "cy": 618.2971, "f": CANON_F}
IDENTITY = {"k1": 0.0, "k2": 0.0, "cx": CANON_SIZE[0] / 2,
            "cy": CANON_SIZE[1] / 2, "f": CANON_F}


def radial_gain(r, k1, k2):
    r2 = np.asarray(r, float) ** 2
    return 1.0 + k1 * r2 + k2 * r2 * r2


def radial_forward(r, k1, k2):
    return np.asarray(r, float) * radial_gain(r, k1, k2)


def is_monotonic(k1, k2, r_hi, n=512):
    """False means the model folds over itself within r_hi."""
    r = np.linspace(1e-6, r_hi, n)
    return bool(np.all(np.diff(radial_forward(r, k1, k2)) > 0))


def radial_inverse(r_d, k1, k2, n_fixed=8, n_newton=6):
    """Distorted radius -> straight radius.

    Fixed-point first, then Newton. Fixed-point alone leaves 7 px of error in
    the corner at k1 ~ -1.35 (measured); Newton takes it to machine precision.
    """
    r_d = np.asarray(r_d, float)
    r = r_d.copy()
    for _ in range(n_fixed):
        r = r_d / radial_gain(r, k1, k2)
    for _ in range(n_newton):
        r2 = r * r
        f = r * (1.0 + k1 * r2 + k2 * r2 * r2) - r_d
        df = 1.0 + 3.0 * k1 * r2 + 5.0 * k2 * r2 * r2
        r = r - f / np.where(np.abs(df) < 1e-12, 1e-12, df)
    return r


# Anchor convention, must match distort_v1..v3 metadata:
# anchors live in the DISTORTED radius domain and m = undistort magnification,
#     m_i = radial_inverse(a_i) / a_i          (> 1, since g < 1)
# Verified: theta0 (k1 -1.15963, k2 1.12685) -> m = (1.135603, 1.368882),
# which is exactly envelope.m_ref in distort_v3_meta.json.
# NOT m = g(a): that convention flips the sign of k1.

def mags_from_kk(k1, k2, anchors=ANCHORS):
    a = np.asarray(anchors, float)
    return tuple(float(x) for x in (radial_inverse(a, k1, k2) / a))


def kk_from_mags(m1, m2, anchors=ANCHORS):
    """At r_i = m_i*a_i the gain must be 1/m_i, which is linear in (k1, k2)."""
    a1, a2 = anchors
    r1, r2 = m1 * a1, m2 * a2
    A = np.array([[r1 ** 2, r1 ** 4], [r2 ** 2, r2 ** 4]], float)
    b = np.array([1.0 / m1 - 1.0, 1.0 / m2 - 1.0], float)
    k1, k2 = np.linalg.solve(A, b)
    return float(k1), float(k2)


def distort_points(pts, theta):
    """Straight -> distorted (what the camera does)."""
    p = np.asarray(pts, float).reshape(-1, 2)
    f, cx, cy = theta.get("f", CANON_F), theta["cx"], theta["cy"]
    x = (p[:, 0] - cx) / f
    y = (p[:, 1] - cy) / f
    g = radial_gain(np.hypot(x, y), theta["k1"], theta["k2"])
    return np.column_stack([x * g * f + cx, y * g * f + cy])


def undistort_points(pts, theta, n_fixed=8, n_newton=6):
    """Distorted -> straight. Ray direction is kept, only the radius changes."""
    p = np.asarray(pts, float).reshape(-1, 2)
    f, cx, cy = theta.get("f", CANON_F), theta["cx"], theta["cy"]
    xd = (p[:, 0] - cx) / f
    yd = (p[:, 1] - cy) / f
    rd = np.hypot(xd, yd)
    r = radial_inverse(rd, theta["k1"], theta["k2"], n_fixed, n_newton)
    s = np.where(rd < 1e-12, 1.0, r / np.where(rd < 1e-12, 1.0, rd))
    return np.column_stack([xd * s * f + cx, yd * s * f + cy])


def grid_points(size=CANON_SIZE, nx=33, ny=19):
    w, h = size
    gx, gy = np.meshgrid(np.linspace(0, w, nx), np.linspace(0, h, ny))
    return np.column_stack([gx.ravel(), gy.ravel()])


def displacement(theta_a, theta_b, size=CANON_SIZE, nx=33, ny=19):
    """How far apart two profiles move the same grid, in px."""
    p = grid_points(size, nx, ny)
    d = np.linalg.norm(distort_points(p, theta_a) - distort_points(p, theta_b), axis=1)
    return {"mean": float(d.mean()), "max": float(d.max()),
            "p95": float(np.percentile(d, 95))}


def max_radius(theta, size=CANON_SIZE):
    w, h = size
    cx, cy, f = theta["cx"], theta["cy"], theta.get("f", CANON_F)
    c = np.array([[0, 0], [w, 0], [0, h], [w, h]], float)
    return float(np.max(np.hypot((c[:, 0] - cx) / f, (c[:, 1] - cy) / f)))


def is_valid(theta, size=CANON_SIZE, gain_range=(0.55, 1.00)):
    """Physically plausible profile? -> (ok, notes)."""
    w, h = size
    notes = []
    if not (0.2 * w < theta["cx"] < 0.8 * w):
        notes.append("cx out of range")
    if not (0.2 * h < theta["cy"] < 0.8 * h):
        notes.append("cy out of range")
    rmax = max_radius(theta, size)
    if not is_monotonic(theta["k1"], theta["k2"], rmax * 1.45):
        notes.append("model not monotonic (fold-over)")
    g = float(radial_gain(rmax, theta["k1"], theta["k2"]))
    if not (gain_range[0] <= g <= gain_range[1]):
        notes.append(f"corner gain {g:.3f} outside {gain_range}")
    return (len(notes) == 0), notes


def black_border(theta, size=CANON_SIZE):
    """Samples of the output border whose source falls outside the frame.

    Resampling looks up source = distort(output), so the test runs in the
    distort direction, not undistort. 0 means no black edge.
    """
    w, h = size
    border = np.concatenate([
        np.column_stack([np.linspace(0, w, 200), np.zeros(200)]),
        np.column_stack([np.linspace(0, w, 200), np.full(200, h)]),
        np.column_stack([np.zeros(200), np.linspace(0, h, 200)]),
        np.column_stack([np.full(200, w), np.linspace(0, h, 200)])])
    s = distort_points(border, theta)
    out = (s[:, 0] < 0) | (s[:, 0] > w - 1) | (s[:, 1] < 0) | (s[:, 1] > h - 1)
    return int(out.sum())


def theta_str(t):
    return (f"k1={t['k1']:+.4f} k2={t['k2']:+.4f} "
            f"cx={t['cx']:.1f} cy={t['cy']:.1f} f={t.get('f', CANON_F):.0f}")
