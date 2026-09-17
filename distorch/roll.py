"""Camera roll: measurement and size correction. THE IMAGE IS NEVER TOUCHED.

The cameras are screwed to the chamber wall a degree or two off level. Measured
over 104 frames, 16 devices: within one device the angle repeats to +-0.01 deg,
across devices it runs from -1.32 to +1.38 deg. It is a mechanical mounting
error, not noise.

Roll is a plane rotation, so it preserves every length. It costs nothing if the
object is measured along its own axis. It costs a lot through an AXIS-ALIGNED
box, because that box is the smallest upright rectangle around a tilted object:

    roll 1.0 deg -> box width +0.5%, box height +5.8%
    roll 2.7 deg -> box width +1.3%, box height +15.6%

A lying bottle's diameter comes from the box height, which is the sensitive one.

WHY NOT JUST ROTATE THE IMAGE: rotating before the detector moves every pixel,
so the detector sees different input and returns different coordinates. Nothing
here does that. `size_from_box` inverts the box analytically, and `deroll_points`
rotates coordinates the detector has already produced. The frame fed to YOLO is
the same frame as before, and YOLO's own output is unchanged.
"""
import numpy as np

from distorch import CANON_SIZE

# A tilt this large makes cos(2*theta) small and the inversion blows up. The
# fleet sits under 1.5 deg, so this only guards against a garbage angle.
MAX_TILT_DEG = 20.0


def from_row(row):
    """Roll angle in degrees from an UNDISTORTED hole row. + means the right
    end sits lower, matching image coordinates (y down)."""
    p = np.asarray(row, float)
    p = p[np.argsort(p[:, 0])]
    slope = np.polyfit(p[:, 0], p[:, 1], 1)[0]
    return float(np.degrees(np.arctan(slope)))


def rotation(roll_deg, centre=None, size=CANON_SIZE):
    """2x3 matrix that takes roll out, about the frame centre by default."""
    cx, cy = centre if centre is not None else (size[0] / 2.0, size[1] / 2.0)
    a = np.radians(-roll_deg)
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, -sa, cx - ca * cx + sa * cy],
                     [sa,  ca, cy - sa * cx - ca * cy]], float)


def deroll_points(pts, roll_deg, centre=None, size=CANON_SIZE):
    """Rotate coordinates the detector already produced. Use this when the
    detector gives a mask, polygon or oriented box: the points carry the shape,
    so rotating them recovers the true extent exactly."""
    m = rotation(roll_deg, centre, size)
    p = np.asarray(pts, float).reshape(-1, 2)
    return p @ m[:, :2].T + m[:, 2]


def undistort_points(pts, theta, roll_deg, size=CANON_SIZE):
    """Undistort AND level, in one step. Distorted pixel -> straight, level."""
    from distorch import geom as G
    u = G.undistort_points(pts, theta)
    return deroll_points(u, roll_deg, size=size)


def maps(theta, roll_deg, size=CANON_SIZE):
    """One remap that undistorts and levels at the same time.

    The rotation costs nothing: it goes into the map that already exists, so
    the frame is still resampled exactly once. Output pixel -> rotate back into
    the unlevelled rectified frame -> distort -> source pixel.
    """
    import cv2
    from distorch import geom as G
    w, h = size
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float64)
    p = np.stack([gx.ravel(), gy.ravel()], 1)
    m = rotation(-roll_deg, size=size)            # geri dondur
    p = p @ m[:, :2].T + m[:, 2]
    src = G.distort_points(p, theta)
    return (src[:, 0].reshape(h, w).astype(np.float32),
            src[:, 1].reshape(h, w).astype(np.float32))


def undistort_image(img, theta, roll_deg):
    """Corrected AND level frame. One resample, same cost as before."""
    import cv2
    h, w = img.shape[:2]
    mx, my = maps(theta, roll_deg, (w, h))
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR)


def size_from_box(w, h, roll_deg, extra_tilt_deg=0.0):
    """Axis-aligned box -> (along_axis, across_axis), same units as w, h.

    An elongated object at angle t produces
        w = L*cos t + D*sin t
        h = L*sin t + D*cos t
    Known t makes this a 2x2 system with determinant cos(2t):
        L = (w*cos t - h*sin t) / cos 2t
        D = (h*cos t - w*sin t) / cos 2t
    Exact, so an axis-aligned box is enough - the box alone cannot give t, but
    we already know it: the camera roll, plus whatever tilt the object has on
    the belt if the caller can supply it.

    Returns w, h unchanged when the angle is beyond MAX_TILT_DEG.
    """
    t = np.radians(float(roll_deg) + float(extra_tilt_deg))
    if abs(np.degrees(t)) > MAX_TILT_DEG:
        return float(w), float(h)
    ct, st, c2 = np.cos(t), np.sin(t), np.cos(2 * t)
    if abs(c2) < 1e-6:
        return float(w), float(h)
    return float((w * ct - h * st) / c2), float((h * ct - w * st) / c2)


def box_from_size(along, across, roll_deg, extra_tilt_deg=0.0):
    """The forward direction, for tests."""
    t = np.radians(float(roll_deg) + float(extra_tilt_deg))
    ct, st = np.cos(t), np.sin(t)
    return float(along * ct + across * st), float(along * st + across * ct)


def box_error_pct(roll_deg, along, across):
    """How much an uncorrected axis-aligned box overstates each side (%)."""
    w, h = box_from_size(along, across, roll_deg)
    return 100.0 * (w - along) / along, 100.0 * (h - across) / across
