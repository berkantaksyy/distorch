#!/usr/bin/env python3
"""Model (distort_v3) against the system (distorch), one metric.

METRIC: measurement inconsistency (%).
    The 7 panel holes are equally spaced in the world (80 mm). After correction
    the neighbouring gaps are measured: (max - min) / mean. The number says
    directly how much the same object changes size depending on where it sits
    in the frame. 0 is perfect.

    python -m tools.compare                # metric over the whole data set
    python -m tools.compare ACO_ANKA_0045  # full-frame visual for one frame
"""
import sys, re, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, cv2
import distorch as D
from distorch import geom as G, net as N, calibrate as C, holes as H, rings as R, roll as RL

TEST_DEVICES = {"ACO_6600_0003", "ACO_6600_0015", "ACO_ANKA_0045"}
VAL_DEVICES = {"ACO_6600_0007", "ACO_ANKA_0033"}
VIEW_W = 1280


def device_of(name):
    m = re.match(r"(ACO_ANKA_\d{4}|ACO_6600_\d{4}|ACO-ANKA-A1100-\d{4})(_\w+)?$", name)
    return m.group(1) if m else name


def undistort_image(img, theta, fit=False, scale=None, roll_deg=0.0):
    """Corrected image. Source of each OUTPUT pixel = distort(output).

    roll_deg levels the frame inside the SAME map: the rotation is folded into
    the coordinates before distort(), so the frame is still resampled exactly
    once and costs nothing extra.

    fit=False fills the frame: the outer region is stretched and any residual
    shows up there. fit=True instead scales the whole corrected extent to fit
    inside the frame, leaving black margins - the same framing the existing
    production correction uses. It changes nothing about the geometry, only
    what is visible; production corrects coordinates, never the image.
    """
    h, w = img.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    out = np.column_stack([gx.ravel(), gy.ravel()])
    if fit:
        corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], float)
        u = G.undistort_points(corners, theta)
        lo, hi = u.min(0), u.max(0)
        k = min(w / (hi[0] - lo[0]), h / (hi[1] - lo[1])) if scale is None else scale
        off = np.array([w, h]) / 2 - (lo + hi) / 2 * k
        out = (out - off) / k
    if roll_deg:
        m = RL.rotation(-roll_deg, size=(w, h))
        out = out @ m[:, :2].T + m[:, 2]
    src = G.distort_points(out, theta)
    return cv2.remap(img, src[:, 0].reshape(h, w).astype(np.float32),
                     src[:, 1].reshape(h, w).astype(np.float32),
                     cv2.INTER_LINEAR, borderValue=(0, 0, 0)), (k, off) if fit else None


def _largest_inner_rect(valid):
    """Largest axis-aligned rectangle fully inside the valid mask (maximal
    rectangle in a binary matrix, histogram method)."""
    h, w = valid.shape
    heights = np.zeros(w, np.int32)
    best = (0, 0, 0, 0, 0)          # area, x0, y0, x1, y1
    for y in range(h):
        heights = np.where(valid[y], heights + 1, 0)
        stack = []
        for x in range(w + 1):
            cur = heights[x] if x < w else 0
            start = x
            while stack and stack[-1][1] >= cur:
                sx, sh = stack.pop()
                area = sh * (x - sx)
                if area > best[0]:
                    best = (area, sx, y - sh + 1, x, y + 1)
                start = sx
            stack.append((start, cur))
    return best[1:]


def crop_tight(img, theta, up=0.45, down=0.55, side=0.18, kes=False):
    """Crop to the chamber opening, the way the existing production correction
    frames it: panel bounding box grown by a fraction of its own size, then
    clipped to the region that has real source pixels.

    This HIDES the outer residual, it does not fix it.

    kes=False IS THE DEFAULT AND CROPS NOTHING. The corrected frame comes back
    whole, offset (0, 0), and the panel is never even looked for. Cropping is
    opt-in here, exactly the way the panel does it: its four sliders sit at 0
    until somebody moves them. kes=True brings back the panel-relative crop
    described below; the CLI --tight passes it, and --up/--down/--side set the
    amounts without editing this file.

    down was 0.25 and cut the bottom off: bottles sit low, on the belt below the
    panel, and were being lost. Measured over 40 frames, the room below the
    panel is 0.64 of the panel height (median), 0.47 at worst, so 0.55 keeps
    the whole ring row and the belt. Where there is less room it simply stops
    at the frame edge. up stays at 0.45 - that side has only 0.47 (median) and
    is where the stretched residual lives. side has 0.06 available, so 0.18
    already means full width.
    """
    out, _ = undistort_image(img, theta)
    if not kes:
        return out, (0, 0)
    grey = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32)
    _, filled, _ = H.panel_mask(grey)
    ys, xs = np.nonzero(filled)
    if len(ys) == 0:
        return out, (0, 0)
    ph, pw = ys.max() - ys.min(), xs.max() - xs.min()
    y0 = int(max(0, ys.min() - up * ph))
    y1 = int(min(out.shape[0], ys.max() + down * ph))
    x0 = int(max(0, xs.min() - side * pw))
    x1 = int(min(out.shape[1], xs.max() + side * pw))
    vy0, vy1, vx0, vx1 = _valid_box(img, theta)
    y0, y1 = max(y0, vy0), min(y1, vy1)
    x0, x1 = max(x0, vx0), min(x1, vx1)
    return out[y0:y1, x0:x1], (x0, y0)


def _valid_box(img, theta, margin=2, step=4):
    h, w = img.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    src = G.distort_points(np.column_stack([gx.ravel(), gy.ravel()]), theta)
    valid = ((src[:, 0] >= margin) & (src[:, 0] < w - margin) &
             (src[:, 1] >= margin) & (src[:, 1] < h - margin)).reshape(h, w)
    x0, y0, x1, y1 = _largest_inner_rect(valid[::step, ::step])
    return y0 * step, y1 * step, x0 * step, x1 * step


def crop_to_valid(img, theta, margin=2):
    """Corrected image cropped to the region that has real source pixels.

    This is what the existing production correction does: the stretched outer
    region is thrown away instead of being shown. It hides the residual, it
    does not remove it - and it changes nothing in production, where the
    coordinates are corrected and the image never is.
    """
    out, _ = undistort_image(img, theta)
    h, w = img.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    src = G.distort_points(np.column_stack([gx.ravel(), gy.ravel()]), theta)
    valid = ((src[:, 0] >= margin) & (src[:, 0] < w - margin) &
             (src[:, 1] >= margin) & (src[:, 1] < h - margin)).reshape(h, w)
    step = 4                                    # search on a coarse grid
    x0, y0, x1, y1 = _largest_inner_rect(valid[::step, ::step])
    x0, y0, x1, y1 = x0 * step, y0 * step, x1 * step, y1 * step
    return out[y0:y1, x0:x1], (x0, y0)


def _to_view(img, theta, mode, kes_orani=None):
    """-> (image, point mapper). The mapper takes raw points to view pixels.

    mode: "full" fills the frame · "fit" scales everything in · "crop" keeps
    only the region that has real source pixels · "tight" is the only mode that
    crops around the panel, and it has to ask for it (crop_tight kes=True).
    """
    if theta is None:
        return img, (lambda p: p)
    if mode in ("crop", "tight"):
        out, (x0, y0) = (crop_to_valid(img, theta) if mode == "crop" else
                         crop_tight(img, theta, kes=True, **(kes_orani or {})))
        return out, (lambda p: G.undistort_points(p, theta) - np.array([x0, y0]))
    out, tf = undistort_image(img, theta, fit=(mode == "fit"))
    if mode != "fit":
        return out, (lambda p: G.undistort_points(p, theta))
    k, off = tf
    return out, (lambda p: G.undistort_points(p, theta) * k + off)


def inconsistency(centres, theta):
    g = np.diff(G.undistort_points(centres, theta)[:, 0])
    return float((g.max() - g.min()) / g.mean() * 100)


def _mark_row(img, centres, colour, above, marker):
    """Draw one row of equally spaced points and write the gaps. -> spread %."""
    gaps = np.diff(centres[:, 0])
    for x, y in centres:
        cv2.drawMarker(img, (int(x), int(y)), colour, marker, 26, 3)
    for i, gap in enumerate(gaps):
        mx = int((centres[i][0] + centres[i + 1][0]) / 2)
        my = int((centres[i][1] + centres[i + 1][1]) / 2)
        cv2.line(img, (int(centres[i][0]), my + above), (int(centres[i + 1][0]), my + above),
                 colour, 2)
        cv2.putText(img, f"{gap:.0f}", (mx - 30, my + above - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, colour, 2)
    return float((gaps.max() - gaps.min()) / gaps.mean() * 100)


def _panel(img, holes, rings, title, colour, ring_colour=(255, 120, 0)):
    """Frame with the hole row and, when present, the ring row."""
    out = img.copy()
    if rings is not None and len(rings) >= 3:
        inside = ((rings[:, 0] > 0) & (rings[:, 0] < out.shape[1]) &
                  (rings[:, 1] > 0) & (rings[:, 1] < out.shape[0]))
        rings = rings if inside.all() else None
    spread = _mark_row(out, holes, colour, -52, cv2.MARKER_CROSS)
    text = f"{title}    delik %{spread:.2f}"
    if rings is not None and len(rings) >= 3:
        rs = _mark_row(out, rings, ring_colour, 74, cv2.MARKER_DIAMOND)
        text += f"    bilezik %{rs:.2f}"
    cv2.rectangle(out, (0, 0), (out.shape[1], 52), (255, 255, 255), -1)
    cv2.putText(out, text, (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2)
    return cv2.resize(out, (VIEW_W, int(out.shape[0] * VIEW_W / out.shape[1])))


def visual(name, mode="full", kes_orani=None):
    path = D.FRAMES_DIR / f"{name}.jpg"
    if not path.exists():
        sys.exit(f"frame not found: {path}")
    img = cv2.imread(str(path))
    r = C.calibrate(img)
    holes = np.array(r["stage1"]["centres"], float)
    theta_net = N.theta_from_frame(img, correct_bias=False)

    # Rings come from the calibration itself when stage 2 used them; otherwise
    # detect them once on the RAW frame. Either way they are raw-frame points,
    # carried through each theta exactly like the holes, so all three panels
    # show the same physical points.
    rings = None
    if r["stage2"].get("centres"):
        rings = np.array(r["stage2"]["centres"], float)
    else:
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, filled, _ = H.panel_mask(grey.astype(np.float32))
        found, _ = R.find(grey, filled, r["theta"], hole_row=holes)
        if len(found) >= 3:
            rings = R.to_array(found)

    panels = []
    for title, theta, colour in (
            (f"{name}  HAM (duzeltmesiz)", None, (0, 0, 255)),
            (f"{name}  MODEL ile (distort_v3)", theta_net, (0, 140, 255)),
            (f"{name}  SISTEM ile (distorch)", r["theta"], (0, 160, 0))):
        view, to_view = _to_view(img, theta, mode, kes_orani)
        panels.append(_panel(view, to_view(holes),
                             None if rings is None else to_view(rings),
                             title, colour))
    w = max(p.shape[1] for p in panels)
    panels = [p if p.shape[1] == w else
              cv2.copyMakeBorder(p, 0, 0, 0, w - p.shape[1], cv2.BORDER_CONSTANT, value=0)
              for p in panels]
    out = D.MEASURE_DIR / (f"compare_{name}.jpg" if mode == "full"
                           else f"compare_{name}_{mode}.jpg")
    cv2.imwrite(str(out), np.vstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 88])
    line = lambda label, th: (f"  {label:8s} delik %{inconsistency(holes, th):5.2f}" +
                              (f"   bilezik %{inconsistency(rings, th):5.2f}"
                               if rings is not None else "   bilezik -"))
    print(name)
    print(line("ham", G.IDENTITY)); print(line("model", theta_net))
    print(line("sistem", r["theta"]))
    print(f"-> {out}")


def table():
    rows = []
    for p in sorted(D.FRAMES_DIR.glob("*.jp*g")):
        img = cv2.imread(str(p))
        r = C.calibrate(img)
        if "theta" not in r:
            continue
        c = np.array(r["stage1"]["centres"], float)
        theta_net = N.theta_from_frame(img, correct_bias=False)
        rows.append(dict(frame=p.stem, device=device_of(p.stem),
                         raw=inconsistency(c, G.IDENTITY),
                         model=inconsistency(c, theta_net) if theta_net else None,
                         system=inconsistency(c, r["theta"])))
    print("\nOLCUM TUTARSIZLIGI (%) - ayni cismin kadraj boyunca boy farki, medyan\n")
    print("| grup                   |   n | duzeltmesiz |  model | sistem | kazanc |")
    print("|---|---|---|---|---|---|")
    for label, sel in (("TUM VERI SETI", lambda r: True),
                       ("TEST (gorulmemis)", lambda r: r["device"] in TEST_DEVICES),
                       ("VAL", lambda r: r["device"] in VAL_DEVICES),
                       ("TRAIN (model gordu)",
                        lambda r: r["device"] not in TEST_DEVICES | VAL_DEVICES)):
        v = [r for r in rows if sel(r)]
        if not v:
            continue
        raw = np.median([r["raw"] for r in v])
        mo = np.median([r["model"] for r in v])
        sy = np.median([r["system"] for r in v])
        print(f"| {label:22s} | {len(v):3d} | {raw:6.2f} | {mo:6.2f} | {sy:6.2f} "
              f"| {mo/sy:5.1f}x |")
    mo = np.array([r["model"] for r in rows])
    sy = np.array([r["system"] for r in rows])
    print(f"\nmodel  : medyan {np.median(mo):.2f}%  ort {mo.mean():.2f}%  "
          f"p90 {np.percentile(mo, 90):.2f}%  en kotu {mo.max():.2f}%")
    print(f"sistem : medyan {np.median(sy):.2f}%  ort {sy.mean():.2f}%  "
          f"p90 {np.percentile(sy, 90):.2f}%  en kotu {sy.max():.2f}%")
    print(f"sistem her karede modelden iyi: {bool((sy < mo).all())} "
          f"({(sy < mo).sum()}/{len(sy)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("frame", nargs="?", help="frame name, e.g. ACO_ANKA_0045")
    ap.add_argument("--fit", action="store_true",
                    help="scale the corrected image to fit the frame (black margins)")
    ap.add_argument("--crop", action="store_true",
                    help="crop to the region that has real source pixels")
    ap.add_argument("--tight", action="store_true",
                    help="crop tightly around the chamber opening (hides the outer residual); "
                         "without this flag nothing is cropped")
    ap.add_argument("--up", type=float, default=0.45,
                    help="--tight: grow the panel box upwards by this fraction of its height")
    ap.add_argument("--down", type=float, default=0.55,
                    help="--tight: grow it downwards (0.55 keeps the ring row and the belt)")
    ap.add_argument("--side", type=float, default=0.18,
                    help="--tight: grow it sideways by this fraction of its width")
    a = ap.parse_args()
    mode = "tight" if a.tight else ("crop" if a.crop else ("fit" if a.fit else "full"))
    if a.frame:
        visual(a.frame, mode=mode,
               kes_orani={"up": a.up, "down": a.down, "side": a.side})
    else:
        table()
