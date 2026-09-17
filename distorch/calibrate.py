"""Single entry point: empty-chamber frame -> profile JSON.

    python -m distorch.calibrate frame.jpg [--device NAME] [--out path.json]
                                [--overlay] [--no-net] [--no-rings] [--quiet]
"""
import sys, time, argparse
from pathlib import Path

import numpy as np
import cv2

import distorch as D
from distorch import (geom as G, holes as H, edges as E, solve as S,
                      rings as R, quality as Q, net as N, profile as P,
                      roll as RL)

MM_FLEET = 0.3226        # fleet median over 106 frames, for the scale gate
W_BAND = 0.2             # weight of the top strip lines (see solve.solve)


def _stage1(gray, filled, theta0, threshold):
    """Panel edges + holes, plus the top strip when it is there.

    The strip is the only measurement above the panel. Without it the fit
    extrapolates into the top of the frame and the strip edge stays visibly
    curved after correction; adding it takes the corner error from 4.08 to
    0.86 px (median over 101 frames) and improves panel width consistency
    across the full height from 0.36% to 0.31%.
    """
    holes = H.find(gray)
    edges = E.extract(gray, filled)
    lines = E.usable_lines(edges)
    if len(holes) < 3 or len(lines) < 2:
        return None
    band = E.band_lines(gray, E.upper_band_mask(gray, threshold))
    weights = [1.0] * len(lines) + [W_BAND] * len(band)
    all_lines = lines + band
    row = H.centres(holes)
    res = S.solve(all_lines, [row], theta0=theta0, line_weights=weights)
    return dict(holes=holes, edges=edges, lines=all_lines, weights=weights,
                n_band=len(band), row=row, res=res,
                corner=Q.corner_error(all_lines, [row], res["theta"],
                                      line_weights=weights))


def _stage2(gray, filled, s1):
    """Rings. Returns the stage-1 result unchanged when they do not help."""
    rings, info = R.find(gray, filled, s1["res"]["theta"], hole_row=s1["row"])
    out = {"used": False, "n_rings": len(rings), "n_seeds": info["n_seeds"],
           "ratio": None if info["ratio"] is None else round(info["ratio"], 4),
           "ratio_ok": info["ratio_ok"]}
    if len(rings) < 3:
        return s1["res"], s1["corner"], out
    arr = R.to_array(rings)
    res = S.solve(s1["lines"], [s1["row"], arr], theta0=s1["res"]["theta"],
                  line_weights=s1["weights"])
    corner = Q.corner_error(s1["lines"], [s1["row"], arr], res["theta"],
                            line_weights=s1["weights"])
    out["corner"] = round(corner, 3)
    if corner >= s1["corner"]:          # gate 10: never make it worse
        return s1["res"], s1["corner"], out
    out["used"] = True
    out["step_px"] = round(res["rows"][1]["step_px"], 2)
    out["centres"] = [[round(r["cx"], 2), round(r["cy"], 2)] for r in rings]
    return res, corner, out


def calibrate(frame, use_net=True, use_rings=True):
    """-> report dict. Never raises; failure comes back as verdict 'reject'."""
    t0 = time.perf_counter()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    _, filled, threshold = H.panel_mask(gray.astype(np.float32))

    theta_net = N.theta_from_frame(frame) if use_net else None
    net_info = None
    if theta_net:
        ok, note = N.envelope_ok(theta_net)
        net_info = {"theta": {k: round(theta_net[k], 4) for k in ("k1", "k2", "cx", "cy")},
                    "envelope_ok": bool(ok), "envelope_note": note, "bias_corrected": True}

    s1 = _stage1(gray, filled, theta_net or S.THETA0_FIXED, threshold)
    if s1 is None:
        return {"verdict": "reject", "reason": "not enough measurements",
                "ms": round((time.perf_counter() - t0) * 1e3)}

    res, corner, stage2 = ((s1["res"], s1["corner"], {"used": False, "n_rings": 0})
                           if not use_rings else _stage2(gray, filled, s1))
    theta = res["theta"]

    roll_deg = RL.from_row(G.undistort_points(s1["row"], theta))
    max_resid = Q.max_residual([s1["row"]], theta)
    sym = Q.symmetry(s1["row"])
    diff = G.displacement(theta_net, theta)["mean"] if theta_net else None
    if net_info:
        net_info.update(diff_px=round(diff, 2), threshold_px=Q.NET_THRESHOLD)

    checks = Q.gates(theta, corner, res.get("spread_pct"), res["line_rms_all"],
                     max_resid, sym, res.get("mm_per_px_panel"), MM_FLEET,
                     undistorted=H.is_undistorted(s1["holes"]),
                     net_diff=diff, net_threshold=Q.NET_THRESHOLD,
                     stage2_helped=stage2["used"] if use_rings else None)

    return {
        "verdict": checks["verdict"], "checks": checks["checks"], "theta": theta,
        "stage1": {"n_holes": len(s1["holes"]),
                   "spread_pct": round(res["spread_pct"], 3) if res.get("spread_pct") else None,
                   "step_px": round(res["rows"][0]["step_px"], 2),
                   "centre_method": "ellipse",
                   "centres": [[round(x, 2), round(y, 2)] for x, y in s1["row"]],
                   "centres_centroid": [[round(h["cx"], 2), round(h["cy"], 2)]
                                        for h in s1["holes"]]},
        "stage2": stage2,
        "edges": {**{k: len(v) for k, v in s1["edges"].items()},
                  "band": s1["n_band"]},
        "line_rms": [round(x, 3) for x in res["line_rms"]],
        "line_rms_all": round(res["line_rms_all"], 3) if res["line_rms_all"] else None,
        "quality": {"corner_px": round(corner, 3), "corner_stage1_px": round(s1["corner"], 3),
                    "max_residual_px": round(max_resid, 3) if max_resid else None,
                    "symmetry_px": round(sym, 2) if sym else None},
        "net": net_info, "mm_per_px_panel": res.get("mm_per_px_panel"),
        # Mounting roll, measured but NOT applied: nothing here rotates the
        # image. distorch.roll turns it into a size correction downstream.
        "roll_deg": round(roll_deg, 3),
        "ms": round((time.perf_counter() - t0) * 1e3),
    }


def draw(frame, report, path):
    out = frame.copy()
    for x, y in report["stage1"]["centres"]:
        cv2.drawMarker(out, (int(x), int(y)), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
    for x, y in (report["stage2"].get("centres") or []):
        cv2.circle(out, (int(x), int(y)), 4, (0, 255, 255), -1)
    for x, y in G.undistort_points(G.grid_points(nx=17, ny=10), report["theta"]):
        cv2.circle(out, (int(x), int(y)), 1, (0, 255, 0), -1)
    cv2.putText(out, f"{report['verdict'].upper()}  corner "
                f"{report['quality']['corner_px']:.2f} px", (12, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), out, [cv2.IMWRITE_JPEG_QUALITY, 85])


def _print(name, frame, r):
    q, s2 = r["quality"], r["stage2"]
    print(f"{name}   {frame.shape[1]}x{frame.shape[0]}")
    if r["net"]:
        t = r["net"]["theta"]
        print(f"net        k1 {t['k1']:+.4f} k2 {t['k2']:+.4f} "
              f"cx {t['cx']:.1f} cy {t['cy']:.1f}   (start + guard)")
    print(f"stage1     holes {r['stage1']['n_holes']}  spread {r['stage1']['spread_pct']:.2f}%"
          f"  step {r['stage1']['step_px']:.1f} px = {D.HOLE_SPACING_MM:.0f} mm"
          f" -> {r['mm_per_px_panel']:.4f} mm/px")
    e = r["edges"]
    print(f"edges      top {e['top']} bottom {e['bottom']} left {e['left']} "
          f"right {e['right']} strip {e['band']}   rms {r['line_rms_all']} px")
    if s2.get("n_rings"):
        print(f"stage2     rings {s2['n_rings']} (seeds {s2.get('n_seeds')})  "
              f"ratio {s2.get('ratio')}  " +
              (f"USED step {s2['step_px']:.0f} px" if s2["used"]
               else "not used (did not lower corner error)"))
    else:
        print("stage2     no rings found - skipped")
    t = r["theta"]
    print(f"theta      k1 {t['k1']:+.4f}  k2 {t['k2']:+.4f}  "
          f"cx {t['cx']:.1f}  cy {t['cy']:.1f}")
    print(f"quality    corner {q['corner_px']:.2f} px (stage1 {q['corner_stage1_px']:.2f})"
          f"  residual {q['max_residual_px']:.2f}  symmetry {q['symmetry_px']:.1f}")
    if r["net"]:
        print(f"guard      net differs by {r['net']['diff_px']:.1f} px "
              f"(threshold {r['net']['threshold_px']})")
    print(f"verdict    {r['verdict'].upper()}")
    for c in r["checks"]:
        print(f"           [{c['level']}] {c['message']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frame"); ap.add_argument("--device")
    ap.add_argument("--out"); ap.add_argument("--overlay", action="store_true")
    ap.add_argument("--no-net", action="store_true")
    ap.add_argument("--no-rings", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    frame = cv2.imread(a.frame)
    if frame is None:
        sys.exit(f"cannot read frame: {a.frame}")
    r = calibrate(frame, use_net=not a.no_net, use_rings=not a.no_rings)
    name = a.device or Path(a.frame).stem

    if "theta" not in r:
        print(f"{name}: REJECT - {r['reason']}")
        sys.exit(2)
    if not a.quiet:
        _print(name, frame, r)

    out = Path(a.out) if a.out else D.MEASURE_DIR / f"{name}.json"
    if r["verdict"] != "reject":
        P.write(out, r["theta"], device=name, stage1=r["stage1"], stage2=r["stage2"],
                quality=r["quality"], net=r["net"],
                guard={"verdict": r["verdict"], "checks": r["checks"]},
                line_rms=r["line_rms"], mm_per_px_panel=r["mm_per_px_panel"],
                roll_deg=r["roll_deg"])
        if not a.quiet:
            print(f"profile    {out}")
    elif not a.quiet:
        print("profile    NOT WRITTEN (reject)")
    if a.overlay:
        draw(frame, r, D.OVERLAY_DIR / f"{name}_calib.jpg")
    if not a.quiet:
        print(f"time       {r['ms']/1000:.2f} s")
    sys.exit(0 if r["verdict"] != "reject" else 2)


if __name__ == "__main__":
    main()
