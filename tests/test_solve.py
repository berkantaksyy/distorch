"""solve.py and quality.py acceptance tests.  python -m tests.test_solve"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, cv2
import distorch as D
from distorch import geom as G, holes as H, edges as E, solve as S, quality as Q

TRUTH = {"k1": -1.30, "k2": 1.50, "cx": 980.0, "cy": 540.0, "f": 1920.0}


def _synthetic():
    x = np.linspace(300, 1600, 7)
    row = np.column_stack([x, np.full(7, 720.0)])
    lines = [np.column_stack([np.linspace(250, 1650, 60), np.full(60, y)])
             for y in (300.0, 760.0)]
    lines += [np.column_stack([np.full(40, x0), np.linspace(320, 740, 40)])
              for x0 in (280.0, 1640.0)]
    return [G.distort_points(l, TRUTH) for l in lines], G.distort_points(row, TRUTH)


def _real(name="ACO_6600_0003"):
    g = cv2.imread(str(D.FRAMES_DIR / f"{name}.jpg"), 0)
    found = H.find(g)
    _, filled, _ = H.panel_mask(g.astype(np.float32))
    return E.usable_lines(E.extract(g, filled)), H.centres(found), found


def test_synthetic_recovery():
    lines, row = _synthetic()
    r = S.solve(lines, [row], theta0=S.THETA0_FIXED)
    d = G.displacement(r["theta"], TRUTH)["mean"]
    assert d < 0.01, f"synthetic recovery {d:.4f} px > 0.01"
    print(f"  synthetic recovery {d:.2e} px")


def test_scale():
    lines, row = _synthetic()
    r = S.solve(lines, [row])
    step = r["rows"][0]["step_px"]
    expected = float(np.diff(np.linspace(300, 1600, 7))[0])
    assert abs(step - expected) < 0.05
    assert abs(r["mm_per_px_panel"] - D.HOLE_SPACING_MM / expected) < 1e-6
    print(f"  scale: step {step:.3f} px = {D.HOLE_SPACING_MM} mm "
          f"-> {r['mm_per_px_panel']:.4f} mm/px")


def test_two_point_row_carries_nothing():
    """An n-point row carries 2n-4 information; n=2 carries none, so it is skipped."""
    lines, row = _synthetic()
    full = S.solve(lines, [row])
    pair = S.solve(lines, [row[:2]])
    assert len(pair["rows"]) == 0
    assert len(full["rows"]) == 1
    print("  two-point row is ignored by the solver")


def test_real_frame():
    lines, row, found = _real()
    assert len(found) == 7
    r = S.solve(lines, [row])
    corner = Q.corner_error(lines, [row], r["theta"])
    assert r["spread_pct"] < Q.SPREAD_REJECT and corner < Q.CORNER_REJECT
    print(f"  real frame: spread {r['spread_pct']:.2f}%  corner {corner:.2f} px  "
          f"line {r['line_rms_all']:.2f} px")


def test_corruption_is_rejected():
    """A hole shifted by 8 px must be REJECTED by the gates."""
    lines, row, _ = _real()
    for k, d in ((3, 8.0), (1, 8.0), (5, 12.0)):
        bad = row.copy()
        bad[k, 0] += d
        r = S.solve(lines, [bad])
        g = Q.gates(r["theta"], Q.corner_error(lines, [bad], r["theta"]),
                    r["spread_pct"], r["line_rms_all"],
                    Q.max_residual([bad], r["theta"]), Q.symmetry(bad))
        assert g["verdict"] == "reject", f"hole {k} +{d} px not caught: {g}"
        print(f"  hole {k} +{d:.0f} px -> REJECT  (spread {r['spread_pct']:.2f}%)")


if __name__ == "__main__":
    for fn in (test_synthetic_recovery, test_scale, test_two_point_row_carries_nothing,
               test_real_frame, test_corruption_is_rejected):
        print(fn.__name__); fn()
    print("\ntest_solve: ALL PASSED")
