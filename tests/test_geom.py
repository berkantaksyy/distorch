"""geom.py acceptance tests.  python -m tests.test_geom"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from distorch import geom as G

THETAS = [("plinebig", G.THETA_PLINEBIG),
          ("fleet_mean", dict(k1=-1.350, k2=1.910, cx=993.1, cy=583.7, f=1920.0)),
          ("plumb0003", dict(k1=-1.1596, k2=1.1268, cx=972.05, cy=523.41, f=1920.0)),
          ("aggressive", dict(k1=-1.60, k2=2.60, cx=960.0, cy=540.0, f=1920.0))]


def test_roundtrip():
    rng = np.random.default_rng(0)
    p = rng.uniform([0, 0], [1920, 1080], (20000, 2))
    for name, theta in THETAS:
        err = float(np.abs(G.distort_points(G.undistort_points(p, theta), theta) - p).max())
        assert err < 0.01, f"{name}: roundtrip {err:.3e} px > 0.01"
        print(f"  roundtrip {name:12s} {err:.2e} px")


def test_anchors():
    for name, theta in THETAS:
        m1, m2 = G.mags_from_kk(theta["k1"], theta["k2"])
        k1, k2 = G.kk_from_mags(m1, m2)
        assert abs(k1 - theta["k1"]) < 1e-9 and abs(k2 - theta["k2"]) < 1e-9, name
    print("  anchor roundtrip exact")


def test_validity():
    ok, notes = G.is_valid(G.THETA_PLINEBIG)
    assert ok, notes
    assert G.black_border(G.THETA_PLINEBIG) == 0
    assert not G.is_monotonic(-3.0, 0.0, 1.0)
    print("  validity / monotonicity / black border")


def test_displacement():
    a, b = THETAS[0][1], THETAS[1][1]
    assert abs(G.displacement(a, b)["mean"] - G.displacement(b, a)["mean"]) < 1e-9
    assert G.displacement(a, a)["max"] < 1e-12
    print(f"  displacement symmetric, {G.displacement(a, b)['mean']:.2f} px apart")


def test_centre_fixed():
    for name, theta in THETAS:
        c = np.array([[theta["cx"], theta["cy"]]])
        assert np.abs(G.distort_points(c, theta) - c).max() < 1e-9
        assert np.abs(G.undistort_points(c, theta) - c).max() < 1e-9
    print("  centre stays fixed")


if __name__ == "__main__":
    for fn in (test_roundtrip, test_anchors, test_validity, test_displacement,
               test_centre_fixed):
        print(fn.__name__); fn()
    print("\ntest_geom: ALL PASSED")
