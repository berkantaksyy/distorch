"""net.py acceptance tests (the guard CNN).  python -m tests.test_net

The only job here is to load the CNN and run it correctly. A failing
reproduction test means the preprocessing chain is broken.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2
import distorch as D
from distorch import net as N, geom as G

# distort_v2's RAW output on ACO_6600_0003, pinned the day the model was
# adopted. It is a fingerprint of the preprocessing chain, not a quality claim:
# if this drifts, the grey / 960x540 / 512x288 / ImageNet chain changed.
NET_0003 = {"k1": -1.1123, "k2": 0.9766, "cx": 978.7132, "cy": 559.5696, "f": 1920.0}


def _frame(name="ACO_6600_0003"):
    return cv2.imread(str(D.FRAMES_DIR / f"{name}.jpg"))


def test_load():
    ok, status = N.load()
    assert ok, f"model did not load: {status}"
    meta = N.meta()
    assert tuple(meta["input_size"]) == N.INPUT
    assert meta.get("version") == "distort_v2"
    print(f"  {status} · input {meta['input_size']} · anchors {meta['anchors']}")


def test_reproduction():
    theta = N.theta_from_frame(_frame(), correct_bias=False)
    d = G.displacement(theta, NET_0003)["mean"]
    assert d < 0.5, f"reproduction {d:.3f} px > 0.5 - preprocessing chain broken"
    print(f"  reproduction {d:.4f} px")


def test_anchor_convention():
    t0 = N.meta()["theta0"]
    m = G.mags_from_kk(t0["k1"], t0["k2"])
    ref = N.meta()["envelope"]["m_ref"]
    assert abs(m[0] - ref[0]) < 1e-4 and abs(m[1] - ref[1]) < 1e-4, f"{m} != {ref}"
    k = G.kk_from_mags(*m)
    assert abs(k[0] - t0["k1"]) < 1e-4 and abs(k[1] - t0["k2"]) < 1e-4
    print(f"  anchors {tuple(round(x, 6) for x in m)} match meta m_ref")


def test_bias_correction():
    b = N.bias()
    assert abs(b["cy"]) > 1.0, "bias file not read (all zeros)"
    raw = N.theta_from_frame(_frame(), correct_bias=False)
    fixed = N.theta_from_frame(_frame(), correct_bias=True)
    assert G.displacement(raw, fixed)["mean"] > 1.0, "bias not applied"
    print(f"  bias m1 {b['m1']:+.4f} m2 {b['m2']:+.4f} "
          f"cx {b['cx']:+.1f} cy {b['cy']:+.1f} px")


def test_envelope():
    for name in ("ACO_6600_0003", "ACO_ANKA_0030", "ACO_ANKA_0047"):
        if not (D.FRAMES_DIR / f"{name}.jpg").exists():
            continue
        ok, note = N.envelope_ok(N.theta_from_frame(_frame(name), correct_bias=False))
        assert ok, f"{name} outside envelope: {note}"
    print("  sample frames sit inside the training envelope")


if __name__ == "__main__":
    for fn in (test_load, test_reproduction, test_anchor_convention,
               test_bias_correction, test_envelope):
        print(fn.__name__); fn()
    print("\ntest_net: ALL PASSED")
