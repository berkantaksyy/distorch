"""End-to-end acceptance tests.  python -m tests.test_pipeline"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2
import distorch as D
from distorch import calibrate as C, profile as P, geom as G, quality as Q

FRAMES = ["ACO_6600_0003", "ACO_6600_0015", "ACO_ANKA_0045",
          "ACO_ANKA_0030", "ACO_ANKA_0047"]


def _frame(name):
    return cv2.imread(str(D.FRAMES_DIR / f"{name}.jpg"))


def test_five_devices():
    for name in FRAMES:
        r = C.calibrate(_frame(name))
        assert r["verdict"] != "reject", f"{name}: {r.get('checks')}"
        assert r["quality"]["corner_px"] < Q.CORNER_REJECT
        assert 0.30 < r["mm_per_px_panel"] < 0.35
        print(f"  {name:16s} corner {r['quality']['corner_px']:5.2f} px  "
              f"mm/px {r['mm_per_px_panel']:.4f}  {r['verdict']}")


def test_deterministic():
    for name in FRAMES[:3]:
        a = C.calibrate(_frame(name))["theta"]
        b = C.calibrate(_frame(name))["theta"]
        for k in ("k1", "k2", "cx", "cy"):
            assert a[k] == b[k], f"{name}.{k}: {a[k]} != {b[k]}"
    print("  same frame -> bit-identical theta (3 devices)")


def test_speed():
    C.calibrate(_frame(FRAMES[0]))          # warm the model load
    t = time.perf_counter()
    for name in FRAMES:
        C.calibrate(_frame(name))
    dt = (time.perf_counter() - t) / len(FRAMES)
    assert dt < 2.0, f"{dt:.2f} s/frame > 2.0"
    print(f"  {dt:.2f} s/frame (target < 2.0)")


def test_profile_roundtrip():
    name = FRAMES[0]
    r = C.calibrate(_frame(name))
    path = D.MEASURE_DIR / "_roundtrip.json"
    P.write(path, r["theta"], device=name, stage1=r["stage1"], stage2=r["stage2"],
            quality=r["quality"], net=r["net"], line_rms=r["line_rms"],
            mm_per_px_panel=r["mm_per_px_panel"])
    theta, data = P.read(path)
    assert G.displacement(theta, r["theta"])["max"] < 1e-9
    assert data["metric_valid"] is True and data["fx_assumed"] is True
    assert data["image_size"] == [1920, 1080]
    path.unlink()
    print("  profile write/read exact · metric_valid True")


def test_old_profile_reads():
    theta, data = P.read(D.PLINEBIG_JSON)
    assert abs(theta["k1"] - G.THETA_PLINEBIG["k1"]) < 1e-6
    assert data["metric_valid"] is False        # old profiles carry no scale
    print("  old profile (plinebig) reads, metric_valid False")


def test_stage2_never_hurts():
    for name in FRAMES:
        r = C.calibrate(_frame(name))
        assert r["quality"]["corner_px"] <= r["quality"]["corner_stage1_px"] + 1e-9, name
    print("  stage 2 did not degrade any frame")


if __name__ == "__main__":
    for fn in (test_five_devices, test_deterministic, test_speed,
               test_profile_roundtrip, test_old_profile_reads, test_stage2_never_hurts):
        print(fn.__name__); fn()
    print("\ntest_pipeline: ALL PASSED")
