"""holes.py and edges.py acceptance tests.  python -m tests.test_holes"""
import sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, cv2
import distorch as D
from distorch import holes as H, edges as E

PATTERN = np.array([170.6, 211.3, 242.2, 240.9, 209.8, 168.4])   # measured, 106 frames


def _frames(folder):
    return sorted(folder.glob("*.jp*g"))


def test_seven_holes_everywhere():
    missing = []
    for p in _frames(D.FRAMES_DIR) + _frames(D.FRAMES_UNDISTORTED_DIR):
        n = len(H.find(cv2.imread(str(p), 0)))
        if n != D.N_HOLES:
            missing.append((p.name, n))
    assert len(missing) <= 1, f"frames without 7 holes: {missing}"
    total = len(_frames(D.FRAMES_DIR)) + len(_frames(D.FRAMES_UNDISTORTED_DIR))
    print(f"  {total} frames, 7/7" + (f" (exception {missing})" if missing else ""))


def test_gap_pattern():
    gaps = []
    for p in _frames(D.FRAMES_DIR):
        found = H.find(cv2.imread(str(p), 0))
        if len(found) == D.N_HOLES:
            gaps.append(np.diff([h["cx"] for h in found]))
    mean = np.array(gaps).mean(0)
    drift = np.abs(mean - PATTERN).max()
    assert drift < 3.0, f"pattern drifted: {mean.round(1)} vs {PATTERN}"
    print(f"  pattern {mean.round(1)}  (max drift {drift:.2f} px)")


def test_within_device_repeatability():
    devices = {}
    for p in _frames(D.FRAMES_DIR):
        m = re.match(r"(ACO_ANKA_\d{4}|ACO_6600_\d{4})(_\d+)?\.", p.name)
        if not m:
            continue
        found = H.find(cv2.imread(str(p), 0))
        if len(found) == D.N_HOLES:
            devices.setdefault(m.group(1), []).append(H.centres(found))
    worst, who = 0.0, None
    for dev, v in devices.items():
        if len(v) < 4:
            continue
        a = np.array(v)
        s = max(a[:, :, 0].std(0).mean(), a[:, :, 1].std(0).mean())
        if s > worst:
            worst, who = s, dev
    assert worst < 1.5, f"{who}: centre std {worst:.2f} px"
    print(f"  worst within-device centre std {worst:.2f} px ({who})")


def test_undistorted_flag():
    raw = sum(H.is_undistorted(H.find(cv2.imread(str(p), 0))) for p in _frames(D.FRAMES_DIR))
    done = sum(H.is_undistorted(H.find(cv2.imread(str(p), 0)))
               for p in _frames(D.FRAMES_UNDISTORTED_DIR))
    assert raw == 0, f"{raw} raw frames wrongly flagged"
    assert done == len(_frames(D.FRAMES_UNDISTORTED_DIR))
    print(f"  raw 0/{len(_frames(D.FRAMES_DIR))} · corrected {done}/{done}")


def test_edge_lines():
    thin = []
    for p in _frames(D.FRAMES_DIR):
        g = cv2.imread(str(p), 0)
        _, filled, _ = H.panel_mask(g.astype(np.float32))
        n = len(E.usable_lines(E.extract(g, filled)))
        if n < 3:
            thin.append((p.name, n))
    assert not thin, f"fewer than 3 lines: {thin}"
    print("  at least 3 usable edge lines in every frame")


if __name__ == "__main__":
    for fn in (test_seven_holes_everywhere, test_gap_pattern,
               test_within_device_repeatability, test_undistorted_flag, test_edge_lines):
        print(fn.__name__); fn()
    print("\ntest_holes: ALL PASSED")
