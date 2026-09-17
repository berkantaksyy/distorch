"""distorch - hole-assisted automatic lens calibration.

Single entry point: `python -m distorch.calibrate frame.jpg`.
All fixed paths and physical constants are declared here; no module computes
its own Path(__file__).
"""
from pathlib import Path

__version__ = "1.0"

ROOT = Path(__file__).resolve().parent.parent

PKG_DIR = ROOT / "distorch"
WEIGHTS_DIR = ROOT / "weights"
DATA_DIR = ROOT / "data"
FRAMES_DIR = DATA_DIR / "frames"
FRAMES_UNDISTORTED_DIR = DATA_DIR / "frames_undistorted"
REFERENCE_DIR = DATA_DIR / "reference"
MEASURE_DIR = ROOT / "measurements"
OVERLAY_DIR = MEASURE_DIR / "ovl"
TESTS_DIR = ROOT / "tests"

# distort_v2, not v3. v3 took its field labels from a single device
# (ACO_6600_0003) and its centre collapsed: cy std 7.5 px where the fleet really
# varies 33.2 px, and a -46.4 px systematic offset. v2 never saw these frames and
# still tracks the fleet (cy std 23.3, offset +7.7 px). Measured over 104 frames,
# bias-corrected hole spread: v2 1.82%, v3 4.31%.
NET_WEIGHTS = WEIGHTS_DIR / "distort_v2.pt"
NET_META = WEIGHTS_DIR / "distort_v2_meta.json"
NET_BIAS = WEIGHTS_DIR / "distort_v2_bias.json"
PLINEBIG_JSON = REFERENCE_DIR / "plinebig.json"
PANEL_FLAT_JSON = REFERENCE_DIR / "panel_flat.json"
PLUMBLINE_0003_JSON = REFERENCE_DIR / "0003_plumbline.json"
SCALE_JSON = REFERENCE_DIR / "scale.json"      # mm_per_px_object, measured later

# Panel geometry, from the manufacturing drawing.
N_HOLES = 7
HOLE_SPACING_MM = 80.0

# Image contract. f is fixed: straightness cannot observe focal length, and the
# k coefficients absorb the choice. 1920 keeps us comparable with old profiles.
CANON_SIZE = (1920, 1080)
CANON_F = 1920.0

for _d in (FRAMES_DIR, FRAMES_UNDISTORTED_DIR, REFERENCE_DIR, MEASURE_DIR, OVERLAY_DIR):
    _d.mkdir(parents=True, exist_ok=True)
