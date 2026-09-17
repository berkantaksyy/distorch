"""Profile JSON. The main project reads this file.

Backwards compatible: camera_matrix and dist_coeffs are unchanged, the rest is
new. Old profiles without `metric_valid` read back as False.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from distorch import CANON_F, CANON_SIZE, HOLE_SPACING_MM, __version__

NOTE = ("Scale was MEASURED from the 80 mm hole spacing of the bright panel, "
        "not assumed. fx is still unobservable from straightness "
        "(fx_assumed=true), but the metric measurement comes from the scale, "
        "not from fx, hence metric_valid=true. The CNN does not decide: the "
        "output is always the geometric solution.")


def write(path, theta, *, device=None, stage1=None, stage2=None, quality=None,
          net=None, guard=None, line_rms=None, mm_per_px_panel=None,
          mm_per_px_object=None, roll_deg=None, size=CANON_SIZE):
    data = {
        "camera_matrix": [[CANON_F, 0.0, theta["cx"]],
                          [0.0, CANON_F, theta["cy"]],
                          [0.0, 0.0, 1.0]],
        "dist_coeffs": [[theta["k1"], theta["k2"], 0.0, 0.0, 0.0]],
        "image_size": list(size),
        "device": device,
        "method": "edges + holes (+ rings) geometric fit, CNN supervised",
        "fx_assumed": True,
        "metric_valid": mm_per_px_panel is not None,
        "mm_per_px_panel": mm_per_px_panel,
        "mm_per_px_object": mm_per_px_object,
        "scale_source": f"bright panel holes, centre to centre {HOLE_SPACING_MM} mm",
        # Mounting roll in degrees. The corrected image is NOT rotated; this is
        # here so the measuring side can undo it on coordinates instead.
        "roll_deg": roll_deg,
        "stage1": stage1, "stage2": stage2,
        "line_rms": line_rms, "quality": quality, "net": net, "guard": guard,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "version": f"distorch-{__version__}",
        "notes": NOTE,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return p


def read(path):
    """-> (theta, full dict). Reads old profiles too."""
    d = json.loads(Path(path).read_text())
    k = d["camera_matrix"]
    c = d["dist_coeffs"][0]
    theta = {"k1": float(c[0]), "k2": float(c[1]),
             "cx": float(k[0][2]), "cy": float(k[1][2]), "f": float(k[0][0])}
    d.setdefault("metric_valid", False)
    d.setdefault("mm_per_px_panel", None)
    d.setdefault("roll_deg", None)
    return theta, d
