#!/usr/bin/env python3
"""Move already-corrected frames out of the raw set.

Test: std of the hole x-gaps < 6 px. A raw frame runs 170..242..168 px
(std ~30); a corrected one is flat at ~217.

    python -m tools.split_undistorted [--apply]
"""
import sys, shutil, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, cv2
import distorch as D
from distorch import holes as H

STD_THR = 6.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually move the files")
    args = ap.parse_args()
    hits = []
    for p in sorted(D.FRAMES_DIR.glob("*.jp*g")):
        found = H.find(cv2.imread(str(p), 0))
        if len(found) < 3:
            print(f"  skipped ({len(found)} holes): {p.name}")
            continue
        std = float(np.std(np.diff([h["cx"] for h in found])))
        if std < STD_THR:
            hits.append((p, std))
    print(f"\n{len(hits)} frames look already corrected (std < {STD_THR} px):")
    for p, std in hits:
        print(f"  {p.name:26s} std {std:.2f}")
    if args.apply:
        for p, _ in hits:
            shutil.move(str(p), str(D.FRAMES_UNDISTORTED_DIR / p.name))
        print(f"\nmoved {len(hits)} -> {D.FRAMES_UNDISTORTED_DIR.name}/")
    else:
        print("\n(dry run - pass --apply to move)")


if __name__ == "__main__":
    main()
