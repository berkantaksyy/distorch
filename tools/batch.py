#!/usr/bin/env python3
"""Calibrate every frame, write the fleet report.

    python -m tools.batch [--overlay]

Outputs: measurements/fleet.jsonl and measurements/fleet_report.md
"""
import sys, json, re, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, cv2
import distorch as D
from distorch import calibrate as C

TEST_DEVICES = {"ACO_6600_0003", "ACO_6600_0015", "ACO_ANKA_0045"}
VAL_DEVICES = {"ACO_6600_0007", "ACO_ANKA_0033"}


def device_of(name):
    m = re.match(r"(ACO_ANKA_\d{4}|ACO_6600_\d{4}|ACO-ANKA-A1100-\d{4})(_\w+)?$", name)
    return m.group(1) if m else name


def run(overlay=False):
    rows, t0 = [], time.time()
    for p in sorted(D.FRAMES_DIR.glob("*.jp*g")):
        frame = cv2.imread(str(p))
        r = C.calibrate(frame)
        if "theta" not in r:
            rows.append({"frame": p.stem, "verdict": "reject", "reason": r.get("reason")})
            continue
        if overlay:
            C.draw(frame, r, D.OVERLAY_DIR / f"{p.stem}_calib.jpg")
        rows.append({
            "frame": p.stem, "device": device_of(p.stem), "verdict": r["verdict"],
            **{k: r["theta"][k] for k in ("k1", "k2", "cx", "cy")},
            "corner": r["quality"]["corner_px"], "corner1": r["quality"]["corner_stage1_px"],
            "spread": r["stage1"]["spread_pct"], "line_rms": r["line_rms_all"],
            "mm_per_px": r["mm_per_px_panel"], "n_holes": r["stage1"]["n_holes"],
            "n_rings": r["stage2"].get("n_rings", 0), "rings_used": r["stage2"]["used"],
            "ratio": r["stage2"].get("ratio"),
            "net_diff": (r["net"] or {}).get("diff_px"),
            "checks": [c["check"] for c in r["checks"]]})
    with open(D.MEASURE_DIR / "fleet.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return rows, time.time() - t0


def report(rows, seconds):
    ok = [r for r in rows if r["verdict"] != "reject"]
    col = lambda k: np.array([r[k] for r in ok if r.get(k) is not None], float)
    devices = {}
    for r in ok:
        devices.setdefault(r["device"], []).append(r)

    out = ["# distorch - Filo Raporu\n",
           f"{len(rows)} kare · {len(devices)} cihaz · "
           f"{time.strftime('%Y-%m-%d %H:%M')} · {seconds/len(rows):.2f} s/kare\n",
           "## Ozet\n", "| olcut | medyan | ort | p90 | en kotu |", "|---|---|---|---|---|"]
    for label, key in (("kose hatasi (px)", "corner"), ("delik yayilimi (%)", "spread"),
                       ("cizgi RMS (px)", "line_rms"), ("mm/px", "mm_per_px")):
        v = col(key)
        out.append(f"| {label} | {np.median(v):.3f} | {v.mean():.3f} | "
                   f"{np.percentile(v, 90):.3f} | {v.max():.3f} |")
    tally = {}
    for r in ok:
        tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
    out.append("\nKarar: " + " · ".join(f"**{v} {k}**" for k, v in sorted(tally.items())))
    used = [r for r in ok if r["rings_used"]]
    if used:
        out.append(f"\nBilezik (Asama 2) kullanilan: **{len(used)}/{len(ok)}** kare "
                   f"(%{len(used)/len(ok)*100:.0f}). Bu karelerde kose hatasi "
                   f"{np.median([r['corner1'] for r in used]):.2f} -> "
                   f"{np.median([r['corner'] for r in used]):.2f} px.")
    out += ["\n## Cihaz basina\n",
            "| cihaz | n | bolum | k1 | cx | mm/px | kose px | yayilim % |",
            "|---|---|---|---|---|---|---|---|"]
    for dev, v in sorted(devices.items()):
        part = "TEST" if dev in TEST_DEVICES else ("VAL" if dev in VAL_DEVICES else "train")
        g = lambda k: np.array([x[k] for x in v], float)
        out.append(f"| {dev} | {len(v)} | {part} | {g('k1').mean():+.3f}±{g('k1').std():.3f} "
                   f"| {g('cx').mean():.1f}±{g('cx').std():.2f} "
                   f"| {g('mm_per_px').mean():.4f}±{g('mm_per_px').std():.4f} "
                   f"| {np.median(g('corner')):.2f} | {np.median(g('spread')):.2f} |")
    fired = {}
    for r in ok:
        for c in r["checks"]:
            fired[c] = fired.get(c, 0) + 1
    out += ["\n## Kapi tetiklenmeleri\n", "| kapi | kare |", "|---|---|"]
    for c, n in sorted(fired.items(), key=lambda z: -z[1]):
        out.append(f"| {c} | {n} |")
    text = "\n".join(out)
    (D.MEASURE_DIR / "fleet_report.md").write_text(text)
    return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--overlay", action="store_true")
    a = ap.parse_args()
    rows, secs = run(a.overlay)
    print(report(rows, secs)[:1200])
    print(f"\n-> measurements/fleet_report.md  ·  {secs:.0f} s")
