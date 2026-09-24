"""
Re-export Precision, Recall, mAP@0.5 and mAP@0.5:0.95 so that the four numbers in
each table row come from ONE validation run, and check them against the geometric
constraint that AP >= P x R.

    python export_metrics.py --data data/inhouse.yaml --split test --out metrics_inhouse.csv

Why this is needed
------------------
Ultralytics reports P and R at the confidence that maximises mean F1, while AP
integrates the whole precision-recall curve. For every class, the interpolated AP
is at least the area of the rectangle under any point of its own curve, so

    AP50_c  >=  P_c * R_c        (per class)
    mAP@0.5 >=  P  * R           (in practice, unless P and R are strongly
                                  negatively correlated across classes)

A row that breaks this means P/R and mAP did not come from the same run, the same
confidence threshold, or the same split. This script makes that impossible and
flags any residual violation per class.

It also reports FPS from the same run, so the ablation rows stay additive in
latency (1/FPS), which the current Table 8 does not.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# EDIT THIS: paper row label -> {seed: checkpoint}. Use the SAME seeds for every
# row. If you only kept one seed for some rows, retrain or drop the +/- SD claim
# for those rows rather than mixing seed counts.
# ---------------------------------------------------------------------------
CONFIGS = {
    "YOLOv8n (baseline)": {
        0:    "runs/ablation/base_s0/weights/best.pt",
        42:   "runs/ablation/base_s42/weights/best.pt",
        2024: "runs/ablation/base_s2024/weights/best.pt",
    },
    "DCMI-YOLO (full)": {
        0:    "runs/ablation/full_s0/weights/best.pt",
        42:   "runs/ablation/full_s42/weights/best.pt",
        2024: "runs/ablation/full_s2024/weights/best.pt",
    },
    # ... add every row of Tables 5-7, 11, 12, 13, 16-19 here
}


def run_val(weights, data, split, imgsz, batch, conf, iou, max_det, device):
    from ultralytics import YOLO

    model = YOLO(str(weights))
    res = model.val(
        data=data, split=split, imgsz=imgsz, batch=batch,
        conf=conf, iou=iou, max_det=max_det,
        device=device, plots=False, verbose=False, save_json=False,
    )
    b = res.box
    sp = dict(res.speed)  # milliseconds per image
    inf = float(sp.get("inference", 0.0))
    total = inf + float(sp.get("preprocess", 0.0)) + float(sp.get("postprocess", 0.0))

    # per-class check: AP@0.5 must not fall below P x R at the reported operating point
    violations = []
    try:
        names = getattr(res, "names", {}) or {}
        for i, ci in enumerate(b.ap_class_index):
            ap50 = float(b.all_ap[i, 0])
            p, r = float(b.p[i]), float(b.r[i])
            if ap50 + 1e-6 < p * r:
                violations.append(f"{names.get(int(ci), int(ci))}(AP50={ap50:.3f}<P*R={p*r:.3f})")
    except Exception:  # noqa: BLE001
        pass

    return dict(
        P=float(b.mp), R=float(b.mr),
        mAP50=float(b.map50), mAP50_95=float(b.map),
        ms_inference=round(inf, 2), ms_total=round(total, 2),
        fps_inference=round(1000.0 / inf, 1) if inf else float("nan"),
        fps_end_to_end=round(1000.0 / total, 1) if total else float("nan"),
        violations="; ".join(violations),
    )


def agg(values):
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset yaml")
    ap.add_argument("--split", default="test", choices=["val", "test"],
                    help="use ONE split per table; never mix val and test numbers in one table")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.001, help="must be identical for every row")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU, identical for every row")
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="metrics.csv")
    args = ap.parse_args()

    per_run, summary = [], []
    for label, seeds in CONFIGS.items():
        print(f"\n[*] {label}")
        runs = []
        for seed, weights in seeds.items():
            if not Path(weights).exists():
                print(f"    ! seed {seed}: {weights} not found, skipped")
                continue
            row = run_val(weights, args.data, args.split, args.imgsz, args.batch,
                          args.conf, args.iou, args.max_det, args.device)
            row.update(config=label, seed=seed, split=args.split, weights=weights)
            runs.append(row)
            per_run.append(row)
            print(f"    seed {seed:<5} P {row['P']:.3f}  R {row['R']:.3f}  "
                  f"mAP50 {row['mAP50']:.3f}  mAP50-95 {row['mAP50_95']:.3f}  "
                  f"FPS {row['fps_inference']}")
            if row["violations"]:
                print(f"      ! per-class AP50 < P*R: {row['violations']}")

        if not runs:
            continue

        out = dict(config=label, n_seeds=len(runs), split=args.split)
        for key in ("P", "R", "mAP50", "mAP50_95", "fps_inference", "fps_end_to_end", "ms_inference"):
            m, s = agg([r[key] for r in runs])
            out[key] = round(m, 4 if key.startswith(("P", "R", "mAP")) else 2)
            out[key + "_sd"] = round(s, 4 if key.startswith(("P", "R", "mAP")) else 2)

        # table-level sanity check
        pr = out["P"] * out["R"]
        out["check_mAP50_ge_PxR"] = "OK" if out["mAP50"] >= pr - 1e-6 else f"FAIL (P*R={pr:.3f})"
        summary.append(out)
        print(f"    mean   P {out['P']:.3f}+-{out['P_sd']:.3f}  R {out['R']:.3f}+-{out['R_sd']:.3f}  "
              f"mAP50 {out['mAP50']:.3f}+-{out['mAP50_sd']:.3f}  "
              f"mAP50-95 {out['mAP50_95']:.3f}+-{out['mAP50_95_sd']:.3f}   [{out['check_mAP50_ge_PxR']}]")

    if not summary:
        print("Nothing evaluated - check the paths in CONFIGS.")
        return 1

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    detail = Path(args.out).with_name(Path(args.out).stem + "_per_run.csv")
    with open(detail, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_run[0]))
        w.writeheader()
        w.writerows(per_run)

    failed = [s["config"] for s in summary if s["check_mAP50_ge_PxR"] != "OK"]
    print(f"\nWritten to {args.out} and {detail}")
    if failed:
        print("\n!! These rows still break mAP50 >= P x R after a single-run export:")
        for f in failed:
            print(f"   - {f}")
        print("   Check that every row used the same --conf, --iou and --split.")
    else:
        print("\nAll rows satisfy mAP50 >= P x R.")

    lat = {s["config"]: s["ms_inference"] for s in summary}
    print("\nLatency per image (ms) - use these to check that Table 8 is additive:")
    for k, v in lat.items():
        print(f"   {k:<30} {v:.2f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
