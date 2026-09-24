"""
Measure FPS for every configuration under one protocol, and check that the
per-module latencies add up.

    python bench_fps.py --imgsz 640 --out fps.csv

Reports three numbers per model, because "FPS" in a paper is ambiguous:

  fps_model   pure forward pass of the network (no letterbox, no NMS)
  fps_nms     forward pass + non-maximum suppression
  fps_e2e     the full Ultralytics predict pipeline (load, letterbox, forward,
              NMS, scale boxes) - this is what a deployed system sees

Whichever you report, report the SAME one for every row and say so in Section 4.1.2.

Protocol notes baked in below, all of which change the answer by tens of percent
if you get them wrong: fixed batch size, fixed precision, warm-up iterations
before timing, torch.cuda.synchronize() around the timed region, and a median
over many repeats rather than a mean over a few.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# EDIT THIS: the paper's row label -> checkpoint. Keep the first entry as the
# baseline; the additivity check is computed against it.
# ---------------------------------------------------------------------------
CONFIGS = [
    ("YOLOv8n (baseline)",      "runs/ablation/base/weights/best.pt"),
    ("+ SCI-Gamma",             "runs/ablation/sci/weights/best.pt"),
    ("+ Star-ADown",            "runs/ablation/adown/weights/best.pt"),
    ("+ MultiSEAM",             "runs/ablation/seam/weights/best.pt"),
    ("SCI-Gamma + Star-ADown",  "runs/ablation/sci_adown/weights/best.pt"),
    ("SCI + ADown + MultiSEAM", "runs/ablation/sci_adown_seam/weights/best.pt"),
    ("DCMI-YOLO (full)",        "runs/ablation/full/weights/best.pt"),
]


def timed(fn, warmup: int, repeat: int, sync) -> float:
    """Median milliseconds per call."""
    for _ in range(warmup):
        fn()
    sync()
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        sync()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def bench(weights, imgsz, batch, device, half, warmup, repeat, sample_image):
    import torch
    from ultralytics import YOLO

    y = YOLO(str(weights))
    model = y.model.to(device).eval()
    if half:
        model = model.half()
    sync = torch.cuda.synchronize if str(device).startswith("cuda") else (lambda: None)

    x = torch.zeros(batch, 3, imgsz, imgsz, device=device,
                    dtype=torch.half if half else torch.float)

    with torch.inference_mode():
        ms_model = timed(lambda: model(x), warmup, repeat, sync)

        from ultralytics.utils import ops
        def fwd_nms():
            out = model(x)
            pred = out[0] if isinstance(out, (list, tuple)) else out
            ops.non_max_suppression(pred, conf_thres=0.001, iou_thres=0.7, max_det=300)
        ms_nms = timed(fwd_nms, warmup, repeat, sync)

    # full pipeline on a real image, through the public API
    ms_e2e = float("nan")
    if sample_image and Path(sample_image).exists():
        import cv2
        img = cv2.imread(str(sample_image))
        if img is not None:
            ms_e2e = timed(
                lambda: y.predict(img, imgsz=imgsz, device=device, half=half,
                                  conf=0.001, iou=0.7, max_det=300, verbose=False),
                warmup, repeat, sync)

    per_img = lambda ms: ms / batch  # noqa: E731
    return dict(
        ms_model=round(per_img(ms_model), 3),
        ms_nms=round(per_img(ms_nms), 3),
        ms_e2e=round(ms_e2e, 3),
        fps_model=round(1000.0 / per_img(ms_model), 1),
        fps_nms=round(1000.0 / per_img(ms_nms), 1),
        fps_e2e=round(1000.0 / ms_e2e, 1) if ms_e2e == ms_e2e else float("nan"),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=1, help="the paper reports batch = 1")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--half", action="store_true", help="FP16; use the same setting for every row")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--repeat", type=int, default=300)
    ap.add_argument("--image", default="", help="a real test image, for the end-to-end number")
    ap.add_argument("--out", default="fps.csv")
    args = ap.parse_args()

    rows = []
    for label, path in CONFIGS:
        if not Path(path).exists():
            print(f"[!] {label}: {path} not found, skipped")
            continue
        print(f"[*] {label}")
        r = bench(path, args.imgsz, args.batch, args.device, args.half,
                  args.warmup, args.repeat, args.image)
        r.update(config=label)
        rows.append(r)
        print(f"    model {r['ms_model']:.2f} ms ({r['fps_model']} FPS) | "
              f"+NMS {r['ms_nms']:.2f} ms ({r['fps_nms']} FPS) | "
              f"e2e {r['ms_e2e']} ms ({r['fps_e2e']} FPS)")

    if not rows:
        print("Nothing benchmarked - check the paths in CONFIGS.")
        return 1

    fields = ["config", "ms_model", "ms_nms", "ms_e2e", "fps_model", "fps_nms", "fps_e2e"]
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows({k: r[k] for k in fields} for r in rows)
    print(f"\nWritten to {args.out}")

    # --- additivity check on the metric you will report -----------------------
    MODULES = {"sci": "SCI-Gamma", "adown": "Star-ADown", "seam": "MultiSEAM"}
    by = {r["config"]: r for r in rows}
    base_label = rows[0]["config"]
    base = by[base_label]["ms_nms"]

    def present(cfg):
        c = cfg.lower()
        return {k for k in MODULES if k in c}

    singles = {}
    for r in rows:
        mods = present(r["config"])
        if len(mods) == 1:
            singles[next(iter(mods))] = r["ms_nms"] - base

    if singles:
        print(f"\nPer-module latency, measured alone (vs '{base_label}', +NMS timing):")
        for k, v in singles.items():
            print(f"   {MODULES[k]:<26} {v:+.2f} ms")
        print("\nCombined rows: measured vs the sum of the parts")
        for r in rows:
            mods = present(r["config"])
            if len(mods) < 2 or not mods <= set(singles):
                continue
            pred = base + sum(singles[m] for m in mods)
            gap = r["ms_nms"] - pred
            tag = "OK" if abs(gap) <= 0.05 * pred else "MISMATCH"
            print(f"   {r['config']:<26} measured {r['ms_nms']:6.2f}  predicted {pred:6.2f}  "
                  f"gap {gap:+.2f} ms  [{tag}]")
        print("\nA gap beyond about 5 % means one of the rows was not measured "
              "under this protocol; re-run that row rather than explaining the gap.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
