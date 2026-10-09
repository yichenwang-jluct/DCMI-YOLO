"""
Evaluate one checkpoint on the test split with the paper's settings and print every
metric from the SAME val() call.

    python val.py --weights runs/train/<run>/weights/best.pt --data data/inhouse.yaml

Settings (Section IV-A2): confidence 0.001, NMS IoU 0.7, at most 300 detections per
image, square 640x640 letterboxing. Precision and recall are read at the confidence that
maximizes mean F1 (Ultralytics); AP and mAP integrate the whole precision-recall curve.

Sanity check printed at the end: for any operating point (R, P) on an interpolated
precision-recall curve, AP@0.5 >= P * R, so r = P * R / mAP@0.5 must not exceed 1.
"""

from __future__ import annotations

import argparse
import math


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="data/inhouse.yaml")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()

    from ultralytics import YOLO

    res = YOLO(a.weights).val(data=a.data, split=a.split, imgsz=a.imgsz, batch=a.batch,
                              conf=a.conf, iou=a.iou, max_det=a.max_det,
                              device=a.device, plots=False, verbose=True)
    b = res.box
    P, R, m50, m = float(b.mp), float(b.mr), float(b.map50), float(b.map)
    r = P * R / m50 if m50 else float("nan")

    print(f"\n  Precision      {P:.4f}")
    print(f"  Recall         {R:.4f}")
    print(f"  mAP@0.5        {m50:.4f}")
    print(f"  mAP@0.5:0.95   {m:.4f}")
    print(f"\n  r = P*R/mAP@0.5 = {r:.3f}   [{'OK' if r <= 1.0 else 'CHECK'}]")

    sp = getattr(res, "speed", {}) or {}
    if sp:
        tot = sp.get("inference", math.nan) + sp.get("postprocess", 0.0)
        print(f"\n  inference {sp.get('inference', float('nan')):.2f} ms + NMS "
              f"{sp.get('postprocess', float('nan')):.2f} ms per image")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
