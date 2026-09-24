"""
Evaluate one checkpoint and print every metric from the SAME val() call.

    python val.py --weights runs/train/.../weights/best.pt --data data/inhouse.yaml

P, R, mAP@0.5 and mAP@0.5:0.95 are not independent. For any point (R, P) on a
precision-recall curve the interpolated average precision obeys AP@0.5 >= P * R,
because every recall below R carries interpolated precision of at least P. So

    r = (P * R) / mAP@0.5

must be <= 1.00, and for a real detector it lands around 0.85-0.96. A value above
1.00 does not mean the model is unusual - it means the four numbers came from
different runs. Reading them off one result object, as this script does, makes
that impossible.
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
                              device=a.device, plots=False, verbose=False)
    b = res.box
    P, R, m50, m = float(b.mp), float(b.mr), float(b.map50), float(b.map)
    r = P * R / m50 if m50 else float("nan")

    print(f"\n  Precision      {P:.4f}")
    print(f"  Recall         {R:.4f}")
    print(f"  mAP@0.5        {m50:.4f}")
    print(f"  mAP@0.5:0.95   {m:.4f}")
    print(f"\n  r = P*R/mAP@0.5 = {r:.3f}   "
          f"[{'OK' if 0.85 <= r <= 0.97 else 'CHECK' if r <= 1.0 else 'BROKEN - not one run'}]")

    try:                                              # the same bound, per class
        bad = sum(1 for ap50, p, rc in zip(b.all_ap[:, 0], b.p, b.r)
                  if float(ap50) < float(p) * float(rc) - 1e-6)
        if bad:
            print(f"  ! {bad}/{len(b.p)} classes break AP50 >= P*R inside this single run -"
                  f" that is a bug, not a reporting mismatch")
    except Exception:
        pass

    sp = getattr(res, "speed", {}) or {}
    if sp:
        tot = sp.get("inference", math.nan) + sp.get("postprocess", 0.0)
        print(f"\n  inference {sp.get('inference', float('nan')):.2f} ms + NMS "
              f"{sp.get('postprocess', float('nan')):.2f} ms  ->  {1000/tot:.1f} FPS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
