"""
Print parameters and GFLOPs (Ultralytics model.info()) for every trained checkpoint.

    python info_all.py
    python info_all.py --glob "runs/train/*/weights/best.pt" --out complexity.csv

Sorted by folder name; deltas are printed against the first checkpoint, so name the
baseline run so that it sorts first. These are the Params / FLOPs columns of
Tables 3, 4, 6, 7, 11 and 12 of the paper (nc = 17 in-house, nc = 8 iWildCam).
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import warnings

warnings.filterwarnings("ignore")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="runs/train/*/weights/best.pt")
    ap.add_argument("--out", default="complexity.csv")
    args = ap.parse_args()

    from ultralytics import YOLO

    paths = sorted(glob.glob(args.glob))
    if not paths:
        print(f"no checkpoints matched {args.glob}")
        return

    rows = []
    print(f"{'run':<14}{'layers':>8}{'params (M)':>13}{'GFLOPs':>9}")
    print("-" * 44)
    for p in paths:
        run = os.path.basename(os.path.dirname(os.path.dirname(p)))
        try:
            layers, params, _grads, gflops = YOLO(p).info(detailed=False, verbose=False)
        except Exception as exc:  # noqa: BLE001
            print(f"{run:<14}  failed: {exc}")
            continue
        rows.append(dict(run=run, layers=layers, params_M=round(params / 1e6, 3),
                         gflops=round(float(gflops), 2), path=p))
        print(f"{run:<14}{layers:>8}{params / 1e6:>13.3f}{float(gflops):>9.2f}")

    if not rows:
        return
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwritten to {args.out}")

    base = rows[0]
    print(f"\ndeltas against '{base['run']}' ({base['params_M']:.3f} M / {base['gflops']:.2f} G):")
    for r in rows[1:]:
        print(f"  {r['run']:<14} {r['params_M'] - base['params_M']:+8.3f} M   "
              f"{r['gflops'] - base['gflops']:+7.2f} G")


if __name__ == "__main__":
    main()
