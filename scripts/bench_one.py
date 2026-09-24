"""
Measure the FPS of ONE model.

    python bench_one.py --weights runs/ablation/base/weights/best.pt
    python bench_one.py --weights yolov8n.pt --image test/images/0001.jpg

Prints three numbers, because "FPS" in a paper is ambiguous:

  model   pure forward pass                     (no letterbox, no NMS)
  +NMS    forward pass + non-maximum suppression   <- report this one
  e2e     the full predict pipeline             (needs --image)

Whatever you report, use the SAME setting for every row of the table and state
it in Section 4.1.2. Warm-up, CUDA synchronisation and a median over many
repeats are handled here; getting any of those wrong moves the answer by tens
of percent.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help=".pt checkpoint")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=1, help="the paper reports batch = 1")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--half", action="store_true", help="FP16; keep it the same for every row")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--repeat", type=int, default=300)
    ap.add_argument("--image", default="", help="a real test image, for the end-to-end number")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO
    from ultralytics.utils import ops

    y = YOLO(args.weights)
    model = y.model.to(args.device).eval()
    if args.half:
        model = model.half()

    on_cuda = str(args.device).startswith("cuda")
    sync = torch.cuda.synchronize if on_cuda else (lambda: None)
    if on_cuda:
        print(f"device : {torch.cuda.get_device_name(0)}")
    n_par = sum(p.numel() for p in model.parameters())
    print(f"weights: {args.weights}")
    print(f"params : {n_par / 1e6:.3f} M | imgsz {args.imgsz} | batch {args.batch} | "
          f"{'FP16' if args.half else 'FP32'} | warmup {args.warmup} | repeat {args.repeat}\n")

    x = torch.zeros(args.batch, 3, args.imgsz, args.imgsz, device=args.device,
                    dtype=torch.half if args.half else torch.float)

    with torch.inference_mode():
        ms_model = timed(lambda: model(x), args.warmup, args.repeat, sync) / args.batch

        def fwd_nms():
            out = model(x)
            pred = out[0] if isinstance(out, (list, tuple)) else out
            ops.non_max_suppression(pred, conf_thres=0.001, iou_thres=0.7, max_det=300)

        ms_nms = timed(fwd_nms, args.warmup, args.repeat, sync) / args.batch

    print(f"  model        {ms_model:7.3f} ms   {1000 / ms_model:7.1f} FPS")
    print(f"  model + NMS  {ms_nms:7.3f} ms   {1000 / ms_nms:7.1f} FPS   <- report this")

    if args.image and Path(args.image).exists():
        import cv2

        img = cv2.imread(args.image)
        if img is not None:
            ms_e2e = timed(
                lambda: y.predict(img, imgsz=args.imgsz, device=args.device, half=args.half,
                                  conf=0.001, iou=0.7, max_det=300, verbose=False),
                args.warmup, args.repeat, sync)
            print(f"  end-to-end   {ms_e2e:7.3f} ms   {1000 / ms_e2e:7.1f} FPS")
    elif args.image:
        print(f"  end-to-end   skipped: {args.image} not found")


if __name__ == "__main__":
    main()
