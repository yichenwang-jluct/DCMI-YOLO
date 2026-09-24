"""
Train DCMI-YOLO, or any row of the Table 6 ablation, under the paper's protocol.

    python train.py --cfg models/yolov8-DCMI.yaml --data data/inhouse.yaml --seed 0
    python train.py --cfg models/yolov8n-baseline.yaml --data data/inhouse.yaml --no-dfciou

Every model in the paper - baselines and competitors included - was trained with
exactly the settings below. Change any of them and the comparison stops being a
comparison. Results are reported as the mean over seeds 0 / 42 / 2024, so run
this three times and average afterwards.
"""

from __future__ import annotations

import argparse

# Section IV-A2 of the paper, verbatim.
HYP = dict(
    epochs=200, imgsz=640, batch=8, nbs=64,          # mini-batch 8 accumulated to 64
    optimizer="SGD", lr0=0.01, lrf=0.01,             # 0.01 -> 1e-4 linear
    momentum=0.937, weight_decay=5e-4,
    warmup_epochs=3.0, cos_lr=False,
    amp=True, deterministic=True, patience=100,
    box=7.5, cls=0.5, dfl=1.5,                       # YOLOv8 defaults, retained
    mosaic=1.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    fliplr=0.5, translate=0.1, scale=0.5, erasing=0.4,
    degrees=0.0, shear=0.0, perspective=0.0, mixup=0.0, copy_paste=0.0,
    flipud=0.0, close_mosaic=0,
)


def add_dfciou_callback(model, epochs: int):
    """
    The DF-CIoU interval of Eq. (13) depends on the epoch, which Ultralytics does
    not pass into the criterion. Push it in at the start of each epoch.
    """
    def _set_epoch(trainer):
        crit = getattr(trainer.model, "criterion", None)
        bl = getattr(crit, "bbox_loss", None) if crit else None
        if bl is not None:
            bl.epoch, bl.epochs = trainer.epoch, epochs
    model.add_callback("on_train_epoch_start", _set_epoch)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="models/yolov8-DCMI.yaml")
    ap.add_argument("--data", default="data/inhouse.yaml")
    ap.add_argument("--weights", default="yolov8n.pt", help="COCO-pretrained init")
    ap.add_argument("--epochs", type=int, default=HYP["epochs"])
    ap.add_argument("--seed", type=int, default=0, help="the paper uses 0, 42 and 2024")
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default=None)
    ap.add_argument("--no-dfciou", action="store_true",
                    help="keep plain CIoU - use for the Table 6 rows without DF-CIoU")
    a = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(a.cfg)
    if a.weights:
        model.load(a.weights)                        # COCO-pretrained YOLOv8n weights
    if not a.no_dfciou:
        add_dfciou_callback(model, a.epochs)

    hyp = dict(HYP, epochs=a.epochs)
    model.train(data=a.data, seed=a.seed, device=a.device,
                name=a.name or f"{a.cfg.split('/')[-1].replace('.yaml','')}_seed{a.seed}",
                **hyp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
