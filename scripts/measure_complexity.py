"""
Measure parameters and FLOPs consistently for every configuration in the paper.

Run once per dataset, then paste the CSV into Tables 5-8, 11, 12, 16, 17 and 19:

    python measure_complexity.py --nc 17 --out complexity_inhouse.csv
    python measure_complexity.py --nc 8  --out complexity_iwildcam.csv

Three independent counters are reported per model:

  ultra_gflops   what Ultralytics prints (model.info()). It profiles the model
                 at stride x stride (32x32) and SCALES the result to 640x640,
                 assuming every module's cost is quadratic in spatial size.
  thop_gflops    the same counter run directly at the real 640x640 input.
  fvcore_gflops  an independent library, if installed.

If ultra_gflops and thop_gflops disagree, the scaling assumption is broken by one
of your modules (MultiSEAM's fixed patch sizes are the likely candidate) and the
number to report is thop_gflops. If thop and fvcore disagree by more than ~10 %,
a custom op is not being counted at all - check the "unsupported" column.
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# EDIT THIS: paper row label -> trained checkpoint (preferred) or model .yaml.
# Checkpoints are safer than .yaml: they are the exact models you evaluated, so
# the complexity numbers cannot drift away from the reported accuracy.
# ---------------------------------------------------------------------------
CONFIGS = [
    ("YOLOv8n (baseline)",        "runs/ablation/base/weights/best.pt"),
    ("+ SCI-Gamma",               "runs/ablation/sci/weights/best.pt"),
    ("+ Star-ADown",              "runs/ablation/adown/weights/best.pt"),
    ("+ MultiSEAM",               "runs/ablation/seam/weights/best.pt"),
    ("+ DF-CIoU",                 "runs/ablation/dfciou/weights/best.pt"),
    ("SCI-Gamma + Star-ADown",    "runs/ablation/sci_adown/weights/best.pt"),
    ("SCI + ADown + MultiSEAM",   "runs/ablation/sci_adown_seam/weights/best.pt"),
    ("DCMI-YOLO (full)",          "runs/ablation/full/weights/best.pt"),
]


def load_model(path: str, nc: int):
    """Build an nn.Module from a .pt checkpoint or a .yaml config at `nc` classes."""
    from ultralytics import YOLO

    if str(path).endswith((".yaml", ".yml")):
        from ultralytics.nn.tasks import DetectionModel

        model = DetectionModel(cfg=str(path), nc=nc, verbose=False)
    else:
        model = YOLO(str(path)).model
    model.eval()
    return model


def model_nc(model) -> int | None:
    for attr in ("nc",):
        if hasattr(model, attr):
            return int(getattr(model, attr))
    head = getattr(model, "model", [None])[-1]
    return int(getattr(head, "nc", 0)) or None


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def ultralytics_gflops(model, imgsz: int) -> float:
    try:
        from ultralytics.utils.torch_utils import get_flops
    except ImportError:  # older layout
        from ultralytics.yolo.utils.torch_utils import get_flops  # type: ignore
    try:
        return float(get_flops(copy.deepcopy(model), imgsz))
    except Exception as exc:  # noqa: BLE001
        print(f"    ! ultralytics get_flops failed: {exc}")
        return float("nan")


def thop_gflops(model, imgsz: int) -> float:
    """thop at the true input size. Ultralytics counts one multiply-add as 2 ops."""
    try:
        import thop
        import torch
    except ImportError:
        return float("nan")
    try:
        im = torch.zeros(1, 3, imgsz, imgsz)
        macs, _ = thop.profile(copy.deepcopy(model), inputs=(im,), verbose=False)
        return float(macs) / 1e9 * 2
    except Exception as exc:  # noqa: BLE001
        print(f"    ! thop failed: {exc}")
        return float("nan")


def fvcore_gflops(model, imgsz: int):
    """Independent cross-check. fvcore counts MACs, so multiply by 2 to match."""
    try:
        import torch
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        return float("nan"), ""
    try:
        fca = FlopCountAnalysis(copy.deepcopy(model), torch.zeros(1, 3, imgsz, imgsz))
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        fca.tracer_warnings("none")
        total = float(fca.total()) / 1e9 * 2
        unsupported = ",".join(sorted(dict(fca.unsupported_ops()))[:6])
        return total, unsupported
    except Exception as exc:  # noqa: BLE001
        print(f"    ! fvcore failed: {exc}")
        return float("nan"), ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", type=int, required=True, help="number of classes (17 in-house, 8 iWildCam)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="complexity.csv")
    args = ap.parse_args()

    rows = []
    for label, path in CONFIGS:
        print(f"[*] {label}  <- {path}")
        if not Path(path).exists():
            print("    ! file not found, skipped")
            continue
        model = load_model(path, args.nc)

        found_nc = model_nc(model)
        if found_nc and found_nc != args.nc:
            print(f"    ! WARNING: this model has nc={found_nc}, not {args.nc}. "
                  "Params/FLOPs are not comparable with the other rows.")

        params = count_params(model)
        g_ultra = ultralytics_gflops(model, args.imgsz)
        g_thop = thop_gflops(model, args.imgsz)
        g_fv, unsupported = fvcore_gflops(model, args.imgsz)

        note = []
        if g_ultra == g_ultra and g_thop == g_thop and g_thop > 0:
            if abs(g_ultra - g_thop) / g_thop > 0.10:
                note.append("ultra-vs-thop>10%: report thop_gflops")
        if g_fv == g_fv and g_thop == g_thop and g_thop > 0:
            if abs(g_fv - g_thop) / g_thop > 0.10:
                note.append("thop-vs-fvcore>10%: an op is uncounted")
        if unsupported:
            note.append(f"unsupported: {unsupported}")

        rows.append(dict(
            config=label, nc=found_nc or args.nc,
            params_M=round(params / 1e6, 3),
            ultra_gflops=round(g_ultra, 2),
            thop_gflops=round(g_thop, 2),
            fvcore_gflops=round(g_fv, 2),
            note="; ".join(note),
        ))
        print(f"    params {params/1e6:.3f} M | ultra {g_ultra:.2f} | thop {g_thop:.2f} | fvcore {g_fv:.2f}")
        if note:
            print(f"    -> {'; '.join(note)}")

    if not rows:
        print("Nothing measured - check the paths in CONFIGS.")
        return 1

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWritten to {args.out}")

    base = rows[0]
    print(f"\nDeltas against '{base['config']}' (paste these into Table 8):")
    for r in rows[1:]:
        print(f"  {r['config']:<28} dParams {r['params_M']-base['params_M']:+.3f} M   "
              f"dFLOPs {r['thop_gflops']-base['thop_gflops']:+.2f} G")
    return 0


if __name__ == "__main__":
    sys.exit(main())
