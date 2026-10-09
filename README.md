# DCMI-YOLO: Edge-Deployable Low-Light Detection for Nighttime Camera-Trap Wildlife Monitoring

Official code for **DCMI-YOLO** (Detection-Coordinated Multi-scale Illumination-aware YOLO), proposed in:

> Y. Wang, S. Qi, Y. He, and C. C. So, "Edge-Deployable Low-Light Object Detection for Automated Nighttime Wildlife Monitoring: From Detection Accuracy to Ecological Indicators," *IEEE Access*, under review, 2026.

DCMI-YOLO is a lightweight end-to-end detector for nighttime camera-trap images, built on Ultralytics YOLOv8n. Illumination enhancement, downsampling, occlusion handling, and bounding-box regression are optimized jointly under the detection loss. It adds four components to YOLOv8n:

- **SCI-Gamma** (Section III-A): the illumination-estimation network of SCI, trained end to end under the detection loss, followed by a closed-form adaptive gamma correction `I_out = I_in^γ` with `γ = 1 / (1 + α(1 − L))`, where `L` is the normalized mean luminance of each image and `α = 2`. The gamma value has no learnable parameters and no gradient. During training the illumination estimator runs as a weight-shared cascade with the self-calibration module; at inference a single stage is used.
- **Star-ADown** (Section III-B): replaces the six stride-2 convolutions after the stem (four in the backbone, two in the bottom-up path of the neck). A 2×2 average pooling (stride 1, no padding) is followed by a channel split into a spatial path (3×3 stride-2 convolution + ReLU6) and a texture path (3×3 stride-2 max pooling + 1×1 convolution); the two paths are fused by an element-wise product and a 1×1 projection, at about 3.0C² parameters per module.
- **MultiSEAM** (Section III-C): one block after the C2f at each detection scale (P3/8, P4/16, P5/32), with three parallel channel-and-spatial mixing branches of patch sizes 3, 5, and 7. The C2f outputs at P4/16 and P5/32 are narrowed to 64 channels (128 and 256 in YOLOv8n), so all three blocks and the detection-head inputs operate on 64 channels. MultiSEAM is trained together with Repulsion Loss (λ_RepGT = λ_RepBox = 0.5, σ = 0.5).
- **DF-CIoU** (Section III-D): a Focaler-CIoU box loss whose mapping interval narrows linearly from [0, 1] at the start of training to [d0, u0] = [0.2, 0.8] at the last epoch. It is training-only and does not change the network.

The full model has **2.99 M parameters and 6.9 GFLOPs** (640×640 input, 17 classes, as reported by Ultralytics `model.info()`).

> **Data availability.** The in-house dataset contains camera-trap images of endangered species (e.g., the Amur tiger and the Amur leopard) together with their capture timestamps. To prevent misuse that could threaten these animals (e.g., poaching), the raw images and associated location information are **not** publicly available. The data may be made available from the corresponding author upon reasonable request and subject to a data-use agreement, for legitimate academic research only.

---

## Results

Test-split results reported in the paper (camera-site-disjoint splits; mean ± SD over seeds 0, 42, and 2024; Tables 11 and 12):

| Dataset | P | R | mAP@0.5 | mAP@0.5:0.95 | Params (M) | GFLOPs | FPS (RTX 4060) |
|---|---|---|---|---|---|---|---|
| In-house nighttime dataset (17 classes) | 0.943 | 0.925 | 0.948 ± 0.004 | 0.801 ± 0.004 | 2.99 | 6.9 | 69 |
| iWildCam low-light subset (8 classes) | 0.919 | 0.880 | 0.865 ± 0.004 | 0.764 ± 0.004 | 2.99 | 6.9 | 69 |

On the in-house validation split used for model selection and hyperparameter search, the full model reaches mAP@0.5:0.95 = 0.824.

### Ablation (Table 3, in-house test split)

| Row | Config | DF-CIoU | mAP@0.5 | mAP@0.5:0.95 | Params (M) | GFLOPs | FPS |
|---|---|---|---|---|---|---|---|
| 1 | `models/yolov8n-baseline.yaml` | no | 0.903 | 0.770 | 3.01 | 8.2 | 81 |
| 2 | `models/yolov8-SCIGamma.yaml` | no | 0.914 | 0.779 | 3.13 | 8.4 | 75 |
| 3 | `models/yolov8-StarADown.yaml` | no | 0.929 | 0.790 | 2.67 | 7.5 | 81 |
| 4 | `models/yolov8-MultiSEAM.yaml` (+ Repulsion Loss) | no | 0.917 | 0.781 | 3.21 | 7.3 | 70 |
| 5 | `models/yolov8n-baseline.yaml` | yes | 0.912 | 0.776 | 3.01 | 8.2 | 81 |
| 6 | `models/yolov8-SCI-StarADown.yaml` | no | 0.939 | 0.795 | 2.79 | 7.7 | 75 |
| 7 | `models/yolov8-DCMI.yaml` (+ Repulsion Loss) | no | 0.943 | 0.798 | 2.99 | 6.9 | 69 |
| 8 | `models/yolov8-DCMI.yaml` (+ Repulsion Loss) | yes | 0.948 | 0.801 | 2.99 | 6.9 | 69 |

The corresponding results on the iWildCam low-light subset are in Table 4 of the paper.

### Edge deployment (Table 16; 640×640, batch size 1)

| Device | Runtime | FPS | Average power (W) | Energy per image (J) |
|---|---|---|---|---|
| NVIDIA Jetson Xavier NX (15 W mode) | TensorRT FP16 | 22 | 10.5 | 0.48 |
| NVIDIA Jetson Orin Nano (7 W mode) | TensorRT FP16 | 26 | 5.8 | 0.22 |
| Raspberry Pi 4 + Coral Edge TPU | TFLite INT8 | 10 | 3.2 | 0.32 |

---

## Classes

In-house dataset (17 classes, `data/inhouse.yaml`): Amur tiger, badger, Asian black bear, cow, dog, hare, Amur leopard, leopard cat, musk deer, raccoon dog, red fox, roe deer, sable, sika deer, weasel, wild boar, yellow-throated marten.

iWildCam low-light subset (8 classes, `data/iwildcam.yaml`): bobcat, opossum, coyote, raccoon, dog, squirrel, rabbit, skunk.

---

## Repository structure

```
DCMI-YOLO/
├── README.md
├── LICENSE                        # GNU AGPL-3.0 (inherited from Ultralytics)
├── requirements.txt
├── train.py                       # training entry point (Section IV-A2 protocol)
├── val.py                         # evaluation entry point (P, R, mAP@0.5, mAP@0.5:0.95 from one val() call)
├── data/
│   ├── inhouse.yaml               # in-house dataset config (paths are placeholders; data not distributed)
│   └── iwildcam.yaml              # iWildCam low-light subset config
├── models/                        # one config per row of the ablation (Tables 3 and 4)
│   ├── yolov8n-baseline.yaml
│   ├── yolov8-SCIGamma.yaml
│   ├── yolov8-StarADown.yaml
│   ├── yolov8-MultiSEAM.yaml
│   ├── yolov8-SCI-StarADown.yaml
│   └── yolov8-DCMI.yaml           # full DCMI-YOLO
├── ultralytics_patch/             # Ultralytics modifications used for the paper (modules, parse_model, losses)
│   └── INTEGRATION.md             # how to apply them
└── scripts/
    ├── info_all.py                # params / GFLOPs of trained checkpoints
    ├── measure_complexity.py      # params / GFLOPs with three counters
    ├── bench_one.py               # FPS of one checkpoint
    ├── bench_fps.py               # FPS of every ablation checkpoint
    └── export_metrics.py          # mean ± SD over seeds, from single val() calls
```

---

## Installation

The paper's experiments ran on Windows 11 with Python 3.8, PyTorch 2.0.1, and CUDA 11.7 (Intel i7-13650HX CPU, NVIDIA GeForce RTX 4060 8 GB GPU).

```bash
pip install -r requirements.txt
```

Then apply the modifications in `ultralytics_patch/` to the installed Ultralytics package as described in `ultralytics_patch/INTEGRATION.md`. The pinned Ultralytics version in `requirements.txt` is the one the patch was written for; other versions may need small changes to `parse_model` and the loss module.

---

## Training

```bash
# full DCMI-YOLO (Table 3 row 8), one seed
python train.py --cfg models/yolov8-DCMI.yaml --data data/inhouse.yaml --seed 0

# any ablation row without DF-CIoU, e.g. the baseline (row 1)
python train.py --cfg models/yolov8n-baseline.yaml --data data/inhouse.yaml --seed 0 --no-dfciou
```

The paper reports the mean over seeds 0, 42, and 2024. Training protocol (Section IV-A2), applied to every model in the paper:

- initialization from COCO-pretrained YOLOv8n weights; 200 epochs; 640×640 letterboxed input
- mini-batch of 8 accumulated to a nominal batch of 64
- SGD, initial learning rate 0.01, momentum 0.937, weight decay 5×10⁻⁴; 3-epoch linear warm-up; linear decay to 1×10⁻⁴
- FP16 mixed precision, deterministic kernels, early stopping with patience 100
- augmentation: Mosaic throughout training (`close_mosaic=0`), HSV jitter (h = 0.015, s = 0.7, v = 0.4), horizontal flip (p = 0.5), translation 0.1, scaling 0.5; rotation, shear, perspective, MixUp, and Copy-Paste disabled
- loss weights: YOLOv8 defaults (box 7.5, cls 0.5, dfl 1.5); Repulsion Loss (λ_RepGT = λ_RepBox = 0.5, σ = 0.5) for the configurations that contain MultiSEAM; DF-CIoU (d0 = 0.2, u0 = 0.8) unless `--no-dfciou` is given

---

## Evaluation

```bash
python val.py --weights path/to/best.pt --data data/inhouse.yaml --split test
```

Detections are filtered at confidence 0.001 with NMS at IoU 0.7 and at most 300 detections per image. Following Ultralytics, precision and recall are read at the confidence that maximizes mean F1, whereas AP and mAP integrate the whole precision–recall curve. `val.py` evaluates with square 640×640 letterboxing, as in the paper; the per-epoch validation printed during training uses rectangular batches, so its numbers differ slightly.

Complexity and speed:

```bash
python scripts/info_all.py --glob "runs/train/*/weights/best.pt"
python scripts/bench_one.py --weights path/to/best.pt --imgsz 640 --batch 1
```

---

## Pretrained weights

The trained full model (in-house dataset, 17 classes) is attached to the [v1.0 release](https://github.com/yichenwang-jluct/DCMI-YOLO/releases/tag/v1.0):

```
https://github.com/yichenwang-jluct/DCMI-YOLO/releases/download/v1.0/best.pt
```

After applying `ultralytics_patch/`, `YOLO("best.pt").info()` should report 2.99 M parameters and 6.9 GFLOPs.

> The weights were trained on sensitive endangered-species data and are released for research reproducibility only. Please do not use them in ways that could facilitate locating or harming wildlife.

---

## License

This project is released under the GNU AGPL-3.0 license, inherited from Ultralytics YOLO, on which it is built. See `LICENSE` for the full text.

---

## Acknowledgements

This work builds on [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) and on the ideas of SCI (Ma et al., CVPR 2022), ADown (YOLOv9), Rewrite the Stars (StarNet), SEAM/MultiSEAM (YOLO-FaceV2), Repulsion Loss, and Focaler-IoU; see the paper for the full references.

## Citation

```bibtex
@article{wang2026dcmiyolo,
  title   = {Edge-Deployable Low-Light Object Detection for Automated Nighttime Wildlife Monitoring: From Detection Accuracy to Ecological Indicators},
  author  = {Wang, Yichen and Qi, Shenao and He, Yuli and So, Chi Chiu},
  journal = {IEEE Access (under review)},
  year    = {2026},
  note    = {Code: https://github.com/yichenwang-jluct/DCMI-YOLO}
}
```

The citation will be updated with the volume, pages, and DOI after publication.
