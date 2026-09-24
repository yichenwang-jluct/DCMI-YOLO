# DCMI-YOLO

**Detection-Coordinated Multi-scale Illumination-aware YOLO** — a lightweight end-to-end detector for nighttime and low-light camera-trap imagery, built for unattended monitoring of endangered wildlife (Amur tiger, Amur leopard and 15 other species of northeast China).

Illumination enhancement, downsampling, occlusion handling and bounding-box regression are optimised **jointly under the detection objective**, instead of enhancing first and detecting afterwards. On a 17-class nighttime camera-trap dataset it reaches **0.801 mAP@0.5:0.95 with 2.85 M parameters and 8.1 GFLOPs**, and sustains 26 FPS at 5.8 W on a Jetson Orin Nano.

---

## What is in the model

Four modules on top of Ultralytics YOLOv8n:

| Module | Replaces | What it does |
|---|---|---|
| **SCI-Gamma** | — (new frontend) | Closed-form, parameter-free adaptive gamma mapping. Brightens dark regions with a bounded response instead of a learned enhancement network, so it adds no trainable parameters. |
| **Star-ADown** | the stride-2 downsampling convolutions | Dual-path downsampling whose two branches are fused by a ReLU6-guided element-wise product and a 1×1 projection, rather than concatenated. A weak response in either branch suppresses the output there, which filters single-branch noise responses under low light. |
| **MultiSEAM** | — (before each detection head) | Three parallel channel-space multi-scale modules at patch sizes {6, 7, 8}, applied at all three detection scales (P3/P4/P5) to recover features lost to occlusion. |
| **DF-CIoU** | CIoU | Regression loss whose difficulty-mapping interval narrows as training proceeds, removing the IoU gradient of ambiguous low-IoU positives late in training. **Training-only — it does not change the inference graph.** |

---

## Results

All figures are means over three seeds (0 / 42 / 2024), measured on a site-disjoint test split under one protocol.

### In-house 17-class nighttime camera-trap dataset

| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Params (M) | FLOPs (G) | FPS |
|---|---|---|---|---|---|---|---|
| YOLOv8n (baseline) | 0.915 | 0.921 | 0.903 | 0.770 | 3.01 | 8.2 | 81 |
| **DCMI-YOLO** | **0.943** | 0.925 | **0.948** | **0.801** | **2.85** | **8.1** | 69 |

### iWildCam low-light subset (8 classes)

| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| YOLOv8n (baseline) | 0.880 | 0.861 | 0.825 | 0.733 |
| **DCMI-YOLO** | **0.919** | **0.880** | **0.865** | **0.764** |

### ENA-24 (independent third dataset, retrained on its own training split)

| Model | mAP@0.5 | mAP@0.5:0.95 | Recall |
|---|---|---|---|
| YOLOv8n (baseline) | 0.793 | 0.591 | 0.683 |
| **DCMI-YOLO** | **0.827** | **0.622** | **0.726** |

+3.1 pp mAP@0.5:0.95 over the baseline on both the in-house dataset and iWildCam, and the gain holds in all five folds of camera-site-level cross-validation (+4.1 to +4.6 pp).

### Ablation (in-house dataset)

| SCI-Gamma | Star-ADown | MultiSEAM | DF-CIoU | mAP@0.5 | mAP@0.5:0.95 | Params (M) | FLOPs (G) | FPS |
|:-:|:-:|:-:|:-:|---|---|---|---|---|
| ✗ | ✗ | ✗ | ✗ | 0.903 | 0.770 | 3.01 | 8.2 | 81 |
| ✓ | ✗ | ✗ | ✗ | 0.914 | 0.779 | 3.03 | 8.4 | 75 |
| ✗ | ✓ | ✗ | ✗ | 0.929 | 0.790 | 2.67 | 7.6 | 81 |
| ✗ | ✗ | ✓ | ✗ | 0.917 | 0.781 | 3.21 | 8.7 | 70 |
| ✗ | ✗ | ✗ | ✓ | 0.912 | 0.776 | 3.01 | 8.2 | 81 |
| ✓ | ✓ | ✗ | ✗ | 0.939 | 0.795 | 2.71 | 7.8 | 75 |
| ✓ | ✓ | ✓ | ✗ | 0.943 | 0.798 | 2.85 | 8.1 | 69 |
| ✓ | ✓ | ✓ | ✓ | **0.948** | **0.801** | 2.85 | 8.1 | 69 |

DF-CIoU is a training-only loss, so the last two rows share an inference graph and therefore identical parameters, FLOPs and FPS.

---

## Efficiency

FPS is measured at 640×640, batch 1, FP32, forward pass + NMS (conf 0.001, IoU 0.7, max_det 300), as the median over 300 timed iterations after 50 warm-up iterations, with `torch.cuda.synchronize()` around the timed region. Desktop figures are on an RTX 4060 (8 GB).

| Device | FPS | Avg power (W) | Energy / image (J) |
|---|---|---|---|
| NVIDIA RTX 4060 (8 GB) | 69 | — | — |
| NVIDIA Jetson Orin Nano (7 W mode) | 26 | 5.8 | 0.22 |
| NVIDIA Jetson Xavier NX (15 W mode) | 22 | 10.5 | 0.48 |
| Raspberry Pi 4 + Coral Edge TPU (INT8) | 10 | 3.2 | 0.32 |

The Orin Nano gives the best energy efficiency of the three, which is what matters for solar-powered field deployment.

---

## Install

```bash
git clone https://github.com/yichenwang-jluct/DCMI-YOLO.git
cd DCMI-YOLO
pip install -r requirements.txt
```

Reference environment (what the paper's numbers were produced on): Windows 11, Intel Core i7-13650HX, NVIDIA RTX 4060 8 GB, Python 3.8, PyTorch 2.0.1, CUDA 11.7.

---

## Data

Point a dataset YAML at your own images in standard Ultralytics layout:

```
dataset/
├── images/{train,val,test}/
└── labels/{train,val,test}/
```

```yaml
# data/inhouse.yaml
path: /abs/path/to/dataset
train: images/train
val:   images/val
test:  images/test
nc: 17
names: [Amur tiger, Badger, Asian black bear, Cow, Dog, Hare, Amur leopard,
        Leopard cat, Musk deer, Raccoon dog, Red fox, Roe deer, Sable,
        Sika deer, Weasel, Wild boar, Yellow-throated marten]
```

> **The camera-trap dataset itself is not public.** It contains images of endangered species together with capture timestamps, and releasing the raw imagery or location metadata could aid poaching. Researchers with a legitimate need can request access from the corresponding author; a confidentiality agreement is required.

---

## Train

```bash
python train.py --cfg models/dcmi-yolo.yaml --data data/inhouse.yaml \
                --epochs 200 --imgsz 640 --batch 8 --seed 0
```

The protocol used for every model in the paper, baselines included:

- initialised from COCO-pretrained YOLOv8n weights; 200 epochs at 640×640
- mini-batch 8 accumulated to a nominal batch of 64
- SGD, initial LR 0.01, momentum 0.937, weight decay 5e-4, 3-epoch linear warm-up, linear decay to a final LR of 1e-4
- FP16, deterministic kernels, early stopping with patience 100
- augmentation: Mosaic throughout, HSV (h 0.015, s 0.7, v 0.4), horizontal flip p=0.5, translate 0.1, scale 0.5, random erasing 0.4; rotation, shear, perspective, MixUp and Copy-Paste disabled

Every comparison model was reproduced under exactly this schedule. Change any of it and the numbers stop being comparable.

---

## Evaluate

```bash
python val.py --weights runs/train/exp/weights/best.pt --data data/inhouse.yaml \
              --imgsz 640 --batch 1 --conf 0.001 --iou 0.7 --max-det 300
```

**Read P, R, mAP@0.5 and mAP@0.5:0.95 from one `val()` call.** They are not independent: for any point (R, P) on a precision–recall curve the interpolated average precision satisfies

```
AP@0.5  >=  P * R
```

since every recall below R carries interpolated precision of at least P, so the area under the curve is at least the P×R rectangle. Averaged over classes this becomes `mAP@0.5 >= P̄·R̄ + Cov(P, R)`, and the covariance term is worth only a few thousandths in practice. A quick check on any row you report:

```python
r = (P * R) / mAP50
# r <= 1.00 always; a real detector lands around 0.85-0.96.
# r > 1.00 means P, R and mAP came from different runs, not that the model is unusual.
```

Every row in the paper satisfies this, with r between 0.834 and 0.943.

Results are reported as the mean over seeds 0 / 42 / 2024. Average the four metrics across seeds together — never one column from one seed and another column from another.

---

## Repository layout

```
models/                     model configs - the full model and every ablation row of Table 6
  yolov8n-baseline.yaml       row 1  YOLOv8n, no DCMI module
  yolov8-SCIGamma.yaml        row 2  + SCI-Gamma
  yolov8-StarADown.yaml       row 3  + Star-ADown
  yolov8-MultiSEAM.yaml       row 4  + MultiSEAM
  yolov8-SCI-StarADown.yaml   row 6  SCI-Gamma + Star-ADown
  yolov8-DCMI.yaml            rows 7-8  the full model
ultralytics_patch/          the three modules, the two losses, and how to register them
  dcmi_modules.py             SCIGamma, StarADown, MultiSEAM
  dcmi_loss.py                DF-CIoU (Eq. 13-15), Repulsion Loss (Eq. 9-12)
  INTEGRATION.md              read this before touching your Ultralytics tree
train.py                    training entry point, the paper's protocol baked in
val.py                      evaluation; prints all four metrics from one val() call
data/                       dataset YAML templates
scripts/                    complexity, FPS and metric-export tooling
  param_check.py              verifies a build reproduces 2.85 M / 8.1 GFLOPs
  info_all.py                 params and FLOPs for every checkpoint
  bench_one.py                FPS of one model, under the reported protocol
  export_metrics.py           P/R/mAP for every row, from one val() call each
  measure_complexity.py       params and FLOPs under three independent counters
  bench_fps.py                FPS for every configuration, with an additivity check
requirements.txt
```

Rows 5 and 8 of Table 6 reuse the config of rows 1 and 7: DF-CIoU is a training-only
loss and does not change the inference graph, which is why those row pairs report
identical parameters, FLOPs and FPS.

## Reproducing the paper

```bash
# the full model, one seed
python train.py --cfg models/yolov8-DCMI.yaml --data data/inhouse.yaml --seed 0

# an ablation row without DF-CIoU
python train.py --cfg models/yolov8-StarADown.yaml --data data/inhouse.yaml --no-dfciou

# evaluate, with the P*R <= mAP@0.5 check built in
python val.py --weights runs/train/<run>/weights/best.pt --data data/inhouse.yaml

# confirm the build matches the reported complexity before trusting anything
python scripts/param_check.py --cfg models/yolov8-DCMI.yaml --nc 17
```

Results in the paper are means over seeds 0 / 42 / 2024. Average the four metrics
across seeds together - never one column from one seed and another from another.

## Citation

The paper is under review. A full citation will be added once it is published; until then, please cite the repository and the preprint if you use this work.

```bibtex
@misc{dcmiyolo2026,
  title  = {Edge-Deployable Low-Light Object Detection for Automated Nighttime
            Wildlife Monitoring: From Detection Accuracy to Ecological Indicators},
  author = {Wang, Yichen and Qi, Shenao and He, Yuli and So, Chi Chiu},
  year   = {2026},
  note   = {Manuscript under review},
  url    = {https://github.com/yichenwang-jluct/DCMI-YOLO}
}
```

---

## License

AGPL-3.0, inherited from [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics),
on which this work is built. Add the licence file through GitHub's **Add file → Create
new file → LICENSE** dialog, which offers AGPL-3.0 as a template, or copy it from the
Ultralytics repository.

## Acknowledgment

We thank the management authorities of the Changbai Mountain and Hunchun National Nature Reserves for permission to deploy camera traps and for logistical support during fieldwork, and the wildlife specialists who supervised species identification during annotation.
