# DCMI-YOLO: Nighttime Camera-Trap Wildlife Detection

Official implementation of **DCMI-YOLO** (Detection-Coordinated Multi-scale Illumination-aware YOLO), proposed in:

> Y. Wang, S. Qi, Y. He, and C. C. So, "Edge-Deployable Low-Light Object Detection for Automated Nighttime Wildlife Monitoring: From Detection Accuracy to Ecological Indicators," submitted to *IEEE Access*, 2026.

DCMI-YOLO is a lightweight end-to-end detector for nighttime and low-light camera-trap images, built on Ultralytics YOLOv8 for unattended monitoring of endangered wildlife (e.g., the Amur tiger and the Amur leopard). It adds four components to YOLOv8n:

- **SCI-Gamma**: self-calibrated illumination enhancement with a closed-form adaptive gamma correction, trained end to end under the detection loss
- **Star-ADown**: lightweight downsampling that fuses two branches with a star (element-wise multiplication) operation
- **MultiSEAM**: multi-scale occlusion-aware attention, used together with Repulsion Loss
- **DF-CIoU**: bounding-box regression loss with a dynamically narrowing Focaler interval

The full model has **2.89 M parameters and 8.3 GFLOPs** (640×640 input) and detects 17 wildlife classes.

> ⚠️ **Data availability**: Due to the conservation sensitivity of the species involved (endangered Amur tiger and Amur leopard), **the dataset is NOT publicly released**. Raw images and location/timestamp information are withheld to avoid any risk of misuse (e.g., poaching). Researchers seeking access for legitimate academic purposes may contact the corresponding author; access may require a data-use/confidentiality agreement.

---

## Results

Test-split results reported in the paper (camera-site-disjoint splits, mean of three seeds):

| Dataset | P | R | mAP@0.5 | mAP@0.5:0.95 | Params (M) | GFLOPs | FPS (RTX 4060) |
|---|---|---|---|---|---|---|---|
| In-house nighttime dataset (17 classes) | 0.943 | 0.925 | 0.948 | 0.801 | 2.89 | 8.3 | 69 |
| iWildCam low-light subset (8 classes) | 0.919 | 0.880 | 0.865 | 0.764 | 2.89 | 8.3 | 69 |

On the validation split used for model selection, the full model reaches mAP@0.5:0.95 = 0.824 on the in-house dataset. `results/results.csv` logs the per-epoch metrics on the validation split recorded during training (Ultralytics default `rect=True`); they are therefore not the test-split numbers in the table above.

Edge deployment: 26 FPS at 5.8 W average power on an NVIDIA Jetson Orin Nano (TensorRT FP16, 7 W mode, batch size 1).

---

## Classes (17)

```
AmurTiger, Badger, BlackBear, Cow, Dog, Hare, Leopard, LeopardCat,
MuskDeer, RaccoonDog, RedFox, RoeDeer, Sable, SikaDeer, Weasel,
WildBoar, Y.T.Marten
```

In the paper these classes are named Amur tiger, badger, Asian black bear, cow, dog, hare, Amur leopard, leopard cat, musk deer, raccoon dog, red fox, roe deer, sable, sika deer, weasel, wild boar, and yellow-throated marten.

---

## Repository structure

```
DCMI-YOLO/
├── README.md
├── LICENSE                         # AGPL-3.0 (inherited from Ultralytics)
├── requirements.txt
├── .gitignore
├── models/
│   └── dcmi-yolo.yaml              # model architecture (nc=17)
├── train.py                        # training entry
├── val.py                          # validation entry
├── eval_metrics.py                 # per-class metrics + params + GFLOPs (CSV export)
├── FPS.py                          # latency / FPS benchmark
├── data/
│   └── A_my_data.yaml.template     # dataset config TEMPLATE (no real paths/data)
└── results/                        # training curves & metrics (NO raw imagery)
    ├── results.csv
    ├── results.png
    ├── confusion_matrix.png
    ├── confusion_matrix_normalized.png
    ├── P_curve.png
    ├── R_curve.png
    ├── F1_curve.png
    └── PR_curve.png
```

> Note: `train_batch*.jpg`, `val_batch*.jpg`, `labels*.jpg`, and TensorBoard event files are intentionally **excluded** because they contain raw camera-trap imagery or dataset statistics.

---

## Installation

```
# Python 3.8+ recommended
pip install -r requirements.txt
```

The exact `ultralytics` and `torch` versions used in the paper are pinned in `requirements.txt`. If you build on a different YOLO version, results may differ slightly.

---

## Dataset format

This project expects the standard Ultralytics detection layout. Prepare your own dataset and a `data.yaml` such as:

```
# A_my_data.yaml  (template: fill in YOUR OWN paths)
path: ./datasets/your_dataset      # dataset root
train: images/train
val: images/val
test: images/test

nc: 17
names: [AmurTiger, Badger, BlackBear, Cow, Dog, Hare, Leopard, LeopardCat,
        MuskDeer, RaccoonDog, RedFox, RoeDeer, Sable, SikaDeer, Weasel,
        WildBoar, Y.T.Marten]
```

---

## Usage

### 1. Train

```
python train.py
```

Key settings used in the paper: `imgsz=640`, `epochs=200`, `batch=8` (accumulated to a nominal batch of 64), `optimizer=SGD`, `close_mosaic=0`, seeds 0/42/2024. Edit the model/data paths inside `train.py` before running.

### 2. Evaluate (per-class metrics + Params + GFLOPs)

```
python eval_metrics.py
```

Outputs per-class precision, recall, mAP@0.5, and mAP@0.5:0.95, the overall metrics, model parameters and GFLOPs, and saves `per_class_metrics.csv`.

> To reproduce the end-of-training validation numbers, use `rect=True`.
> For metrics comparable with the paper, use `rect=False` (square evaluation) on the test split.

### 3. Benchmark speed (FPS / latency)

```
python FPS.py --weights path/to/best.pt --batch 1 --imgs 640 640 --device 0
```

---

## Pretrained weights

The trained weights (`best.pt`, ≈5.6 MB, in-house dataset, 17 classes) can be downloaded from the [v1.0 release](https://github.com/yichenwang-jluct/DCMI-YOLO/releases/tag/v1.0):

```
https://github.com/yichenwang-jluct/DCMI-YOLO/releases/download/v1.0/best.pt
```

> ⚠️ The weights were trained on sensitive endangered-species data and are released for **research reproducibility** only. Please use them responsibly and do not deploy them in ways that could facilitate locating or harming wildlife.

---

## License

This project is released under the GNU AGPL-3.0 license, inherited from Ultralytics YOLO, on which it is built. See `LICENSE` for the full text. If you use this code, you must comply with the AGPL-3.0 terms (including making source code available for networked use).

---

## Acknowledgements & Citation

This work is built upon Ultralytics YOLOv8:

> G. Jocher, A. Chaurasia, and J. Qiu. (2023). *Ultralytics YOLO* (Version 8.0.0) [Computer software]. https://github.com/ultralytics/ultralytics

If you use this repository in your research, please cite our paper:

```
@article{wang2026dcmiyolo,
  title   = {Edge-Deployable Low-Light Object Detection for Automated Nighttime Wildlife Monitoring: From Detection Accuracy to Ecological Indicators},
  author  = {Wang, Yichen and Qi, Shenao and He, Yuli and So, Chi Chiu},
  journal = {IEEE Access (under review)},
  year    = {2026},
  note    = {Code: https://github.com/yichenwang-jluct/DCMI-YOLO}
}
```

The citation will be updated with the volume, pages, and DOI after publication.
