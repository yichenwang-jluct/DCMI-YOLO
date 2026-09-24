# Wiring DCMI-YOLO into Ultralytics

Four files:

| file | what it is |
|---|---|
| `yolov8-DCMI.yaml` | the model config — SCI-Gamma + 6× Star-ADown + 3× MultiSEAM on a standard YOLOv8 PANet head |
| `dcmi_modules.py` | `SCIGamma`, `StarADown`, `MultiSEAM` |
| `dcmi_loss.py` | DF-CIoU (Eq. 13–15) and Repulsion Loss (Eq. 9–12) |
| `param_check.py` | verifies the build reproduces 2.85 M / 8.1 GFLOPs before you publish it |

> **Read this first.** The equations, constants and layer placement below are taken
> from the manuscript and are exact. Three things are *not* stated in the paper and
> are therefore reconstructions: the width/depth/stage count of SCI-Gamma's
> illumination network, and the internal width of a MultiSEAM CSMM branch. Run
> `param_check.py`; if the deltas do not match Table 8, ship **your** modules, not
> these. Publishing code that does not reproduce the reported numbers is worse than
> publishing none.

---

## 1. Register the modules

Copy `dcmi_modules.py` to `ultralytics/nn/modules/dcmi.py`, then:

```python
# ultralytics/nn/modules/__init__.py
from .dcmi import SCIGamma, StarADown, MultiSEAM
__all__ += ("SCIGamma", "StarADown", "MultiSEAM")
```

```python
# ultralytics/nn/tasks.py  — inside parse_model()
from ultralytics.nn.modules import SCIGamma, StarADown, MultiSEAM
```

`parse_model` decides how a module gets its channel arguments by which set it is
listed in. The three need different handling:

```python
# the set that receives (c1, c2) — add StarADown and MultiSEAM here
if m in {Classify, Conv, ConvTranspose, ..., StarADown, MultiSEAM}:
    c1, c2 = ch[f], args[0] if args else ch[f]
    if c2 != nc:
        c2 = make_divisible(min(c2, max_channels) * width, 8)
    args = [c1, c2, *args[1:]]
```

`SCIGamma` needs neither: it is channel-preserving and takes only its own
arguments, so leave it out of that set. `parse_model` then falls through to
`c2 = ch[f]` and calls `SCIGamma(3)` from the YAML's `[3]`, which is what you want.

`MultiSEAM` must be in the set — the YAML writes it as `[]`, so without
registration it would be called with no arguments and fail on the required `c1`.

---

## 2. The config

`yolov8-DCMI.yaml` mirrors the paper exactly:

```
0   SCIGamma [3]          low-light frontend
1   Conv [64, 3, 2]       P1/2  — stem stays Conv on purpose
2   StarADown [128]       P2/4
4   StarADown [256]       P3/8
6   StarADown [512]       P4/16
8   StarADown [1024]      P5/32
...
17  MultiSEAM []          P3/8  ┐
21  MultiSEAM []          P4/16 ├ one per detection scale
25  MultiSEAM []          P5/32 ┘
26  Detect [[17, 21, 25]]
```

**Six** stride-2 positions are replaced, not seven. The stem at layer 1 takes a
3-channel input and `chunk(2, 1)` cannot split 3 channels evenly, so it stays a
`Conv`. This is also what makes the arithmetic work: replacing six positions costs
−0.36 M, which is the −0.34 M of Table 8.

The head is YOLOv8's standard PANet. An RT-DETR-style hybrid encoder that
projects every lateral to one width gives ~1.9 M parameters, not 2.85 M.

---

## 3. DF-CIoU

Replace the CIoU call in `BboxLoss.forward`:

```python
from ultralytics.utils.dcmi_loss import df_ciou

# was: iou = bbox_iou(pred[fg], target[fg], xywh=False, CIoU=True)
iou = df_ciou(pred_bboxes[fg_mask], target_bboxes[fg_mask],
              epoch=self.epoch, epochs=self.epochs)     # d0=0.2, u0=0.8
loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
```

The interval of Eq. (13) depends on the epoch, which Ultralytics does not pass
into the criterion. Push it in from a callback:

```python
def _set_epoch(trainer):
    bl = trainer.model.criterion.bbox_loss
    bl.epoch, bl.epochs = trainer.epoch, trainer.epochs

model.add_callback("on_train_epoch_start", _set_epoch)
```

and give `BboxLoss.__init__` the two attributes (`self.epoch = 0`,
`self.epochs = 200`) so the first call before any callback still works.

At epoch 0 the interval is `[0, 1]` and DF-CIoU is numerically identical to
CIoU — that is the paper's design, so a sanity run at epoch 0 must match your
CIoU baseline to the last decimal. By the final epoch it has narrowed to
`[0.2, 0.8]`.

---

## 4. Repulsion Loss

Eq. (12) adds two terms to the YOLOv8 objective with λ_RepGT = λ_RepBox = σ = 0.5.
In `v8DetectionLoss.__call__`, after the assigner has produced `fg_mask` and
`target_gt_idx`:

```python
from ultralytics.utils.dcmi_loss import repulsion_loss, LAMBDA_REPGT, LAMBDA_REPBOX

rep_gt = rep_box = 0.0
for i in range(batch_size):                       # per image — see the docstring
    m = fg_mask[i]
    if not m.any():
        continue
    g = gt_bboxes[i][mask_gt[i].squeeze(-1)]      # this image's ground truth
    if g.numel() == 0:
        continue
    a, b = repulsion_loss(
        pred_bboxes[i][m] * stride_tensor.squeeze(-1)[m, None],
        target_bboxes[i][m],
        target_gt_idx[i][m].long(),
        g,
    )
    rep_gt, rep_box = rep_gt + a, rep_box + b
rep_gt, rep_box = rep_gt / batch_size, rep_box / batch_size

loss[0] += LAMBDA_REPGT * rep_gt + LAMBDA_REPBOX * rep_box
```

Two things to get right, because both silently produce wrong numbers:

* **Per image.** `fg_mask` spans the batch. Pooled together, boxes from different
  images would repel each other.
* **One coordinate space.** `pred_bboxes` is in stride units inside the loss while
  `target_bboxes` may already have been divided by `stride_tensor`. Put both in the
  same space before calling, or IoG comes out meaningless.

Section III-C states that the "MultiSEAM" ablation column covers the attention
module **and** these two loss terms, so the +0.011 mAP in Table 8 belongs to the
pair. Do not present the module alone as the source of that gain.

---

## 5. Verify before publishing

```bash
python param_check.py --cfg yolov8-DCMI.yaml --nc 17
```

Then the per-module ablations. The decisive one is Star-ADown's delta: it must be
about **−0.34 M**. A positive delta means the module is replacing an `ADown`
rather than a stride-2 `Conv`, i.e. the config is not the one behind the paper.

Finally, confirm which config actually trained the reported checkpoint:

```bash
grep "^model:" runs/train/exp15/args.yaml
```

That line is the ground truth. Everything above is a reconstruction of it from the
manuscript, and where the two disagree, the checkpoint wins.
