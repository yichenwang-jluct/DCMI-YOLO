# -*- coding: utf-8 -*-
"""
DCMI-YOLO modules, written from the equations in the manuscript.

    SCIGamma    Section III-A, Eq. (1)-(7)
    StarADown   Section III-B, Eq. (8)
    MultiSEAM   Section III-C

Drop this file in as `ultralytics/nn/modules/dcmi.py` and register the three
classes (see INTEGRATION.md).

WHAT IS EXACT AND WHAT IS NOT
-----------------------------
Exact, taken straight from the paper:
  * the closed-form gamma   gamma = 1 / (1 + alpha * (1 - L))          Eq. (5)
    with alpha = 2.0, verified against every value in Tables 1 and 2
  * the SCI-Net iteration   x_{t+1} = x_t + H(v_t),  v_t = y + K(y / x_t)   Eq. (1)-(3)
  * the output mapping      I_out = A * I_in ** gamma,  A = 1            Eq. (4)
  * Star-ADown's forward pass, which reproduces the ~3.0*C^2 of Table 3

NOT stated in the paper, so exposed as arguments you must set to match the
checkpoint that produced the reported numbers:
  * SCIGamma: `channels`, `layers`, `stages` of the illumination network
  * MultiSEAM: the internal width of each CSMM branch
Check them with `python param_check.py` before trusting any complexity number:
the paper reports SCI-Gamma at +0.02 M / +0.2 GFLOPs and MultiSEAM at
+0.20 M / +0.5 GFLOPs over the YOLOv8n baseline (Table 8).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SCIGamma", "StarADown", "MultiSEAM"]


def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard YOLO Conv (Conv2d + BN + SiLU). Use ultralytics' own if importing."""

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ---------------------------------------------------------------------------
# Section III-A : SCI-Gamma
# ---------------------------------------------------------------------------
class _ResidualBlock(nn.Module):
    """H_theta / K_vartheta: a small weight-shared residual predictor."""

    def __init__(self, c_in: int, channels: int, layers: int):
        super().__init__()
        self.in_conv = nn.Sequential(nn.Conv2d(c_in, channels, 3, 1, 1), nn.ReLU(inplace=True))
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1), nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
            for _ in range(layers)
        )
        self.out_conv = nn.Conv2d(channels, c_in, 3, 1, 1)

    def forward(self, x):
        f = self.in_conv(x)
        for b in self.blocks:
            f = f + b(f)                     # residual, as in SCI-Net
        return self.out_conv(f)


class SCIGamma(nn.Module):
    """
    SCI-Gamma low-light frontend (Section III-A).

    Illumination is refined by the weight-shared iteration of Eq. (1)-(3):

        z_t = y / x_t                 enhanced image at stage t      Eq. (2)
        s_t = K(z_t)                  self-calibration
        v_t = y + s_t                 guided input
        u_t = H(v_t)                  residual prediction            Eq. (1), after Eq. (3)
        x_{t+1} = x_t + u_t           updated illumination,  x_0 = y

    The enhanced image is y / x_T, to which a closed-form gamma is applied:

        I_out = A * I_in ** gamma,   A = 1                           Eq. (4)
        gamma = 1 / (1 + alpha * (1 - L))                            Eq. (5)

    L is the normalised mean luminance of each image in [0, 1]. gamma carries no
    gradient (`torch.no_grad`) and adds no parameters; H and K are still trained
    by the detection loss, which is what keeps the frontend end-to-end trainable.

    Args:
        c (int): image channels, 3 for RGB. Declared in the YAML as `[3]`.
        alpha (float): brightening slope of Eq. (7). The paper grid-searches
            {1.0, 1.5, 2.0, 2.5, 3.0, 4.0} in Table 2 and adopts 2.0.
        stages (int): iterations of Eq. (1). NOT stated in the paper - set it to
            whatever your trained checkpoint used.
        channels, layers (int): width and depth of H and K. NOT stated in the
            paper - see the module docstring.
    """

    def __init__(self, c: int = 3, alpha: float = 2.0, stages: int = 1,
                 channels: int = 8, layers: int = 1, eps: float = 1e-4):
        super().__init__()
        assert alpha > 0, "alpha must be positive (Eq. 5)"
        self.alpha, self.stages, self.eps = float(alpha), int(stages), float(eps)
        self.H = _ResidualBlock(c, channels, layers)      # weight-shared across stages
        self.K = _ResidualBlock(c, channels, layers)      # self-calibration
        # luminance weights (ITU-R BT.601), registered so .to(device) follows the model
        self.register_buffer("_luma", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))

    @torch.no_grad()
    def _gamma(self, img: torch.Tensor) -> torch.Tensor:
        """gamma = 1 / (1 + alpha * (1 - L)), one scalar per image. Eq. (5)."""
        if img.shape[1] == 3:
            lum = (img * self._luma.to(img.dtype)).sum(1, keepdim=True)
        else:
            lum = img.mean(1, keepdim=True)
        L = lum.clamp(0, 1).mean(dim=(2, 3), keepdim=True)          # L in [0, 1]
        return 1.0 / (1.0 + self.alpha * (1.0 - L))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        x = y                                                        # x_0 = y
        for _ in range(self.stages):
            z = y / x.clamp_min(self.eps)                            # z_t = y (/) x_t
            v = y + self.K(z)                                        # v_t = y + K(z_t)
            x = x + self.H(v)                                        # x_{t+1} = x_t + H(v_t)
            x = x.clamp(self.eps, 1.0)
        enhanced = (y / x).clamp(0.0, 1.0)                           # SCI output
        return enhanced.pow(self._gamma(enhanced))                   # Eq. (4)


# ---------------------------------------------------------------------------
# Section III-B : Star-ADown
# ---------------------------------------------------------------------------
class StarADown(nn.Module):
    """
    Star-ADown (Section III-B, Eq. 8): ADown whose terminal concatenation is
    replaced by a ReLU6-guided star operation and a 1x1 projection.

        x~        = AvgPool_2x2, stride 1            smooth high-frequency noise
        x1, x2    = split(x~)                        channel split
        y1        = Conv_3x3,s2(x1)                  spatial path
        y2        = Conv_1x1(MaxPool_3x3,s2(x2))     texture path
        out       = Proj_1x1( ReLU6(y1) (*) y2 )     star fusion

    Parameter count is 2.25*C^2 + 0.25*C^2 + 0.5*C^2 = 3.0*C^2, matching Table 3.
    Multiplicative fusion couples the branches at every location: a weak response
    in either branch suppresses the output, which filters single-branch noise
    responses under low light.

    The stem (layer 1, 3-channel input) is deliberately NOT replaced - chunk(2, 1)
    cannot split 3 channels evenly.
    """

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)     # spatial branch
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)     # texture branch
        self.proj = Conv(self.c, c2, 1, 1, 0)         # restore channels
        self.act = nn.ReLU6()

    def forward(self, x):
        x = F.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        y1 = self.cv1(x1)
        y2 = self.cv2(F.max_pool2d(x2, 3, 2, 1))
        return self.proj(self.act(y1) * y2)


# ---------------------------------------------------------------------------
# Section III-C : MultiSEAM
# ---------------------------------------------------------------------------
class _CSMM(nn.Module):
    """
    Channel-Space Multi-scale Module: one MultiSEAM branch at a given patch size.

    Depthwise-separable convolutions with residual connections, then a 1x1
    cross-channel reorganisation, as described in Section III-C. Patch size counts
    feature-map cells; the patch embedding and its transpose are depthwise so the
    branch stays cheap at the wide P5 scale.
    """

    def __init__(self, c: int, patch: int, depth: int = 1):
        super().__init__()
        self.patch = patch
        self.embed = nn.Sequential(
            nn.Conv2d(c, c, patch, patch, groups=c), nn.SiLU(), nn.BatchNorm2d(c)
        )
        self.body = nn.ModuleList(
            nn.ModuleDict({
                "dw": nn.Sequential(nn.Conv2d(c, c, 3, 1, 1, groups=c), nn.SiLU(), nn.BatchNorm2d(c)),
                "pw": nn.Sequential(nn.Conv2d(c, c, 1), nn.SiLU(), nn.BatchNorm2d(c)),
            })
            for _ in range(depth)
        )
        self.unembed = nn.ConvTranspose2d(c, c, patch, patch, groups=c)

    def forward(self, x):
        h, w = x.shape[-2:]
        ph = (self.patch - h % self.patch) % self.patch
        pw = (self.patch - w % self.patch) % self.patch
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
        f = self.embed(x)
        for m in self.body:
            f = m["pw"](f + m["dw"](f))          # residual depthwise, then pointwise
        f = self.unembed(f)
        return f[..., :h, :w]


class MultiSEAM(nn.Module):
    """
    MultiSEAM (Section III-C): three CSMMs in parallel at patch sizes {6, 7, 8}.

    Their outputs are average-pooled, channel-expanded by a two-layer fully
    connected interaction, and combined by element-wise multiplication; the result
    gates the input. One block sits at each detection scale, immediately before
    the head, so P3/8, P4/16 and P5/32 each get occlusion-aware refinement.

    NOTE: the paper does not state the internal width of a CSMM branch. This
    implementation keeps every branch at full width with depthwise patch
    embeddings. Run `param_check.py` and compare against the +0.20 M of Table 8;
    if it does not match, your trained module differs internally and you should
    ship yours rather than this one.

    Args:
        c1, c2 (int): input/output channels (equal - the block is channel-preserving).
        patch_sizes (tuple): {6, 7, 8}; Table 10 grid-searches the combination.
        reduction (int): bottleneck of the fully connected interaction.
    """

    def __init__(self, c1: int, c2: int = None, patch_sizes=(6, 7, 8),
                 reduction: int = 16, depth: int = 1):
        super().__init__()
        c2 = c1 if c2 is None else c2
        assert c1 == c2, "MultiSEAM is channel-preserving"
        self.branches = nn.ModuleList(_CSMM(c1, p, depth) for p in patch_sizes)
        hidden = max(c1 // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(c1, hidden, bias=False), nn.ReLU(inplace=True),
            nn.Linear(hidden, c1, bias=False), nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        gate = None
        for br in self.branches:
            y = F.adaptive_avg_pool2d(br(x), 1).view(b, c)           # average-pool
            y = self.fc(y)                                           # ChannelExp
            gate = y if gate is None else gate * y                   # element-wise product
        return x * torch.exp(gate).view(b, c, 1, 1)
