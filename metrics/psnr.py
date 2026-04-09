import torch as _torch
import math as _math
from ..graphics import rgb8 as _rgb8
from ..functional import reduce as _reduce

# 8x8 orthonormal DCT-II basis matrix: D[k,n] = w(k) * cos(pi*k*(2n+1)/16)
# Applied as 2D DCT via: D @ patch @ D.T
_D = _torch.cos(
    _torch.pi
    * _torch.arange(8).unsqueeze(1)
    * (2 * _torch.arange(8).unsqueeze(0) + 1)
    / 16.0
)
_D[0] *= 8.0 ** -0.5
_D[1:] *= (2.0 / 8) ** 0.5

_CSFCof = _torch.tensor([
    [1.608443, 2.339554, 2.573509, 1.608443, 1.072295, 0.643377, 0.504610, 0.421887],
    [2.144591, 2.144591, 1.838221, 1.354478, 0.989811, 0.443708, 0.428918, 0.467911],
    [1.838221, 1.979622, 1.608443, 1.072295, 0.643377, 0.451493, 0.372972, 0.459555],
    [1.838221, 1.513829, 1.169777, 0.887417, 0.504610, 0.295806, 0.321689, 0.415082],
    [1.429727, 1.169777, 0.695543, 0.459555, 0.378457, 0.236102, 0.249855, 0.334222],
    [1.072295, 0.735288, 0.467911, 0.402111, 0.317717, 0.247453, 0.227744, 0.279729],
    [0.525206, 0.402111, 0.329937, 0.295806, 0.249855, 0.212687, 0.214459, 0.254803],
    [0.357432, 0.279729, 0.270896, 0.262603, 0.229778, 0.257351, 0.249855, 0.259950],
])

_MaskCof = _torch.tensor([
    [0.390625, 0.826446, 1.000000, 0.390625, 0.173611, 0.062500, 0.038447, 0.026874],
    [0.694444, 0.694444, 0.510204, 0.277008, 0.147929, 0.029727, 0.027778, 0.033058],
    [0.510204, 0.591716, 0.390625, 0.173611, 0.062500, 0.030779, 0.021004, 0.031888],
    [0.510204, 0.346021, 0.206612, 0.118906, 0.038447, 0.013212, 0.015625, 0.026015],
    [0.308642, 0.206612, 0.073046, 0.031888, 0.021626, 0.008417, 0.009426, 0.016866],
    [0.173611, 0.081633, 0.033058, 0.024414, 0.015242, 0.009246, 0.007831, 0.011815],
    [0.041649, 0.024414, 0.016437, 0.013212, 0.009426, 0.006830, 0.006944, 0.009803],
    [0.019290, 0.011815, 0.011080, 0.010412, 0.007972, 0.010000, 0.009426, 0.010203],
])

# True at all (k,l) in [0:7,0:7] except (1,1) — positions where contrast masking applies
_MASK_AC = _torch.ones(7, 7, dtype=_torch.bool)
_MASK_AC[1, 1] = False


def _prep(a, b):
    a = _rgb8(a).float()
    b = _rgb8(b).float()
    if a.dim() > 3: a = a.flatten(0, -4)
    if b.dim() > 3: b = b.flatten(0, -4)
    for p, q in zip(a.unbind(0), b.unbind(0)):
        yield (p, q)


def _vari(x):
    # Sum of squared deviations (biased variance * N) per patch.
    # Matches numpy: np.var(x.flatten()) * x.size
    # x: (..., h, w)
    mean = x.mean(dim=(-2, -1), keepdim=True)
    return (x - mean).pow(2).sum(dim=(-2, -1))


def _maskeff(z, zdct, mc, mask_ac):
    # Contrast masking effect (Enorm) for a batch of 8x8 patches.
    # z: (B, 8, 8) spatial patches, zdct: (B, 8, 8) DCT coefficients
    # Returns: (B,)
    mc_7 = mc[:7, :7]
    m = (zdct[:, :7, :7].pow(2) * mc_7 * mask_ac).sum(dim=(-2, -1))  # (B,)

    pop = _vari(z)  # (B,) — total patch variance * N
    nonzero = pop != 0
    corners = (
        _vari(z[:, 0:3, 0:3])
        + _vari(z[:, 0:3, 4:7])
        + _vari(z[:, 4:7, 4:7])
        + _vari(z[:, 4:7, 0:3])
    )
    pop = _torch.where(nonzero, corners / pop, _torch.zeros_like(pop))

    return (m * pop).sqrt() / 32


def _psnrhvsm(img1, img2, wstep=8):
    # img1, img2: (3, H, W) float [0, 255] — output of _prep
    # Returns (p_hvs_m, p_hvs) as scalar tensors on the input device.
    with _torch.no_grad():
        dev = img1.device
        D       = _D.to(dev)
        csf     = _CSFCof.to(dev)
        mc      = _MaskCof.to(dev)
        mask_ac = _MASK_AC.to(dev)

        # Use only first channel (luminance proxy), as in the original algorithm
        a = img1[0]  # (H, W)
        b = img2[0]

        # Non-overlapping 8x8 patches: (H//8, W//8, 8, 8) → (B, 8, 8)
        a_p = a.unfold(0, 8, 8).unfold(1, 8, 8).reshape(-1, 8, 8)
        b_p = b.unfold(0, 8, 8).unfold(1, 8, 8).reshape(-1, 8, 8)

        # 2D DCT-II (orthonormal) via separable matrix multiply
        a_dct = D @ a_p @ D.T  # (B, 8, 8)
        b_dct = D @ b_p @ D.T

        # Per-patch masking effect; take the larger of the two
        mask_a = _maskeff(a_p, a_dct, mc, mask_ac)  # (B,)
        mask_b = _maskeff(b_p, b_dct, mc, mask_ac)
        mask_a = _torch.max(mask_a, mask_b)

        # Weighted squared error over 7x7 DCT AC coefficients
        diff  = (a_dct[:, :7, :7] - b_dct[:, :7, :7]).abs()  # (B, 7, 7)
        csf_7 = csf[:7, :7]
        mc_7  = mc[:7, :7]

        # PSNR-HVS: CSF-weighted error only
        s2 = (diff * csf_7).pow(2)  # (B, 7, 7)

        # PSNR-HVS-M: subtract masking threshold before CSF weighting (except at DC-adjacent bin (1,1))
        threshold = mask_a[:, None, None] / mc_7  # (B, 7, 7)
        u  = _torch.where(mask_ac, (diff - threshold).clamp(min=0), diff)
        s1 = (u * csf_7).pow(2)  # (B, 7, 7)

        S1 = s1.sum() / s1.numel()
        S2 = s2.sum() / s2.numel()

        sentinel = _torch.tensor(100000.0, device=dev)
        c        = _torch.tensor(255.0 * 255.0, device=dev)
        p_hvs_m  = _torch.where(S1 == 0, sentinel, 10 * (c / S1).log10())
        p_hvs    = _torch.where(S2 == 0, sentinel, 10 * (c / S2).log10())

    return p_hvs_m, p_hvs


def mse(a, b, reduction="mean"):
    r = [p.sub(q).pow(2).mean() / (255.0 ** 2) for p, q in _prep(a, b)]
    r = _torch.stack(r)
    return _reduce(r, reduction)


def image_psnr(a, b, reduction="mean"):
    r = [20 * _math.log10(255) - 10 * p.sub(q).pow(2).mean().log10() for p, q in _prep(a, b)]
    r = _torch.stack(r)
    return _reduce(r, reduction)


def image_psnr_hvs_m(a, b, wstep=8, reduction="mean"):
    r = [_psnrhvsm(p, q, wstep)[0] for p, q in _prep(a, b)]
    r = _torch.stack(r)
    return _reduce(r, reduction)


def image_psnr_hvs(a, b, wstep=8, reduction="mean"):
    r = [_psnrhvsm(p, q, wstep)[1] for p, q in _prep(a, b)]
    r = _torch.stack(r)
    return _reduce(r, reduction)
