import torch as _torch
import torch.nn.functional as _F
from .. import functional as _Fx


def _evaluate_bezier(control_points, t):
    pts = control_points  # (..., 1, N, 2)  ->  broadcast with t over S dim
    while pts.shape[-2] > 1:
        pts = (1 - t) * pts[..., :-1, :] + t * pts[..., 1:, :]
    return pts[..., 0, :]  # (..., S, 2)


def bezier_points(control_points, num_samples=100):
    control_points = _Fx.tensor(control_points).float()
    if control_points.dim() == 1:
        control_points = control_points.unflatten(-1, (-1, 2))
    t = _torch.linspace(0, 1, num_samples, device=control_points.device)
    # t: (..., S, 1, 1) to broadcast against cp (..., 1, N, 2)
    t = t.view(*([1] * (control_points.dim() - 2)), num_samples, 1, 1)
    cp = control_points.unsqueeze(-3)  # (..., 1, N, 2)
    return _evaluate_bezier(cp, t)


def render_bezier(dst, control_points, color=(1, 1, 1.), thickness=1, num_samples=None, inplace=False):
    if not inplace:
        dst = dst.clone()

    color = _Fx.tensor(color, device=dst.device).float()
    if color.dim() == 1:
        color = color.view(-1, 1)

    control_points = _Fx.tensor(control_points, device=dst.device).float()
    if control_points.dim() == 1:
        control_points = control_points.unflatten(-1, (-1, 2))

    h, w = dst.shape[-2:]
    if num_samples is None:
        num_samples = max(h, w) * 2

    pts = bezier_points(control_points, num_samples)  # (num_samples, 2)
    pts = pts.view(-1, 2)

    # rasterize with thickness
    gy = _torch.arange(h, device=dst.device).float()
    gx = _torch.arange(w, device=dst.device).float()
    yy, xx = _torch.meshgrid(gy, gx, indexing="ij")
    grid = _torch.stack((yy, xx), dim=-1)  # (H, W, 2)

    # distance from each pixel to nearest curve point
    # grid: (H, W, 1, 2), pts: (1, 1, S, 2)
    diff = grid.unsqueeze(-2) - pts.view(1, 1, -1, 2)
    dist = diff.pow(2).sum(-1).min(-1).values.sqrt()  # (H, W)

    mask = dist <= (thickness / 2.0)
    # broadcast color onto dst
    dst[..., mask] = color.expand(dst.shape[-3], -1)

    return dst


def render_bezier_aa(dst, control_points, color=(1, 1, 1.), thickness=1, num_samples=None, inplace=False):
    if not inplace:
        dst = dst.clone()

    color = _Fx.tensor(color, device=dst.device).float()
    if color.dim() == 1:
        color = color.view(-1, 1, 1)

    control_points = _Fx.tensor(control_points, device=dst.device).float()
    if control_points.dim() == 1:
        control_points = control_points.unflatten(-1, (-1, 2))

    h, w = dst.shape[-2:]
    if num_samples is None:
        num_samples = max(h, w) * 2

    pts = bezier_points(control_points, num_samples).view(-1, 2)

    gy = _torch.arange(h, device=dst.device).float()
    gx = _torch.arange(w, device=dst.device).float()
    yy, xx = _torch.meshgrid(gy, gx, indexing="ij")
    grid = _torch.stack((yy, xx), dim=-1)

    diff = grid.unsqueeze(-2) - pts.view(1, 1, -1, 2)
    dist = diff.pow(2).sum(-1).min(-1).values.sqrt()  # (H, W)

    radius = thickness / 2.0
    alpha = (radius + 0.5 - dist).clamp(0, 1)  # (H, W)

    dst = dst * (1 - alpha) + color * alpha

    return dst
