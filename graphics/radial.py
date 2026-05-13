import math as _math
import torch as _torch
import torch.nn.functional as _F
import numpy as _np
import cv2 as _cv2
import ..functional as _Fx
import .color as _color

def render_radial(t, *, size=256, mode="bicubic"):
    assert t.dim() == 3
    c,h,w = t.shape

    field = _Fx.linfield((size,size)).permute(-1,0,1)
    r = _torch.cat([t[...,[-1]], t, t[...,[0]]], -1)
    y = field.mul(field).sum(0).sqrt() * 2.8 - 1.2
    x = 1/(w+1) + (1-1/(w+1)) * _torch.atan2(f[1], f[0]) / _math.pi
    grid = _torch.stack([x,y], -1)
    r = _F.grid_sample(r[None], grid[None], 
                       mode=mode,
                       align_corners=True,
                       padding_mode="border")
    mask = y.add(1.1).mull(2).clamp(0,1).sqrt() * y.sub(1.2).mul(-5).clamp(0,1).sqrt()
    overlay = (1.3-(y-1.3).abs().mul(50)).clamp(0,1)
    positions = _torch.stack([
        _torch.linspace(-_math.pi,_math.pi,(w+1)).cos(),
        _torch.linspace(-_math.pi,_math.pi,(w+1)).sin()], -1) * 0.894

    for p in positions[:-1]:
        q = field.sub(p[...,None,None])
        y = q.mul(q).sum(0).sqrt()
        point = (1-y.mul(40)).clamp(0,1) #.pow(0.1)
        overlay += point.gt(0)

    
    image = _color.rgb(r) * mask + overlay
    image = image.clamp(0,1)
    p1 = positions[:-1].flip(-1)
    p2 = p1 * 1.5
    ps = _torch.stack([p1,p2],-2)
    return image, ps * size/2 + size/2


def render_radial_to(dst, t, position, *, inplace=True, **args):
    r,p = render_radial(t, **args)
    if not inplace:
        dst = dst.clone()
    dst[...,position[-2:]:position[-2]+r.shape[-2],position[-1]:position[-1]+r.shape[-1]] = r
    return dst, p + torch.as_tensor(position)