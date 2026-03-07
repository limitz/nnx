import torch as _torch
import torch.nn.functional as _F
import numpy as _np
import cv2 as _cv2
import math as _math
from .. import functional as _Fx
from .. import projection as _P

def border(x, color=(1,1,1.), thickness=1, inplace=False):
    color = _Fx.tensor(color).unflatten(-1,(-1,1))
    t = thickness
    mask = _torch.ones(x.shape[-2:], device=x.device, dtype=_torch.bool)
    mask[..., t:-t,t:-t] = 0
    if not inplace: x = x.clone()
    x[...,mask] = color
    return x
    
def inner_edge(x, threshold=0.5, background=None, thickness=1):
    if threshold is not None:
        m0 = x.gt(threshold).float()
    else:
        m0 = (~x.eq(background)).float()
    m0 = m0 + _F.max_pool2d(-m0,1+thickness*2,padding=thickness,stride=1)
    return m0.bool()

def outer_edge(x, threshold=0.5, background=0, thickness=1): 
    if threshold is not None:
        m0 = x.gt(threshold).float()
    else:
        m0 = (~x.eq(background)).float()
    m0 = _F.max_pool2d(m0,1+thickness*2,padding=thickness,stride=1) - m0
    return m0.bool()

def edge(x, threshold=0.5, background=0, thickness=1):
    if threshold is not None:
        m0 = x.gt(threshold).float()
    else:
        m0 = (~x.eq(background)).float()
    if thickness > 1:
        m0 = m0 - inner_edge(m0, background, threshold, thickness//2)
    m0 = outer_edge(m0, background, threshold, thickness - thickness//2)
    return m0.bool()

def highlight(x, mask, color, threshold=0.5, alpha=0.5, background_alpha=1, edge=outer_edge, **kwargs):
    edge = edge(mask, threshold=threshold, **kwargs)
    color = _Fx.tensor(color,device=x.device)
    if color.dim() == 0: color = color.view(-1)
    if color.dim() == 1: color = color.unflatten(-1,(-1,1,1))
    #if color.dim() == 3: color = color.expand(3,1,1)
    x = _torch.where(mask.gt(threshold).expand_as(x), x * (1-alpha) + color * alpha,x * background_alpha)
    r = _torch.where(edge.expand_as(x), color.expand_as(x), x)
    return r
    
def floodfill(t, pos, color):
    y,x = pos
    color = _Fx.tensor(color)
    replace = t[...,y,x,None,None]
    mask = _torch.ones_like(t,dtype=_torch.bool)
    mask[t != replace] = 0
    fill = _torch.ones_like(t)
    fill[mask] = 2
    fill[...,y,x] = -color
    prev = None
    while prev is None or not prev.eq(fill).all():
        prev = fill
        fill = mask * -_F.max_pool2d(-fill,3,stride=1,padding=1)
        
    t = t * (fill>=0) - fill * (fill<0)
    return t

def make_grid(x, cols=None, padding=1, pad_value=0, **kwargs):
    if isinstance(x, _np.ndarray): x = _torch.from_numpy(x)
    assert x.dim() <= 5
    while x.dim() < 5: x = x.unsqueeze(0)
    if cols:
        cols_pad = -x.shape[1] % cols
        rows = x.shape[0] * (x.shape[1] + cols - 1) // cols
    else:
        cols_pad = 0
        cols = x.shape[1]
        rows = x.shape[0]
    if cols > 1 or rows > 1:
        x = _F.pad(x, (0,padding,0,padding,0,0,0,cols_pad), "constant", pad_value)
    if cols < x.shape[1]:
        x = x.unflatten(1,(-1,cols)).flatten(0,1)
    x = _torch.cat(x.split(1,1),-1)
    x = _torch.cat(x.split(1,0),-2)
    x = x[0,0]
    if cols > 1 or rows > 1:
        x = _F.pad(x, (padding,0,padding,0), "constant", pad_value)
    return x


def valid_crop_rect(tlbr, shape, wrap=True, clamp=True):
    hwhw = _Fx.tensor(shape[-2:] * 2)
    tlbr = _Fx.tensor([int(v*s) if isinstance(v, float) else int(v) for v,s in zip(tlbr,hwhw)])
    if wrap:
        tlbr = [v+s if v < 0 else v for v,s in zip(tlbr,hwhw)]
    if clamp:
        tlbr = [v.clamp(0,s) for v,s in zip(tlbr,hwhw)]
    return tuple(v.item() for v in tlbr)
    
def blend(dst, x, position, alpha=None, mode="bilinear", inplace=True, **kwargs):
    dim = dst.dim()
    while dst.dim() < 4: dst = dst[None]
    while x.dim() < 4: x = x[None]
        
    if not inplace:
        dst = dst.clone()
    *n,c,h,w = x.shape
    if alpha is not None:
        x = x.expand(*n,3,h,w)
        alpha = _Fx.tensor(alpha).view(-1,1,1).expand_as(x)
        alpha = (~x.eq(alpha).all(-3,keepdim=True)).float()
        x = _torch.cat((x, alpha),-3)
    elif c < 4:
        x = _F.pad(x.expand(*n,3,h,w), (0,0,0,0,0,1), "constant", 1.)

    position = _Fx.tensor(position)
    if position.shape[-2:] == (2,3):
        position = position.expand(*dst.shape[:-3], -1, -1)
        grid = _F.affine_grid(position.float(), dst.shape, align_corners=False)
        x = _F.grid_sample(x, grid, mode=mode, align_corners=False)
        t,l,b,r = 0, 0, *dst.shape[-2:]
    elif position.shape[-2:] == (3,3):
        position = position.expand(*dst.shape[:-3], -1, -1)
        grid = _P.perspective_grid(position.float(), dst.shape)
        x = _F.grid_sample(x, grid, mode=mode, align_corners=False)
        t,l,b,r = 0, 0, *dst.shape[-2:]
    elif len(position) == 2:
        t,l = position
        b,r = t+h,l+w
    elif len(position) == 4:
        t,l,b,r = valid_crop_rect(position, dst.shape)
        h,w = b-t,r-l
        if x.shape[-2:] != (h,w):
            x = _F.interpolate(x, (h,w), mode=mode)
    
    dst[...,t:b,l:r] = dst[...,t:b,l:r] * (1-x[...,-1:,:,:]) + x[...,:-1,:,:]
    while dim < dst.dim(): 
        assert dst.shape[0] == 1
        dst = dst[0]
    
    return dst
    