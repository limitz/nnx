import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _F
import torchvision as _torchvision
import math as _math
import matplotlib.pyplot as _plt
import IPython.display as _ipy_display
import tempfile as _tempfile
import uuid as _uuid
import os as _os
import time as _time
from .. import functional as _Fx
from . import color as _color
from . import functional as _GFx
#from . import text as _text

def rgb(image, dtype=_torch.float, **kwargs):
    if isinstance(image, str):
        image = _text.render_text(image, **kwargs)
    #else:
    #    image = _Fx.tensor(image)
    assert dtype in {_torch.float, _torch.uint8}
    if image.dtype in {_torch.cfloat, _torch.cdouble}:
        image = _torch.cat([image.real, image.imag], -3)
        image = _color.feature_to_rgb(image, **kwargs)
    if image.shape[-3] == 1:
        image = _torch.cat([image]*3, -3)
    elif image.shape[-3] == 2:
        image = _torch.cat([image] + [_torch.full_like(image, 0.5)], -3)
    elif image.shape[-3] > 3:
        image = _color.feature_to_rgb(image, **kwargs)
        image = image.clamp(0,1)
    if image.dtype != dtype:
        if dtype == _torch.float:
            image = image / 255
        elif dtype == _torch.uint8:
            image = image.clamp(0,1).mul(255).round().byte()
    return image
    
def rgb8(image, **kwargs):
    assert "dtype" not in kwargs
    return rgb(image, dtype=_torch.uint8, **kwargs) 

def alpha_blend_2d(dst, src):
    n,c,*_ = dst.shape
    if c == 3:
        dst = _F.pad(dst, [0,0,0,0,0,1], "constant", 1)
        
    alpha = src[:,-1:]
    dst = dst * (1-alpha) + src * alpha
    return dst[:,:c]

def alpha_blend_3d(image, blend_dim=-3):
    assert image.dim() >= 4
    d = image.unbind(blend_dim)
    r = _torch.zeros_like(d[0])
    for plane in d:
        if plane.shape[-3] == 4:
            alpha = plane[..., -1:, :, :]
        else:
            alpha = plane
        r = r * (1-alpha) + plane * alpha
    return r
       
def save(image, path, **kwargs):
    v = rgb8(image, **kwargs)
    if v.dim() > 3:
        v = _GFx.make_grid(v, **kwargs)
    v = v.cpu()
    if path.endswith(".png"):
        _torchvision.io.write_png(v, path)
    else:
        _torchvision.io.write_jpeg(v, path)


def plot(image, title="", width=20, height=None, **kwargs):
    v = rgb8(image, **kwargs) 
    height = height or (width * v.shape[-2] / v.shape[-1])
    if v.dim() > 3:
        v = _GFx.make_grid(v, **kwargs)
    
    v = v.permute(1,2,0)
    v = v.cpu().numpy()
    _plt.figure(figsize=(width, height))
    _plt.axis("off")
    if title: _plt.title(title)
    _plt.imshow(v)
    _plt.show()
                                 
