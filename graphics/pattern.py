import torch as _torch
import torch.nn.functional as _F
import numpy as _np
import cv2 as _cv2
import math as _math
import matplotlib.pyplot as _plt
from .. import modules as _nnx
from .. import functional as _Fx
from . import text as _text

def render_random_pattern(size, pattern_size=_nnx.Between((2,2),(5,5)), num_colors=_nnx.Between(2,5), 
                          roll=True, colorspace="palette"):
    
    if callable(pattern_size): pattern_size = pattern_size()
    if callable(num_colors): num_colors = num_colors()
    pattern = (_torch.randperm(_np.prod(pattern_size)) % num_colors).view(pattern_size)
    if roll:
        roll_dim = _Fx.sample_between(0,1)
        roll_val = _Fx.sample_between(0,pattern_size[roll_dim]-1)

    
    r = pattern.repeat(
        -(size[-2]//-pattern_size[-2]), 
        -(size[-1]//-pattern_size[-1]))
    
    if roll:
        step = pattern_size[1-roll_dim]
        r = r.transpose(roll_dim, -1)
        for i in range(0, r.shape[0], step):
            r[i:] = r[i:].roll(roll_val, dims=-1)
            r = r.transpose(roll_dim, -1)
    
    r = r[None, :size[-2],:size[-1]]
    if colorspace == "rgb":
        r = _text.palette_to_rgb(r)
    return r