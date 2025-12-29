import torch as _torch
import torch.nn.functional as _F
import numpy as _np
import cv2 as _cv2
import math as _math
import matplotlib.pyplot as _plt

def nth_color(i, n, channels=3):
    return _torch.arange(channels).div(channels).add(i/n).mul(2*_math.pi).cos().mul(0.5).add(0.5)

def nth_color_u8(i,n,channels=3): 
    return nth_color(i,n,channels).mul(255).byte()

def n_colors(n, channels=3):
    return _torch.stack([nth_color(i, n, channels) for i in range(n)])

def n_colors_u8(n, channels=3):
    return _torch.stack([nth_color_u8(i, n, channels) for i in range(n)])

def rgb_to_yuv(rgb, clamp=True):
    if rgb.dtype == _torch.uint8: 
        rgb = rgb / 255
    m = _torch.tensor([
        [ 0.21260,  0.71520, 0.07220],
        [-0.09991, -0.33609, 0.43600],
        [ 0.61500, -0.55861,-0.05639]],
        device=rgb.device)
    yuv = (rgb.transpose(-3,-1) @ m.mT).transpose(-3,-1).contiguous()
    if clamp:
        yuv.select(-3, 0).clamp_(0,1)
        yuv.select(-3, 1).clamp_(-1,1)
        yuv.select(-3, 2).clamp_(-1,1)
    return yuv

def yuv_to_rgb(yuv, clamp=True):
    m = _torch.tensor([
        [1, 0.00000, 1.28033],
        [1,-0.21482,-0.38059],
        [1, 2.12798, 0.00000]], 
        device=yuv.device)
    rgb = (yuv.transpose(-3,-1) @ m.mT).transpose(-3,-1).contiguous()
    return rgb.clamp(0,1) if clamp else rgb

def feature_to_yuv(v, normalize=True):
    c = v.shape[-3]
    z = _torch.polar(
        _torch.ones(c), 
        _torch.linspace(0, 2*_math.pi, c))
    v = (v * z.to(v.device).view(-1,1,1)).sum(-3)
    if normalize:
        v = v.sub(v.mean()).div(v.std().add(1e-8))/2
    a = v.abs()
    v = _torch.stack((a*0.7, v.real, v.imag), -3)
    return v
    
def feature_to_rgb(v, **kwargs):
    return yuv_to_rgb(feature_to_yuv(v, **kwargs))
    