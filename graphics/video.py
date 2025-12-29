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

def plot_video(x, path=None, width=None, height=None, **kwargs):
    path = path or _os.path.join("generated", str(_uuid.uuid1())+".mp4")
    save_video(x, path, **kwargs)
    if width is not None and height is None:
        height = width * x.shape[-2] / x.shape[-1]
    elif height is not None and width is None:
        width = height * x.shape[-1] / x.shape[-2]
    _ipy_display.display(
        _ipy_display.Video(path,
                           width=width, 
                           height=height))

def save_video(x, path, fps=30, cols=None, 
               q=10, qmin=None, qmax=None, 
               codec="h264", **kwargs):
    assert codec in { "h264", "h265", "x264", "x265" }
    
    codecs = dict(h264="libx264", x264="libx264", 
                  h265="libx265", x265="libx265")
    
    default_options = {"qmin": str(qmin or qmax or q), 
                       "qmax":str(qmax or qmin or q), 
                       "profile":"main"}
    
    options = dict(h264=default_options, x264=default_options,
                   h265=default_options, x265=default_options)
    
    options = options[codec]
    codec = codecs[codec]
    
    v = _Fx.tensor(x)
    assert v.dim() == 4
    if v.dtype != _torch.uint8:
        v = v.clamp(0,1).mul(255).byte()
    px = (-v.shape[-1]) % 4
    py = (-v.shape[-2]) % 4
    if px>0 or py>0:
        v = _F.pad(v, (0,px,0,py))
    frames = v.permute(0,2,3,1)
    _torchvision.io.write_video(path, frames, fps=fps, 
                                video_codec=codec, 
                                options=options)
                                 
