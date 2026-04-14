import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _F
try:
    import torchaudio as _torchaudio
except (ImportError, OSError):
    _torchaudio = None
import math as _math
import matplotlib.pyplot as _plt
import IPython.display as _ipy_display
import tempfile as _tempfile
import uuid as _uuid
import os as _os
import time as _time
from .. import functional as _Fx

def render_waveform(audio, height=256, pool=1):
    p = _F.max_pool1d(audio, 2+pool, stride=pool, padding=1)
    n = -_F.max_pool1d(audio, 2+pool, stride=pool, padding=1)
    r = _torch.linspace(-1,1,height).unsqueeze(-1).expand(-1, p.shape[-1])
    r = _torch.where(r.gt(n)&r.lt(p), _torch.ones_like(r), _torch.zeros_like(r))
    return r
        

def plot_audio(x, path=None, width=None, height=None, encoding="mp4", **kwargs):
    if path is None:
        path = _os.path.join(_tempfile.gettempdir(), str(_uuid.uuid1()) + "." + encoding)
    save_audio(x, path, **kwargs)
    _ipy_display.display(
        _ipy_display.Audio(path))

def save_audio(x, path, sample_rate=44100, compression=None, **kwargs):
    v = _Fx.tensor(x)
    assert v.dtype == _torch.float
    assert v.dim() == 2
    v = v.clamp(-1,1)
    
    _torchaudio.save(path, v, sample_rate=sample_rate, compression=compression)
                                 
