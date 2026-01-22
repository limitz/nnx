import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _F
import torchaudio as _torchaudio
import math as _math
import matplotlib.pyplot as _plt
import IPython.display as _ipy_display
import tempfile as _tempfile
import uuid as _uuid
import os as _os
import time as _time
from .. import functional as _Fx

def plot_audio(x, path=None, width=None, height=None, **kwargs):
    path = path or _os.path.join("generated", str(_uuid.uuid1())+".mp4")
    save_audio(x, path, **kwargs)
    _ipy_display.display(
        _ipy_display.Audio(path))

def save_audio(x, path, sample_rate=44100, compression=None, **kwargs):
    v = _Fx.tensor(x)
    assert v.dtype == _torch.float
    assert v.dim() == 2
    v = v.clamp(-1,1)
    
    _torchaudio.save(path, v, sample_rate=sample_rate, compression=compression)
                                 
