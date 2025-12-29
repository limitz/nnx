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
from .font import *
from .color import *
from .video import *
from .image import *
from .functional import *

def blend(dst, src, position, mode="bilinear", **kwargs):
    dst = dst.clone()
    return dst 