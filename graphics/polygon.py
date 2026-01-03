import math as _math
import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _F
import cv2 as _cv2
import numpy as _np
from .. import functional as _Fx

def render_polygon(dst, polygon, color):
    polygon = _Fx.tensor(polygon)
    color = _Fx.tensor(color)
    