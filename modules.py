import math as _math
import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _F
import torchvision as _torchvision
import numpy as _np
import collections as _collections

class StaticConvNd(_nn.Module):
    def __init__(self, n, weight, bias=None, **kwargs):
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)
        self.kwargs = kwargs
        self._f = { 1: _F.conv1d, 2: _F.conv2d, 3: _F.conv3d }[n]
            
    def forward(self, x):
        return self._f(x, self.weight, self.bias, **self.kwargs)
    
    def extra_repr(self):
        r = f"weight=<tensor of shape {self.weight.shape}>"
        if self.bias is not None:
            r = r + f", bias=<tensor of shape {self.bias.shape}>"
        return r

class StaticConv1d(StaticConvNd):
    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, **kwargs)

class StaticConv2d(StaticConvNd):
    def __init__(self, *args, **kwargs):
        super().__init__(2, *args, **kwargs)

class StaticConv3d(StaticConvNd):
    def __init__(self, *args, **kwargs):
        super().__init__(3, *args, **kwargs)

class Skip(_nn.Sequential):
    def forward(self,x):
        return x + super().forward(x)

class TensorFunction(_nn.Module):
    def __init__(self, *args, _override_attr=None, **kwargs):
        super().__init__()
        self.attr = _override_attr or type(self).__name__.lower()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        attr = getattr(x, self.attr)
        if callable(attr):
            return attr(*self.args, **self.kwargs)
        else:
            return attr
            
    def extra_repr(self):
        r = ""
        if len(self.args):
            r += ", ".join([str(v) for v in self.args])
        if len(self.kwargs):
            if len(r): r += ", "
            r += ", ".join([str(k) + "=" + str(v) for k,v in self.kwargs.items()])
        return r
    
class Permute(TensorFunction): ...
class Transpose(TensorFunction): ...
class View(TensorFunction): ...
class Reshape(TensorFunction): ...
class Abs(TensorFunction): ...
class Neg(TensorFunction): ...
class Real(TensorFunction): ...
class Imag(TensorFunction): ...
class Angle(TensorFunction): ...
class Sum(TensorFunction): ...
class Mean(TensorFunction): ...
class Std(TensorFunction): ...
class Prod(TensorFunction): ...
    
class MT(TensorFunction):
    def __init__(self):
        super().__init__(_override_attr="mT")

class CompoundLoss(_nn.ModuleList):
    def __init__(self, losses, strict=True):
        assert isinstance(losses, (dict, list, tuple)), "invalid argument"
        assert len(losses) > 0, "need at least 1 loss function"
        if isinstance(losses, dict):
            self.alphas = list(losses.values())
            super().__init__(list(losses.keys()))
        elif isinstance(losses, (tuple, list)):
            self.alphas = [1.] * len(losses)
            super().__init__(losses)
        else:
            assert False, "should not get here"
        self.strict = strict
        
    def forward(self, pred, target, **kwargs):
        r = []
        for alpha, module in zip(self.alphas, self):
            if self.strict:
                r.append(alpha * module(pred, target, **kwargs))
            else:
                try:
                    r.append(alpha * module(pred, target, **kwargs))
                except TypeError:
                    r.append(alpha * module(pred, target))
        s = r[0].shape
        if all(v.shape == s for v in r[1:]):
            return sum(r)
        else:
            return r