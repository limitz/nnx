import math as _math
import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _F
import torchvision as _torchvision
import numpy as _np
import collections as _collections
from . import functional as _Fx

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

class ModuleFunction(_nn.Module):
    def __init__(self, *args, _module=_torch, _attr=None, _dereference=False, _postprocess=None,
                 **kwargs):
        super().__init__()
        if _attr is None:
            _attr = type(self).__name__.lower()
        
        self.function = getattr(_module, _attr)
        self.dereference = _dereference
        self.postprocess = _postprocess
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        if self.dereference:
            r = self.function(*x, *self.args, **self.kwargs)
        else:
            r = self.function(x, *self.args, **self.kwargs)
        if self.postprocess is not None and callable(self.postprocess):
            r = self.postprocess(r)
        return r
        
    def extra_repr(self):
        r = ""
        if len(self.args):
            r += ", ".join([str(v) for v in self.args])
        if len(self.kwargs):
            if len(r): r += ", "
            r += ", ".join([str(k) + "=" + str(v) for k,v in self.kwargs.items()])
        return r
    
class TensorFunction(_nn.Module):
    def __init__(self, *args, _attr=None, _postprocess=None, **kwargs):
        super().__init__()
        self.attr = _attr or type(self).__name__.lower()
        self.postprocess = _postprocess
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        attr = getattr(x, self.attr)
        
        if callable(attr):
            r = attr(*self.args, **self.kwargs)
        else:
            r = attr
        if self.postprocess is not None and callable(self.postprocess):
            r = self.postprocess(r)
        return r
            
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
class Conj(TensorFunction): ...
class Real(TensorFunction): ...
class Imag(TensorFunction): ...
class Angle(TensorFunction): ...
class Sum(TensorFunction): ...
class Mean(TensorFunction): ...
class Std(TensorFunction): ...
class Prod(TensorFunction): ...
class To(TensorFunction): ...
class Detach(TensorFunction): ...
class Log(TensorFunction): ...
class Exp(TensorFunction): ...
class Pow(TensorFunction): ...
class Sqrt(TensorFunction): ...
class Squeeze(TensorFunction): ...
class Unsqueeze(TensorFunction): ...
class Expand(TensorFunction): ...
class Contiguous(TensorFunction): ...
class Clamp(TensorFunction): ...
class Split(TensorFunction): ...
class Chunk(TensorFunction): ...
class Max(TensorFunction): 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, _postprocess=lambda x: x[0], **kwargs)
class Min(TensorFunction): 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, _postprocess=lambda x: x[0], **kwargs)
class MT(TensorFunction):
    def __init__(self):
        super().__init__(_attr="mT")

class Cat(ModuleFunction): ...
class Stack(ModuleFunction): ...

class FxFunction(ModuleFunction):
    def __init__(self, *args, _attr=None, **kwargs):
        super().__init__(*args, _module=_Fx, _attr=_attr, **kwargs)
        
class PadCat(FxFunction): ...
class PadStack(FxFunction): ...
class VoxelShuffle(FxFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, _attr="voxel_shuffle", **kwargs)
class VoxelUnshuffle(FxFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, _attr="voxel_unshuffle", **kwargs)
class Norm(FxFunction): ...
class Center(FxFunction): ...
class Reduce(FxFunction): ...

class FFTFunction(ModuleFunction):
    def __init__(self, *args, _attr=None, **kwargs):
        super().__init__(*args, _module=_torch.fft, _attr=_attr, **kwargs)

class FFT1(FFTFunction): ...
class FFT2(FFTFunction): ...
class FFTn(FFTFunction): ...
class IFFT1(FFTFunction): ...
class IFFT2(FFTFunction): ...
class IFFTn(FFTFunction): ...

class ForEach(_nn.Sequential):
    def forward(self, x):
        s = super()
        r = [s.forward(v) for v in x]
        if isinstance(x, _torch.Tensor):
            if len(r):
                s = r[0].shape
                if all(v.shape == s for v in s[1:]):
                    return torch.stack(r)
        return r
class Tee(_nn.Sequential):
    def forward(self, x):
        super().forward(x)
        return x

class Ignore(_nn.Sequential):
    def forward(self, x):
        return x

class ResizedLike(_nn.Sequential):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self.kwargs = kwargs
        
    def forward(self, x, **kwargs):
        s = x.shape[2:]
        return _Fx.interpolate(super().forward(x), s, **self.kwargs)

class RandnLike(_nn.Module):
    def forward(self, x, **kwargs):
        return _torch.randn_like(x)

class RandLike(_nn.Module):
    def forward(self, x, **kwargs):
        return _torch.rand_like(x)

class OnesLike(_nn.Module):
    def forward(self, x, **kwargs):
        return _torch.ones_like(x)

class ZerosLike(_nn.Module):
    def forward(self, x, **kwargs):
        return _torch.zeros_like(x)

class Loop(_nn.Sequential):
    def __init__(self, *args, n=0):
        super().__init__(*args)
        self.n = n
    
    def forward(self, x):
        for _ in range(self.n):    
            x = super().forward(x)
        return x
        

class AdaptiveCrossEntropyLoss(_nn.CrossEntropyLoss):
    def __init__(self, num_classes, betas=(0.9, 0.999, 0.99), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.betas = betas
        self.num_classes = num_classes
        self.register_buffer("scores", _torch.ones(num_classes))
        if self.weight is None:
            self.register_buffer("weight", _torch.ones(num_classes))
        
    def reset(self):
        with _torch.no_grad():
            self.scores.copy_(_torch.ones_like(self.scores))
        
    def forward(self, pred, target):
        if _torch.is_grad_enabled():
            with _torch.no_grad():
                correct = pred.argmax(-1) == target
                n = self.num_classes
                
                # update the scores
                total = _torch.bincount(target, minlength=n)
                #correct = _torch.bincount(target[correct], minlength=n)
                correct = pred.softmax(-1) * _F.one_hot(target,n)
                correct = correct.sum(0)
                classes = total.nonzero().view(-1)
                scores = self.scores.mul(self.betas[0])
                scores = scores + (correct/total).mul(1-self.betas[0])
                scores = scores[classes]
                self.scores[classes] = scores

                #update the weights
                weight = self.scores.sum() / (self.num_classes * self.scores)
                weight = self.weight.log() * self.betas[1] + weight.log() * (1-self.betas[2])
                weight = weight.exp()
                self.weight.copy_(weight)
            
        return super().forward(pred, target)

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

class VideoDataset(_torch.utils.data.Dataset):
    def __init__(self, path, start=0, length=10):
        eps = 1e-5
        self.video, *_  = _torchvision.io.read_video(path, 
                                                     start_pts=start, 
                                                     end_pts=start+length-eps, 
                                                     pts_unit="sec", 
                                                     output_format="TCHW")

    def __len__(self):
        return self.video.shape[0]

    def __getitem__(self, idx):
        frame = self.video[idx]/255
        frame = _TF.center_crop(frame, 480)
        frame = _F.interpolate(frame[None], (S,S))[0]
        return frame

