import math as _math
import copy as _copy
import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _F
import torch.distributed as _dist
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
    def forward(self,x,r=None):
        r = r if r is not None else x
        return x + super().forward(r)

class Lambda(_nn.Module):
    def __init__(self, function, unsafe=False):
        super().__init__()
        self.function = function
        self.unsafe = unsafe
        
    def forward(self, x):
        f = self.function
        if isinstance(f, str) and f.startswith("lambda"):
            assert self.unsafe
            f = eval(f)
        assert callable(f)
        return f(x)

    def extra_repr(self):
        r = repr(self.function)
        if self.unsafe:
            r = r + ", unsafe=True"
        return r
            
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
    
class Select(TensorFunction): ...
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
class GreaterThan(TensorFunction):
    def __init__(self):
        super().__init__(_attr="gt")
class LessThan(TensorFunction):
    def __init__(self):
        super().__init__(_attr="lt")
class Equals(TensorFunction):
    def __init__(self):
        super().__init__(_attr="eq")
class Gt(TensorFunction):...
class Lt(TensorFunction):...
class Eq(TensorFunction):...

class Cat(ModuleFunction): ...
class Stack(ModuleFunction): ...

class FxFunction(ModuleFunction):
    def __init__(self, *args, _attr=None, **kwargs):
        super().__init__(*args, _module=_Fx, _attr=_attr, **kwargs)
        
class PadCat(FxFunction): ...
class PadStack(FxFunction): ...
class VoxelShuffle(FxFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, _attr="voxel_shuffle", **kwargs)
class VoxelUnshuffle(FxFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, _attr="voxel_unshuffle", **kwargs)

class SampleShuffle(FxFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, _attr="sample_shuffle", **kwargs)
class SampleUnshuffle(FxFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, _attr="sample_unshuffle", **kwargs)


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
                    return _torch.stack(r)
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
    def __init__(self, *args, n=1, copy=False):
        if copy:
            block = _nn.Sequential(*args)
            super().__init__(*[_copy.deepcopy(block) for _ in range(n)])
            self.n = 1
        else:
            super().__init__(*args)
            self.n = n
    
    def forward(self, x):
        for _ in range(self.n):    
            x = super().forward(x)
        return x

class FuzzyConv2d(_nn.Module):
    def __init__(self, in_features, out_features, kernel_size, 
                 noise_level=3e-2, noise_reference="none", bypass=False, 
                 **kwargs):
        super().__init__()
        assert noise_reference in { "none", "std", "mean", "max", "abs" }
        assert "padding_mode" not in kwargs or kwargs["padding_mode"] == "zeros"
        self.noise_reference = noise_reference
        self.noise_level = noise_level
        self.bypass = bypass
        self.body = _nn.Conv2d(in_features, out_features, kernel_size, **kwargs)
        self.kwargs = kwargs

    #def __getattribute__(self, name):
    #    if name in { "weight", "stride", "bias","dilation", "padding", "kernel_size", "in_features", "out_features", "padding_mode", "weight_fake_quant" }:
    #        return getattr(self.body, name)
    #    else:
    #        return super().__getattribute__(name)
            
    def forward(self, x):
        if self.bypass: return self.body(x)
        if hasattr(self.body, "weight_fake_quant"):
            w = self.body.weight_fake_quant(self.body.weight)
        else:
            w = self.body.weight
        if self.noise_reference == "none":
            w = w + _torch.randn_like(w) * self.noise_level
        elif self.noise_reference == "std":
            w = w + _torch.randn_like(w) * w.std() * self.noise_level
        elif self.noise_reference == "mean":
            w = w + _torch.randn_like(w) * w.mean() * self.noise_level
        elif self.noise_reference == "max":
            w = w + _torch.randn_like(w) * w.max() * self.noise_level
        elif self.noise_reference == "abs":
            w = w + _torch.randn_like(w) * w.abs() * self.noise_level
        else:
            assert False, "invalid noise reference"
        b = self.body.bias
        return _F.conv2d(x, w, b, 
                        stride=self.body.stride, 
                        padding=self.body.padding, 
                        groups=self.body.groups,
                        dilation=self.body.dilation)

class AdaptiveBoxBlurNd(_nn.Module):
    def __init__(self, kernel_sizes=None, channel_dim=1):
        super().__init__()
        assert channel_dim != 0
        self.channel_dim = channel_dim
        self.flows = None
        self.signs = None
        self.update_kernel_sizes(kernel_sizes)
            
    @staticmethod
    def _calculate_flows(k):
        dimensions = len(k.shape)-2
        assert k.shape[-1] == dimensions
        corners = _Fx.all_combinations(*[[-1,1]]*dimensions)
        corners = _torch.tensor(corners, device=k.device)
        grid = []
        flow = []
        for d in range(dimensions):
            s = k.shape[-d-2]
            i = [None]*dimensions
            i[-d-1] = slice(0,s)
            g = _torch.linspace(-1,1,s, device=k.device)[tuple(i)]
            g = g.expand_as(k[...,0])
            g = g - 1/s
            grid.append(g)
            flow.append(k[...,d]/s)
        grid = _torch.stack(grid, -1)
        flow = _torch.stack(flow, -1)
        flows = []
        for c in corners:
            f = grid + c * flow 
            flows.append(f)
        signs = corners.prod(-1)
        return flows, signs
    
    @staticmethod
    def _window_mean(x, flows, signs, areas, channel_dim=-1, 
                         padding_mode="reflection", eps=1e-5):
    
        assert channel_dim != 0
        
        if channel_dim < 0: channel_dim = x.dim() + channel_dim
        excl_channel_dim = (i for i in range(x.dim()) if i != channel_dim)
        excl_channel_dim = tuple(excl_channel_dim)
        mean = x.mean(excl_channel_dim, keepdim=True)
        std = x.std(excl_channel_dim, keepdim=True)
        x = x.sub(mean).div(std.add(eps))
        for d in excl_channel_dim[1:]:
            x = x.cumsum(d)
        
        s = 0
        if channel_dim != 1:
            to_channels_first = (0,channel_dim) + excl_channel_dim[1:]
            x = x.permute(*to_channels_first)
        for f,sign in zip(flows, signs):
            s += sign * _F.grid_sample(x, f, padding_mode=padding_mode, 
                                      align_corners=True)
        if channel_dim != 1:
            to_channels_orig = (to_channels_first.index(i) 
                                for i in range(x.dim()))
            s = s.permute(*to_channels_orig)
        
        return s.div(areas.add(eps)).mul(std).add(mean)
    
    def update_kernel_sizes(self, kernel_sizes):
        if kernel_sizes is not None:
            self.flows, self.signs = self._calculate_flows(kernel_sizes)
            self.kernel_sizes = kernel_sizes
    
    def forward(self, x, kernel_sizes=None):
        self.update_kernel_sizes(kernel_sizes)
        assert self.flows is not None
        assert self.signs is not None
        areas = self.kernel_sizes.prod(-1).unsqueeze(self.channel_dim)
        return self._window_mean(x, self.flows, self.signs, areas, 
                                 channel_dim=self.channel_dim)


        
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

class Sequential(_nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, *args, **kwargs):
        for i, layer in enumerate(self):
            if i == 0: x = layer(*args, **kwargs)
            else: x = layer(x)
        return x
        
class Affine(_nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.weight = _nn.parameter.Parameter(_torch.randn(channels))
        self.bias = _nn.parameter.Parameter(_torch.randn(channels))

    def forward(self, x):
        w = self.weight.view(self.channels, *[1 for _ in range(x.dim()-2)])
        b = self.bias.view(self.channels, *[1 for _ in range(x.dim()-2)])
        return x * w + b

class Trainer(_nn.Module):
    def __init__(self, module, loss, optimizer):
        super().__init__()
        self.module = module
        self.loss = loss
        self.optimizer = optimizer

    def state_dict(self):
        return dict(
            module=self.module.state_dict(),
            loss=self.loss.state_dict(),
            optimizer=self.optimizer.state_dict())

    def load_state_dict(self, state):
        self.module.load_state_dict(state["module"])
        self.loss.load_state_dict(state["loss"])
        self.optimizer.load_state_dict(state["optimizer"])

    def forward(self, x, target):
        pred = self.module(x)
        loss = self.loss(pred, target)
        return pred, loss
        
    def step(self):
        return self.optimizer.step()

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

class TanhL1Loss(_nn.Module):
    def __init__(self, tau=3, reduction="mean"):
        super().__init__()
        self.tau = tau
        self.reduction = reduction

    def forward(self, pred, target):
        assert pred.shape == target.shape
        loss = pred - target
        loss =  loss.mul(self.tau).tanh() * loss
        return _Fx.reduce(loss, self.reduction)
        
class GradientMSELoss(_nn.Module):
    def __init__(self, dim=None, spacing=1, edge_order=1, reduction="mean", keepdim=False):
        super().__init__()
        self.mse = _nn.MSELoss(reduction="none")
        self.dim = dim
        self.reduction = reduction
        self.keepdim = keepdim
        self.spacing = spacing
        self.edge_order = edge_order
        
    def forward(self, pred, target):
        assert pred.shape == target.shape
        dims = self.dim or list(range(2,pred.dim()))
        if not isinstance(dims, (tuple,list)): dims = [dims]
        dims = [d + pred.dim() if d < 0 else d for d in dims]
        pred = _torch.gradient(pred, dim=self.dim, spacing=self.spacing, 
                              edge_order=self.edge_order)
        target = _torch.gradient(target, dim=self.dim, spacing=self.spacing,
                                edge_order=self.edge_order)
        pred = _torch.stack(pred)
        target = _torch.stack(target)
        loss = self.mse(pred, target)
        if self.reduction == "none":
            return loss
        elif self.reduction == "dim":
            return loss.mean(dims, keepdim=self.keepdim)
        else:
            return _Fx.reduce(loss, self.reduction, keepdim=self.keepdim)
        
class DiffMSELoss(_nn.Module):
    def __init__(self, dim=None, reduction="mean", keepdim=False):
        super().__init__()
        self.mse = _nn.MSELoss(reduction="none")
        self.dim = dim
        self.reduction = reduction
        self.keepdim = keepdim

    def forward(self, pred, target):
        assert pred.shape == target.shape
        dims = self.dim or list(range(2,pred.dim()))
        if not isinstance(dims, (tuple,list)): dims = [dims]
        dims = [d + pred.dim() if d < 0 else d for d in dims]
        pred = [pred.diff(dim=d) for d in dims]
        target = [target.diff(dim=d) for d in dims]
        if self.reduction == "none":
            for x in (pred, target):
                for i,d in enumerate(dims):
                    p = x[i].transpose(d,-1)
                    p = (_F.pad(p,(0,1)) + _F.pad(p, (1,0))) / 2
                    x[i] = p.transpose(d,-1).contiguous()
            loss = _torch.stack([self.mse(p,t) for p,t in zip(pred, target)])
            return loss
        else:
            loss = _torch.stack([_Fx.reduce(self.mse(p,t), "mean", 
                                            dim=dims, keepdim=self.keepdim) 
                                 for p,t in zip(pred, target)])
            loss = loss.mean(0)
            
            if self.reduction != "dim":
                loss = _Fx.reduce(loss, self.reduction, keepdim=self.keepdim)
            return loss 
        
class Scope(_nn.Sequential):
    def __init__(self, *args, name="global"):
        super().__init__(*args)
        self._name = name
        self._scope = {}
        self._stack = {}
        self._store = {}
        self._pre_hook = self.register_forward_pre_hook(self.enter)

    @staticmethod
    def enter(self, x, **kwargs):
        device = _Fx.guess_device([x, self])
        if self._scope.get(device) is not None: 
            return
        self._scope[device] = self
        self._stack[device] = []
        self._store[device] = {}
        
        for m in self.modules():
            if isinstance(m, StackAccess):
                m._stack = self._stack
                m._store = self._store
                m._scope = self._scope

    def push(self, x, index=None, name=None, **kwargs):
        device = _Fx.guess_device([x, self])
        if index is not None:
            self._stack[device].insert(x, index)
        elif name is not None:
            self._store[device][name] = x
        else:
            self._stack[device].append(x)
        return x

    def pop(self, index=None, name=None, device=None, **kwargs):
        device = device or _Fx.guess_device([self])
        if index is not None:
            return self._stack[device].pop(index)
        elif name is not None:
            return self._store[device][name]
        else:
            return self._stack[device].pop(-1)

class StackAccess(Scope):
    def __init__(self):
        super().__init__(name = "_inherit")
        
class Push(StackAccess):
    def __init__(self, *index_or_name, index=None, name=None):
        super().__init__()
        if index is None and name is None:
            if len(index_or_name)==1:
                if isinstance(index_or_name[0], str):
                    name = index_or_name[0]
                else:
                    index = index_or_name[0]
        assert index is None or name is None, "use either index or name, not both"
        self.index = index
        self.name = name

    def forward(self, x):
        self.push(x, index=self.index, name=self.name)
        return x

    def extra_repr(self):
        if self.name is not None:
            r = f"\"{self.name}\""
        elif self.index is not None:
            r = f"{self.index}"
        else:
            r = ""
        return r

class Pop(StackAccess):
    def __init__(self, *index_or_name, index=None, name=None, reduction=None, dim=None, 
                 unsafe=False):
        super().__init__()
        if index is None and name is None:
            if len(index_or_name)==1:
                if isinstance(index_or_name[0], str):
                    name = index_or_name[0]
                else:
                    index = index_or_name[0]
        assert index is None or name is None, "use either index or name, not both"
        if isinstance(reduction, str) and reduction.startswith("lambda"):
            assert unsafe, "passing reduction as a string uses `eval` and could be " + \
                           "unsafe, pass unsafe=True if you're sure you want to do this"
            
        self.index = index
        self.name = name
        self.reduction = reduction
        self.dim = dim
        self.unsafe = unsafe
    
    def forward(self, x):
        device = _Fx.guess_device([x,self])
        r = self.pop(index=self.index, name=self.name, device=device)
        reduction = self.reduction
        
        if isinstance(reduction, str) and reduction.startswith("lambda"):
            assert self.unsafe, "unsafe must be set to True when evaluating str"
            reduction = eval(reduction)
        if callable(reduction):
            return reduction(x,r)
        elif reduction in {"skip", "add"}:
            return r + x
        elif reduction == "mul":
            return r * x    
        elif reduction == "max":
            return torch.maximum(x,r)
        elif reduction == "min":
            return torch.minimum(x,r)
        elif reduction == "mean":
            return x.add(r).div(2)
        elif reduction == "cat":
            return _torch.cat([x,r], self.dim or 1)
        elif reduction =="stack":
            return _torch.stack([x,r], self.dim or 1)
        else:
            return r
        
    def extra_repr(self):
        if self.name is not None:
            r = f"\"{self.name}\""
        elif self.index is not None:
            r = f"{self.index}"
        else:
            r = ""
        if self.reduction is not None:
            if len(r)>0: r = r + ", "
            r = r + f"reduction={self.reduction}"
        if self.dim is not None:
            if len(r)>0: r = r + ", "
            r = r + f"dim={self.dim}"
        if self.unsafe:
            if len(r)>0: r = r + ", "
            r = r + "unsafe=True"
        return r


class RecurrentBatchNorm2d(_nn.Module):
    def __init__(self, n, *args, **kwargs):
        super().__init__()
        self.inner = _nn.ModuleList([_nn.BatchNorm2d(*args, **kwargs) for _ in range(n)])
        self.cursor = 0

    def reset(self):
        self.cursor = 0

    def forward(self, x):
        r = self.inner[self.cursor](x)
        self.cursor += 1
        return r
        
class Between:
    def __init__(self, a, b, n=(), inclusive=True):
        self.a = a
        self.b = b
        self.n = n
        self.inclusive = inclusive
    
    def __call__(self,test=None):
        a = self.a() if callable(self.a) else self.a
        b = self.b() if callable(self.b) else self.b
        n = self.n() if callable(self.n) else self.n
        if test is not None:
            return _Fx.is_between(test, a, b, self.inclusive)
        else:
            return _Fx.sample_between(a, b, n, self.inclusive)
        