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

class MoveDim(FxFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, _attr="move_dim", **kwargs)

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
class Interpolate(FxFunction): ...
    
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
                if all(v.shape == s for v in r[1:]):
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
                         padding_mode="border", eps=1e-5):
    
        assert channel_dim != 0
        if channel_dim < 0: channel_dim = x.dim() + channel_dim
        excl_channel_dim = tuple(i for i in range(1,x.dim()) if i != channel_dim)
        for d in excl_channel_dim:
            c = x.cumsum(d)
            m = (c.max(d,keepdim=True)[0] - c.min(d,keepdim=True)[0]) / 2
            x = (c-m)
                
        s = 0
        if channel_dim != 1:
            to_channels_first = (0, channel_dim) + excl_channel_dim
            x = x.permute(*to_channels_first)
        
        for f,sign in zip(flows, signs):
            s = s + sign * _F.grid_sample(x, f, mode="bilinear", 
                                          padding_mode=padding_mode, 
                                          align_corners=False)
            
        if channel_dim != 1:
            to_channels_orig = (to_channels_first.index(i) 
                                for i in range(x.dim()))
            s = s.permute(*to_channels_orig)
        
        return s.div(areas)
    
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


class AdaptiveLocalNormNd(_nn.Module):
    def __init__(self, hidden_dim, radius=(3,31), 
                 nonlinearity=_nn.GELU, mlp_expand=1, 
                 padding_mode="zeros",
                 eps=1e-8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.register_buffer("eps", _Fx.tensor(eps))
        self.register_buffer("radius", _Fx.tensor(radius))
        self.blur = AdaptiveBoxBlurNd()
        self.padding_mode = padding_mode
        self.kernel_size = _nn.Sequential(
            MoveDim(1,-1),
            _nn.Linear(hidden_dim, hidden_dim*mlp_expand),
            _nn.LayerNorm(hidden_dim * mlp_expand),
            nonlinearity(),
            _nn.Linear(hidden_dim*mlp_expand, 2),
            _nn.Sigmoid())
            
    def forward(self, x):
        q,r = self.radius
        rr = r//2+1
        # TODO unsure if the padding should be r or r//2, opt +1
        x = _F.pad(x, (rr,)*4, mode=self.padding_mode)
        ks = self.kernel_size(x)
        ks = ks * r.sub(q) + q
        self.blur.update_kernel_sizes(ks)
        x1 = self.blur(x)
        x2 = self.blur(x.pow(2)) 
        # might produce negative values due to cumsum on float...
        v = x2 - x1.pow(2)
        std = v.clamp(self.eps).sqrt()
        x = (x - x1) / std
        x = x[...,rr:-rr,rr:-rr]
        return x
        
class AdaptiveCrossEntropyLoss(_nn.CrossEntropyLoss):
    def __init__(self, num_classes, adapt="weight", betas=(0.9, 0.999, 0.999),  *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert adapt in { "weight", "score" }
        self.betas = betas
        self.num_classes = num_classes
        self.adapt = adapt
        self.register_buffer("counts", _torch.ones(num_classes))
        self.register_buffer("scores", _torch.ones(num_classes))
        if self.weight is None:
            self.register_buffer("weight", _torch.ones(num_classes))
        
    def reset(self):
        with _torch.no_grad():
            self.counts.copy_(_torch.ones_like(self.counts))
            self.scores.copy_(_torch.ones_like(self.scores))
        
    def forward(self, pred, target):
        if _torch.is_grad_enabled():
            with _torch.no_grad():
                correct = pred.argmax(1) == target
                n = self.num_classes
                
                # update the scores
                total = _torch.bincount(target.view(-1), minlength=n)
                #correct = _torch.bincount(target[correct], minlength=n)
                correct = pred.softmax(1) * _Fx.one_hot(target,n,dim=1)
                correct = correct.transpose(0,1).flatten(1).sum(1)
                classes = total.nonzero().view(-1)
                scores = self.scores.mul(self.betas[0])
                scores = scores + (correct/total).mul(1-self.betas[0])
                scores = scores[classes]
                self.scores[classes] = scores
                total = self.counts.mul(self.betas[0]) + total.mul(1-self.betas[0])
                self.counts.copy_(total)

                w = self.scores if self.adapt == "score" else self.counts
                
                #update the weights
                weight = (w.sum() / (self.num_classes * w)) / self.num_classes
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

class L1Loss(_nn.L1Loss):
    def forward(self, pred, target):
        if pred.dtype in { _torch.cfloat, _torch.cdouble }:
            r = super().forward(
                _torch.view_as_real(pred),
                _torch.view_as_real(target))
            if r.dim() == pred.dim() + 1 and r.shape[-1] == 2:
                r = r.sum(-1)
            return r
        else:
            return super().forward(pred, target)

class MSELoss(_nn.MSELoss):
    def forward(self, pred, target):
        if pred.dtype in { _torch.cfloat, _torch.cdouble }:
            r = super().forward(
                _torch.view_as_real(pred),
                _torch.view_as_real(target))
            if r.dim() == pred.dim() + 1 and r.shape[-1] == 2:
                r = r.sum(-1)
            return r
        else:
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
            return _torch.maximum(x,r)
        elif reduction == "min":
            return _torch.minimum(x,r)
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

class ReLUPow(_nn.Module):
    def __init__(self, gamma=1.0, delta=0.0):
        super().__init__()
        self.register_buffer("gamma", _Fx.tensor(gamma))
        self.register_buffer("delta", _Fx.tensor(delta))
        
    def forward(self,x):
        return x.add(self.delta).relu().pow(self.gamma).sub(self.delta.pow(2))


class Gate(_nn.Module):
    def __init__(self, dim=-1, nonlinearity=_nn.Sigmoid):
        super().__init__()
        self.dim = dim
        self.nonlinearity = nonlinearity()

    def forward(self, x):
        a,b = x.chunk(2, self.dim)
        return a * self.nonlinearity(b)

class ELUPow(_nn.Module):
    def __init__(self, gamma=1.0, alpha=1.0):
        super().__init__()
        self.register_buffer("gamma", _Fx.tensor(gamma))
        self.register_buffer("alpha", _Fx.tensor(alpha))

    def forward(self, x):
        return _torch.where(x>0, x.add(1).pow(self.gamma).sub(1), self.alpha * (x.exp().sub(1)))

class ReLUSquared(ReLUPow):
    def __init__(self, alpha=1.0):
        super().__init__(2.0, alpha)

class ELUSquared(ELUPow):
    def __init__(self, alpha=1.0):
        super().__init__(2.0, alpha)
        
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


# --- Normalization layers with optional complex (cfloat/cdouble) dtype ---
# For complex dtypes, mean is complex and variance is real (E[|x-mean|^2]),
# so std is real and division is well defined. Affine weight/bias are complex.

def _is_complex_dtype(dtype):
    return dtype in {_torch.cfloat, _torch.cdouble}

def _real_dtype_of(dtype):
    return _torch.double if dtype == _torch.cdouble else _torch.float

def _moments(x, dims, is_complex, keepdim=True):
    mean = x.mean(dims, keepdim=keepdim)
    centered = x - (mean if keepdim else mean.view(
        [x.shape[i] if i not in dims else 1 for i in range(x.dim())]))
    if is_complex:
        var = centered.real.pow(2).mean(dims, keepdim=keepdim) \
            + centered.imag.pow(2).mean(dims, keepdim=keepdim)
    else:
        var = centered.pow(2).mean(dims, keepdim=keepdim)
    return mean, var, centered


class BatchNormNd(_nn.Module):
    _allowed_ranks = None

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, device=None, dtype=_torch.float):
        super().__init__()
        if dtype is None: dtype = _torch.float
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.dtype = dtype
        self.is_complex = _is_complex_dtype(dtype)
        var_dtype = _real_dtype_of(dtype) if self.is_complex else dtype

        if affine:
            self.weight = _nn.Parameter(_torch.ones(num_features, dtype=dtype, device=device))
            self.bias = _nn.Parameter(_torch.zeros(num_features, dtype=dtype, device=device))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if track_running_stats:
            self.register_buffer("running_mean", _torch.zeros(num_features, dtype=dtype, device=device))
            self.register_buffer("running_var", _torch.ones(num_features, dtype=var_dtype, device=device))
            self.register_buffer("num_batches_tracked",
                                 _torch.tensor(0, dtype=_torch.long, device=device))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

    def forward(self, x):
        if self._allowed_ranks is not None:
            assert x.dim() in self._allowed_ranks, \
                f"expected input rank in {self._allowed_ranks}, got {x.dim()}"
        dims = [0] + list(range(2, x.dim()))
        view = [1, -1] + [1] * (x.dim() - 2)
        use_running = (not self.training) and self.track_running_stats

        if use_running:
            mean_v = self.running_mean.view(view)
            var_v = self.running_var.view(view)
            centered = x - mean_v
        else:
            mean, var, centered = _moments(x, dims, self.is_complex, keepdim=True)
            mean_v, var_v = mean, var
            if self.training and self.track_running_stats:
                with _torch.no_grad():
                    m = mean.detach().view(-1)
                    v = var.detach().view(-1)
                    self.running_mean.mul_(1 - self.momentum).add_(m * self.momentum)
                    self.running_var.mul_(1 - self.momentum).add_(v * self.momentum)
                    self.num_batches_tracked.add_(1)

        out = centered / (var_v + self.eps).sqrt()
        if self.affine:
            out = out * self.weight.view(view) + self.bias.view(view)
        return out

    def extra_repr(self):
        r = f"{self.num_features}, eps={self.eps}, momentum={self.momentum}"
        r += f", affine={self.affine}, track_running_stats={self.track_running_stats}"
        if self.dtype != _torch.float:
            r += f", dtype={self.dtype}"
        return r


class BatchNorm1d(BatchNormNd):
    _allowed_ranks = (2, 3)

class BatchNorm2d(BatchNormNd):
    _allowed_ranks = (4,)

class BatchNorm3d(BatchNormNd):
    _allowed_ranks = (5,)


class InstanceNormNd(_nn.Module):
    _allowed_ranks = None

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False,
                 track_running_stats=False, device=None, dtype=_torch.float):
        super().__init__()
        if dtype is None: dtype = _torch.float
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.dtype = dtype
        self.is_complex = _is_complex_dtype(dtype)
        var_dtype = _real_dtype_of(dtype) if self.is_complex else dtype

        if affine:
            self.weight = _nn.Parameter(_torch.ones(num_features, dtype=dtype, device=device))
            self.bias = _nn.Parameter(_torch.zeros(num_features, dtype=dtype, device=device))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if track_running_stats:
            self.register_buffer("running_mean", _torch.zeros(num_features, dtype=dtype, device=device))
            self.register_buffer("running_var", _torch.ones(num_features, dtype=var_dtype, device=device))
            self.register_buffer("num_batches_tracked",
                                 _torch.tensor(0, dtype=_torch.long, device=device))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

    def forward(self, x):
        if self._allowed_ranks is not None:
            assert x.dim() in self._allowed_ranks, \
                f"expected input rank in {self._allowed_ranks}, got {x.dim()}"
        dims = list(range(2, x.dim()))
        view = [1, -1] + [1] * (x.dim() - 2)
        use_running = (not self.training) and self.track_running_stats

        if use_running:
            mean_v = self.running_mean.view(view)
            var_v = self.running_var.view(view)
            centered = x - mean_v
        elif not dims:
            centered = _torch.zeros_like(x)
            var_v = _torch.zeros(1, dtype=_real_dtype_of(self.dtype) if self.is_complex else self.dtype,
                                 device=x.device)
            mean_v = x  # unused
        else:
            mean, var, centered = _moments(x, dims, self.is_complex, keepdim=True)
            mean_v, var_v = mean, var
            if self.training and self.track_running_stats:
                with _torch.no_grad():
                    m = mean.view(x.shape[0], x.shape[1]).mean(0).detach()
                    v = var.view(x.shape[0], x.shape[1]).mean(0).detach()
                    self.running_mean.mul_(1 - self.momentum).add_(m * self.momentum)
                    self.running_var.mul_(1 - self.momentum).add_(v * self.momentum)
                    self.num_batches_tracked.add_(1)

        out = centered / (var_v + self.eps).sqrt()
        if self.affine:
            out = out * self.weight.view(view) + self.bias.view(view)
        return out

    def extra_repr(self):
        r = f"{self.num_features}, eps={self.eps}, momentum={self.momentum}"
        r += f", affine={self.affine}, track_running_stats={self.track_running_stats}"
        if self.dtype != _torch.float:
            r += f", dtype={self.dtype}"
        return r


class InstanceNorm1d(InstanceNormNd):
    _allowed_ranks = (2, 3)

class InstanceNorm2d(InstanceNormNd):
    _allowed_ranks = (4,)

class InstanceNorm3d(InstanceNormNd):
    _allowed_ranks = (5,)


class LayerNorm(_nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True,
                 bias=True, device=None, dtype=_torch.float):
        super().__init__()
        if dtype is None: dtype = _torch.float
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.dtype = dtype
        self.is_complex = _is_complex_dtype(dtype)

        if elementwise_affine:
            self.weight = _nn.Parameter(_torch.ones(self.normalized_shape, dtype=dtype, device=device))
            if bias:
                self.bias = _nn.Parameter(_torch.zeros(self.normalized_shape, dtype=dtype, device=device))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        n = len(self.normalized_shape)
        assert tuple(x.shape[-n:]) == self.normalized_shape, \
            f"expected last {n} dims to be {self.normalized_shape}, got {tuple(x.shape[-n:])}"
        dims = list(range(x.dim() - n, x.dim()))
        _, var, centered = _moments(x, dims, self.is_complex, keepdim=True)
        out = centered / (var + self.eps).sqrt()
        if self.elementwise_affine:
            out = out * self.weight
            if self.bias is not None:
                out = out + self.bias
        return out

    def extra_repr(self):
        r = f"{self.normalized_shape}, eps={self.eps}"
        r += f", elementwise_affine={self.elementwise_affine}"
        if self.dtype != _torch.float:
            r += f", dtype={self.dtype}"
        return r


class GroupNorm(_nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True,
                 device=None, dtype=_torch.float):
        super().__init__()
        if dtype is None: dtype = _torch.float
        assert num_channels % num_groups == 0, \
            "num_channels must be divisible by num_groups"
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.dtype = dtype
        self.is_complex = _is_complex_dtype(dtype)

        if affine:
            self.weight = _nn.Parameter(_torch.ones(num_channels, dtype=dtype, device=device))
            self.bias = _nn.Parameter(_torch.zeros(num_channels, dtype=dtype, device=device))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        assert x.shape[1] == self.num_channels
        N, C = x.shape[0], x.shape[1]
        spatial = x.shape[2:]
        grouped = x.reshape(N, self.num_groups, C // self.num_groups, *spatial)
        dims = list(range(2, grouped.dim()))
        _, var, centered = _moments(grouped, dims, self.is_complex, keepdim=True)
        out = (centered / (var + self.eps).sqrt()).reshape(N, C, *spatial)
        if self.affine:
            view = [1, -1] + [1] * (x.dim() - 2)
            out = out * self.weight.view(view) + self.bias.view(view)
        return out

    def extra_repr(self):
        r = f"{self.num_groups}, {self.num_channels}, eps={self.eps}, affine={self.affine}"
        if self.dtype != _torch.float:
            r += f", dtype={self.dtype}"
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


class CausalConv1d(_nn.Conv1d):
    def forward(self, x):
        p = self.kernel_size[0] // 2
        x = _F.pad(x, (p, 0))
        x = super().forward(x)
        return x[..., :x.shape[-1]-p]


class CausalConv2d(_nn.Conv2d):
    def __init__(self, *args, direction="xy", **kwargs):
        super().__init__(*args, **kwargs)
        assert direction in {"x", "y", "xy"}
        self.direction = direction

    def forward(self, x):
        py = self.kernel_size[0] // 2 if "y" in self.direction else 0
        px = self.kernel_size[1] // 2 if "x" in self.direction else 0
        x = _F.pad(x, (px, 0, py, 0))
        x = super().forward(x)
        return x[..., :x.shape[-2]-py, :x.shape[-1]-px]


class CausalNorm1d(_nn.Module):
    def __init__(self, groups=1, num_features=None, dim=-1, eps=1e-10, affine=None):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.groups = groups
        self.num_features = num_features
        if affine is None:
            affine = num_features is not None
        self.affine = affine
        if affine:
            assert num_features is not None
            self.weight = _nn.Parameter(_torch.ones(num_features))
            self.bias = _nn.Parameter(_torch.zeros(num_features))
        else:
            self.weight = None
            self.bias = None

    def forward(self, x):
        token_dim = self.dim % x.dim()
        channel_dim = token_dim - 1
        num_channels = x.size(channel_dim)
        group_size = num_channels // self.groups
        seq_len = x.size(-1)

        x = x.unflatten(channel_dim, (self.groups, group_size))
        t = _torch.arange(1, seq_len + 1, dtype=x.dtype, device=x.device).view(1, 1, 1, seq_len)

        channel_sum = x.sum(dim=-2, keepdim=True)
        mean = channel_sum.cumsum(dim=-1) / (group_size * t)
        var = (x.pow(2).sum(dim=-2, keepdim=True).cumsum(dim=-1) / (group_size * t) - mean.pow(2)).clamp(min=0)

        x = (x - mean) / (var + self.eps).sqrt()
        x = x.flatten(channel_dim, channel_dim + 1)

        if self.weight is not None:
            x = x * self.weight.view(-1, 1) + self.bias.view(-1, 1)
        return x

    def extra_repr(self):
        r = f"groups={self.groups}"
        if self.num_features is not None:
            r += f", num_features={self.num_features}"
        r += f", affine={self.affine}"
        if self.dim != -1:
            r += f", dim={self.dim}"
        return r




class Parse2d(_nn.Sequential):
    def __init__(self, string, hidden_dim=None, kernel_size=3):
        self.string = string
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        super().__init__(*_parse_block(string, hidden_dim, dim=2, kernel_size=kernel_size))

    def extra_repr(self):
        if self.hidden_dim is not None:
            return f"string=\"{self.string}\", hidden_dim={self.hidden_dim}"
        else:
            return f"string=\"{self.string}\""

class Parse1d(_nn.Sequential):
    def __init__(self, string, hidden_dim=None, kernel_size=3):
        self.string = string
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        super().__init__(*_parse_block(string, hidden_dim, dim=1, kernel_size=kernel_size))

    def extra_repr(self):
        if self.hidden_dim is not None:
            return f"string=\"{self.string}\", hidden_dim={self.hidden_dim}"
        else:
            return f"string=\"{self.string}\""

def _parse_block(config, hidden_dim=None, dim=2, kernel_size=3):
    ConvNd = getattr(_nn, f"Conv{dim}d")
    InstanceNormCls = globals()[f"InstanceNorm{dim}d"]
    BatchNormCls = globals()[f"BatchNorm{dim}d"]
    ks = kernel_size
    pad = (ks - 1) // 2
    s = [[]]
    i = o = hidden_dim
    dtype = _torch.float
    for idx in range(len(config)):
        c = config[idx]
        d = config[idx+1] if idx < len(config)-1 else None
        if c in { "C", "c", "L", "l" , "N", "n"}:
            if d == "[":
                e = config[idx+1:].index("]")
                v = config[idx+2:idx+1+e]
                if "," in v:
                    i,o = [int(vv) for vv in v.split(",")]
                else:
                    o = int(v)
                idx = idx + 1 + e

        if c == "[":
            e = config[idx:].index("]")
            v = config[idx+1:idx+e]
            i = o = int(v)
            idx = idx + e
        elif c == "C":
            s[-1].append(ConvNd(i, o, ks, padding=pad, dtype=dtype))
            i = o
        elif c == "c":
            s[-1].append(ConvNd(i, o, ks, padding=pad, dtype=dtype, bias=False))
            i = o
        elif c == "z":
            dtype = _torch.cfloat
            s[-1].append(To(dtype))
            i = o
        elif c == "L":
            s[-1].append(ConvNd(i, o, 1, dtype=dtype))
            i = o
        elif c == "l":
            s[-1].append(ConvNd(i, o, 1, dtype=dtype, bias=False))
            i = o
        elif c == "B":
            s[-1].append(BatchNormCls(i, dtype=dtype))
        elif c == "b":
            s[-1].append(BatchNormCls(i, dtype=dtype, affine=False))
        elif c == "I":
            s[-1].append(InstanceNormCls(i, dtype=dtype, affine=True))
        elif c == "i":
            s[-1].append(InstanceNormCls(i, dtype=dtype, affine=False))
        elif c == "N":
            s[-1].append(GroupNorm(o, i, dtype=dtype, affine=True))
            o = i
        elif c == "n":
            s[-1].append(GroupNorm(o, i, dtype=dtype, affine=False))
            o = i
        elif c == "E":
            assert dtype not in {_torch.cfloat, _torch.cdouble}
            s[-1].append(_nn.ELU())
        elif c == "R":
            assert dtype not in {_torch.cfloat, _torch.cdouble}
            s[-1].append(_nn.ReLU())
        elif c == "G":
            assert dtype not in {_torch.cfloat, _torch.cdouble}
            s[-1].append(_nn.GELU())
        elif c == "S":
            s[-1].append(_nn.Sigmoid())
        elif c == "T":
            s[-1].append(_nn.Tanh())
        elif c == "(":
            s.append([])
        elif c == ")":
            r = s.pop(-1)
            r = Skip(*r)
            s[-1].append(r)
    assert len(s) == 1
    return s[-1]
    #if len(s[-1]) == 1: return s[-1][0]
    #else: return _nn.Sequential(*s[-1])
            
