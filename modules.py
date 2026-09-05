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
        y = super().forward(r)
        # Tuple/list input (e.g. wrapping a Parallel/CrossNorm pair): residual is
        # applied element-wise so (a,b) + (ya,yb) -> (a+ya, b+yb), not concatenation.
        if isinstance(x, (tuple, list)):
            return tuple(xi + yi for xi, yi in zip(x, y))
        return x + y

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

class ExpHardCap(_nn.Module):
    """min(exp(x), cap): exp with a hard upper bound (input clamped at log cap)."""
    def __init__(self, cap=6.0):
        super().__init__()
        self.cap = cap
        self.log_cap = _math.log(cap)

    def forward(self, x):
        return x.clamp(max=self.log_cap).exp()

    def extra_repr(self):
        return f"cap={self.cap}"


class ExpSoftCap(_nn.Module):
    """exp(softmin(x, log cap)) with knee sharpness tau (tau=1 sigmoid cap, tau->inf hard cap)."""
    def __init__(self, cap=6.0, tau=1.0):
        super().__init__()
        self.cap = cap
        self.tau = tau
        self.log_cap = _math.log(cap)

    def forward(self, x):
        return (self.log_cap - _F.softplus(self.tau * (self.log_cap - x)) / self.tau).exp()

    def extra_repr(self):
        return f"cap={self.cap}, tau={self.tau}"

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

class ChannelNorm(_nn.LayerNorm):
    def forward(self, x):
        x = x.transpose(1,-1)
        x = super().forward(x)
        x = x.transpose(1,-1)
        return x

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
    """Apply the wrapped (weight-shared) sequence to each input stream, returning a tuple.

    Weight-shared analogue of :class:`Parallel`: where ``Parallel`` runs the i-th
    sub-module on the i-th input, ``ForEach`` runs the *same* wrapped sequence on every
    input. Accepts the streams as separate args (``forward(a, b)``) or as a single
    tuple/list (``forward((a, b))``) — like :class:`Parallel`/:class:`CrossNormNd`, so the
    three compose in a :class:`Sequential`. Does NOT iterate a tensor's leading/batch dim;
    a lone tensor is treated as a single stream (returns a 1-tuple).
    """
    def forward(self, *inputs):
        if len(inputs) == 1 and isinstance(inputs[0], (tuple, list)):
            inputs = tuple(inputs[0])
        fwd = super().forward
        return tuple(fwd(v) for v in inputs)
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
        
class AdaptiveGuidedFilter(_nn.Module):
    """Guided filter (He, Sun & Tang 2013) with per-pixel, decoder-predicted window
    size *and* regularization epsilon.

    A small decoder reads the guide and emits, per pixel, ``dims`` box-window widths
    (mapped into ``radius``) and one ``eps`` (mapped log-uniformly into ``eps_range``).
    The guided-filter coefficients

        a = cov(I,p) / (var(I) + eps);  b = mean(p) - a*mean(I);  q = mean(a)*I + mean(b)

    are then computed with :class:`AdaptiveBoxBlurNd` (per-pixel box mean) in place of the
    fixed box mean used by :func:`nnx.functional.guided_filter`. This is the
    :class:`AdaptiveLocalNormNd` decoder->adaptive-blur pattern applied to edge-aware
    filtering. Guide channels are filtered independently (separable per-channel guided
    filter), matching the fixed functional version.

    forward(x, guide=None) -> filtered x; ``guide`` defaults to ``x`` (self-guided)."""

    def __init__(self, guide_dim, hidden_dim=None, dims=2, radius=(3,31),
                 eps_range=(1e-4, 1e-1), nonlinearity=_nn.GELU, mlp_expand=1,
                 padding_mode="replicate"):
        super().__init__()
        assert dims in (1, 2, 3)
        # A width-1 hidden layer makes the following LayerNorm emit 0 for every pixel
        # (LayerNorm over a single feature), collapsing the per-pixel params to constants;
        # guard a useful minimum so a 1-channel guide still yields spatial variation.
        hidden_dim = hidden_dim or max(16, guide_dim)
        self.dims = dims
        self.padding_mode = padding_mode
        self.register_buffer("radius", _Fx.tensor(radius, dtype=_torch.float32))
        lo, hi = eps_range
        self.register_buffer("log_eps", _torch.log(_Fx.tensor([lo, hi], dtype=_torch.float32)))
        self.blur = AdaptiveBoxBlurNd(channel_dim=1)
        self.decode = _nn.Sequential(
            MoveDim(1,-1),
            _nn.Linear(guide_dim, hidden_dim*mlp_expand),
            _nn.LayerNorm(hidden_dim*mlp_expand),
            nonlinearity(),
            _nn.Linear(hidden_dim*mlp_expand, dims+1),
            _nn.Sigmoid())

    def _decode_params(self, guide):
        # guide: [B, C, *spatial] -> (kernel sizes [B,*spatial,dims] channels-last for the
        # blur, eps [B,1,*spatial] channels-first to broadcast over the blurred maps).
        klo, khi = self.radius
        elo, ehi = self.log_eps
        params = self.decode(guide)                              # [B,*spatial,dims+1]
        ks = params[..., :self.dims] * (khi - klo) + klo
        eps = (elo + params[..., self.dims:] * (ehi - elo)).exp()
        return ks, eps.movedim(-1, 1)

    def forward(self, x, guide=None):
        if guide is None: guide = x
        rr = int(self.radius[1])//2 + 1
        pad = (rr,) * (2*self.dims)
        I = _F.pad(guide, pad, mode=self.padding_mode)
        p = _F.pad(x, pad, mode=self.padding_mode)

        ks, eps = self._decode_params(I)
        self.blur.update_kernel_sizes(ks)
        blur = self.blur
        mean_I = blur(I)
        mean_p = blur(p)
        var_I = blur(I*I) - mean_I*mean_I
        cov_Ip = blur(I*p) - mean_I*mean_p
        a = cov_Ip / (var_I + eps)
        b = mean_p - a*mean_I
        q = blur(a)*I + blur(b)

        crop = (slice(None), slice(None)) + (slice(rr, -rr),) * self.dims
        return q[crop]


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
                n = self.num_classes

                # mask out ignored / out-of-range targets before gathering stats
                valid = (target >= 0) & (target < n)
                if self.ignore_index is not None:
                    valid = valid & (target != self.ignore_index)
                safe = target.where(valid, _torch.zeros_like(target))

                # update the scores
                total = _torch.bincount(safe[valid].view(-1), minlength=n)
                # summed softmax mass placed on the true class, per class
                onehot = _Fx.one_hot(safe, n, dim=1) * valid.unsqueeze(1)
                correct = pred.softmax(1) * onehot
                correct = correct.transpose(0,1).flatten(1).sum(1)
                classes = total.nonzero().view(-1)
                self.scores[classes] = self.scores[classes].mul(self.betas[0]) \
                    + (correct[classes] / total[classes]).mul(1-self.betas[0])
                self.counts.copy_(self.counts.mul(self.betas[0]) + total.mul(1-self.betas[0]))

                w = self.scores if self.adapt == "score" else self.counts

                #update the weights
                weight = (w.sum() / (self.num_classes * w)) / self.num_classes
                weight = self.weight.log() * self.betas[1] + weight.log() * (1-self.betas[1])
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


class Parallel(_nn.ModuleList):
    """Run N inputs through N parallel sub-modules and return their outputs as a tuple.

    Holds ``n`` sub-modules; ``forward`` pairs the ``i``-th input with the ``i``-th
    sub-module (``out[i] = module[i](input[i])``) and returns ``tuple(out)``. Inputs may be
    given as separate args (``forward(a, b)``) or as a single tuple/list
    (``forward((a, b))``); a **single tensor** is *forked* (broadcast) to every branch,
    while a tuple/list whose length matches the branch count is *zipped* one-per-branch.
    This is what lets it sit first in a :class:`Sequential` (fork the incoming stream) and
    also follow a :class:`CrossNormNd` (zip the returned pair) — mirroring CrossNorm's
    dual signature so the two compose directly for multi-stream architectures::

        Sequential(
            Parallel(convA, convB),   # independent per-stream work
            CrossNorm2d(c),           # couples the streams (shared alpha-weighted field)
            Parallel(convA2, convB2),
        )

    Construct as ``Parallel(m0, m1, ...)`` or ``Parallel([m0, m1, ...])``.
    """
    def __init__(self, *modules):
        if len(modules) == 1 and isinstance(modules[0], (list, tuple)):
            modules = modules[0]
        super().__init__(modules)

    def forward(self, *inputs):
        n = len(self)
        if len(inputs) == 1:
            x = inputs[0]
            if isinstance(x, (list, tuple)) and len(x) == n:
                inputs = tuple(x)          # zip: one input per branch
            else:
                inputs = (x,) * n          # fork: broadcast a single input to all branches
        assert len(inputs) == n, \
            f"Parallel: got {len(inputs)} input(s) for {n} branch(es)"
        return tuple(module(x) for module, x in zip(self, inputs))


class Index(_nn.Module):
    """Select element ``index`` from a tuple/list (indexable) input.

    Used to collapse a multi-stream :class:`Parallel`/:class:`CrossNormNd` pair back to a
    single tensor by picking one stream. In the Parse grammar this is the standalone
    ``@n`` operator (insertable anywhere a tuple is present): ``<...|...>@1`` parses to
    ``... , Index(1)``.
    """
    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        return x[self.index]

    def extra_repr(self):
        return str(self.index)


class Affine(_nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        self.weight = _nn.parameter.Parameter(_torch.randn(shape))
        self.bias = _nn.parameter.Parameter(_torch.randn(shape))

    def forward(self, x):
        if isinstance(self.shape, int):
            w = self.weight.view(self.channels, *[1 for _ in range(x.dim()-2)])
            b = self.bias.view(self.channels, *[1 for _ in range(x.dim()-2)])
        elif isinstance(self.shape, (tuple, list)):
            w = self.weight
            b = self.bias
        else:
            assert False, "invalid shape"
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

class Autodidact(_nn.Module):
    """Self-taught per-element loss weighting.

    Wraps a pointwise loss ``lfn`` (any criterion exposing a ``reduction`` attr,
    e.g. ``nn.MSELoss`` / ``nn.BCEWithLogitsLoss``). The wrapped model is expected
    to emit *two* stacked chunks along ``dim``: the prediction and a confidence map.
    The confidence is standardised and exponentiated to a positive weight
    ``w = exp(z(c)) + eps``; the per-element loss is scaled by ``w / w.detach()**2``
    (denominator clamped to ``maxdiv``). Because the gradient flows only through the
    numerator, the model learns to up-weight elements it is confident about while the
    detached denominator keeps the overall loss scale bounded.

    Returns ``(loss, pred, confidence)`` with confidence squashed to (0, 1).
    """
    def __init__(self, lfn, dim=1, eps=1e-1, maxdiv=1e+2):
        super().__init__()
        assert hasattr(lfn, "reduction"), "lfn must expose a `reduction` attribute"
        lfn.reduction = "none"
        self.lfn = lfn
        self.dim = dim
        self.eps = eps
        self.max = maxdiv

    def forward(self, pred, target):
        pred, confidence = pred.chunk(2, dim=self.dim)
        confidence = confidence.sub(confidence.mean()).div(confidence.std() + 1e-5)
        num = confidence.exp().add(self.eps)
        div = num.detach().pow(2)

        loss = self.lfn(pred, target)
        assert loss.shape == num.shape, \
            f"loss shape {tuple(loss.shape)} != confidence shape {tuple(num.shape)}"
        loss = loss * num / div.clamp_max(self.max)
        return loss.mean(), pred, confidence.sigmoid()

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


class CrossNormNd(_nn.Module):
    """Joint cross-input spatial normalization of a pair ``(a, b)`` with a learned
    per-side weighting of the shared "mean field".

    A single global mean/variance (per batch, per channel) is formed from **both**
    ``a`` and ``b`` and used to normalize them together. Each side carries a learned,
    positive scalar-per-channel ``alpha`` (``alpha_a``, ``alpha_b``) that scales how
    much that side's statistics contribute to the global field before they are pooled.
    Concretely the field is the alpha-weighted pool (law of total variance):

        wa, wb = alpha_a * Na, alpha_b * Nb                  # effective weights
        mu     = (wa*mean_a + wb*mean_b) / (wa + wb)
        var    = (wa*(var_a + |mean_a-mu|^2) + wb*(var_b + |mean_b-mu|^2)) / (wa + wb)

    Both ``a`` and ``b`` are then normalized by this shared ``(mu, var)``, and each side
    gets its **own** learned affine (``weight_a/bias_a``, ``weight_b/bias_b``). So the
    statistics are mixed (alpha-weighted), the affine is not. ``alpha`` is parameterized
    as ``softplus(raw)`` (strictly positive) and initialized to ``1`` — at which point,
    since ``wa,wb`` reduce to the element counts, the field is exactly the plain
    concatenated InstanceNorm:

        x = torch.cat((a.flatten(2), b.flatten(2)), -1); xn = (x-x.mean(-1))/sqrt(var+eps)

    ``a`` and ``b`` must agree on batch/channel dims but may differ in spatial shape.
    Call as ``forward(a, b)`` or ``forward((a, b))``; returns ``(a', b')``. Mirrors
    :class:`InstanceNormNd` (complex dtype + optional running stats; in eval with
    ``track_running_stats`` both branches use the shared running stats). Set
    ``learn_alpha=False`` to recover the plain (count-weighted) joint norm with no
    alpha parameters.
    """
    _allowed_ranks = None

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 learn_alpha=True, track_running_stats=False, device=None,
                 dtype=_torch.float):
        super().__init__()
        if dtype is None: dtype = _torch.float
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.learn_alpha = learn_alpha
        self.track_running_stats = track_running_stats
        self.dtype = dtype
        self.is_complex = _is_complex_dtype(dtype)
        var_dtype = _real_dtype_of(dtype) if self.is_complex else dtype

        if affine:
            # Separate affine per branch (a and b), each per-channel.
            self.weight_a = _nn.Parameter(_torch.ones(num_features, dtype=dtype, device=device))
            self.bias_a = _nn.Parameter(_torch.zeros(num_features, dtype=dtype, device=device))
            self.weight_b = _nn.Parameter(_torch.ones(num_features, dtype=dtype, device=device))
            self.bias_b = _nn.Parameter(_torch.zeros(num_features, dtype=dtype, device=device))
        else:
            for n in ("weight_a", "bias_a", "weight_b", "bias_b"):
                self.register_parameter(n, None)

        if learn_alpha:
            # raw params s.t. softplus(raw) == 1 at init -> recovers plain joint norm.
            sp_inv_1 = _math.log(_math.expm1(1.0))
            self.alpha_a_raw = _nn.Parameter(_torch.full((num_features,), sp_inv_1, device=device))
            self.alpha_b_raw = _nn.Parameter(_torch.full((num_features,), sp_inv_1, device=device))
        else:
            self.register_parameter("alpha_a_raw", None)
            self.register_parameter("alpha_b_raw", None)

        if track_running_stats:
            self.register_buffer("running_mean", _torch.zeros(num_features, dtype=dtype, device=device))
            self.register_buffer("running_var", _torch.ones(num_features, dtype=var_dtype, device=device))
            self.register_buffer("num_batches_tracked",
                                 _torch.tensor(0, dtype=_torch.long, device=device))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

    @property
    def alpha_a(self):
        """Resolved positive per-channel weight for side a (``softplus(raw)``)."""
        return _F.softplus(self.alpha_a_raw) if self.learn_alpha else None

    @property
    def alpha_b(self):
        return _F.softplus(self.alpha_b_raw) if self.learn_alpha else None

    def forward(self, a, b=None):
        if b is None:
            assert isinstance(a, (tuple, list)) and len(a) == 2, \
                "CrossNorm expects forward(a, b) or forward((a, b))"
            a, b = a
        assert a.dim() == b.dim(), \
            f"a and b must have the same rank, got {a.dim()} and {b.dim()}"
        if self._allowed_ranks is not None:
            assert a.dim() in self._allowed_ranks, \
                f"expected input rank in {self._allowed_ranks}, got {a.dim()}"
        assert a.shape[0] == b.shape[0] and a.shape[1] == b.shape[1], \
            f"a and b must share batch/channel dims, got {tuple(a.shape[:2])} vs {tuple(b.shape[:2])}"
        assert a.shape[1] == self.num_features, \
            f"expected {self.num_features} channels, got {a.shape[1]}"

        B, C = a.shape[0], a.shape[1]
        sa, sb = a.shape[2:], b.shape[2:]
        na_, nb_ = _math.prod(sa), _math.prod(sb)
        af = a.reshape(B, C, na_)
        bf = b.reshape(B, C, nb_)
        view = (1, -1, 1)
        use_running = (not self.training) and self.track_running_stats

        if use_running:
            mean_v = self.running_mean.view(view)
            var_v = self.running_var.view(view)
        elif self.learn_alpha:
            # Per-side moments, then alpha-weighted pooling into the shared field.
            mean_a, var_a, _ = _moments(af, [2], self.is_complex, keepdim=True)
            mean_b, var_b, _ = _moments(bf, [2], self.is_complex, keepdim=True)
            wa = _F.softplus(self.alpha_a_raw).view(view) * na_
            wb = _F.softplus(self.alpha_b_raw).view(view) * nb_
            wsum = wa + wb
            mean_v = (wa * mean_a + wb * mean_b) / wsum
            da, db = mean_a - mean_v, mean_b - mean_v
            if self.is_complex:
                ssa = da.real.pow(2) + da.imag.pow(2)
                ssb = db.real.pow(2) + db.imag.pow(2)
            else:
                ssa, ssb = da.pow(2), db.pow(2)
            var_v = (wa * (var_a + ssa) + wb * (var_b + ssb)) / wsum
        else:
            # Plain count-weighted joint norm (== concatenated InstanceNorm).
            mean_v, var_v, _ = _moments(_torch.cat((af, bf), dim=2), [2],
                                        self.is_complex, keepdim=True)

        if (not use_running) and self.training and self.track_running_stats:
            with _torch.no_grad():
                m = mean_v.reshape(B, C).mean(0).detach()
                v = var_v.reshape(B, C).mean(0).detach()
                self.running_mean.mul_(1 - self.momentum).add_(m * self.momentum)
                self.running_var.mul_(1 - self.momentum).add_(v * self.momentum)
                self.num_batches_tracked.add_(1)

        inv = (var_v + self.eps).rsqrt()
        an = (af - mean_v) * inv
        bn = (bf - mean_v) * inv
        if self.affine:
            an = an * self.weight_a.view(view) + self.bias_a.view(view)
            bn = bn * self.weight_b.view(view) + self.bias_b.view(view)
        return an.reshape(B, C, *sa), bn.reshape(B, C, *sb)

    def extra_repr(self):
        r = f"{self.num_features}, eps={self.eps}, momentum={self.momentum}"
        r += f", affine={self.affine}, learn_alpha={self.learn_alpha}"
        r += f", track_running_stats={self.track_running_stats}"
        if self.dtype != _torch.float:
            r += f", dtype={self.dtype}"
        return r


class CrossNorm1d(CrossNormNd):
    _allowed_ranks = (3,)

class CrossNorm2d(CrossNormNd):
    _allowed_ranks = (4,)

class CrossNorm3d(CrossNormNd):
    _allowed_ranks = (5,)


class SACrossNormNd(_nn.Module):
    """Attention-pooled cross-stream normalization over a tuple of arbitrary length.

    Generalises :class:`CrossNormNd` from a fixed pair with *static*, learned
    per-channel mixing (``softplus(alpha)``) to ``N`` streams whose mixing weights
    are computed *from the data*, transformer-style. Given streams
    ``(x_0, ..., x_{N-1})`` that share batch/channel dims (spatial shapes may
    differ), each stream is summarised by its per-channel spatial mean (an
    ``adaptive_avg_pool`` to size 1). Stacking the N summaries gives ``P`` of shape
    ``(B, C, N)``; two channel-wise linear maps produce queries and keys::

        q = q_proj(P), k = k_proj(P)               # (B, qk_dim, N)
        attn[b, i, j] = sum_c q[b,c,i] * k[b,c,j]  # (B, N, N), dest i, source j
        alpha = softmax(attn / sqrt(qk_dim), dim=-1)    # over the *source* axis

    ``alpha[b, i, j]`` is the weight with which source ``j`` contributes to
    destination ``i`` (e.g. ``alpha[:, 0, 1]`` weights stream-1's statistics into
    stream-0's field). The "values" of the attention are the per-stream
    means/variances and the softmax is the weighted pool (law of total variance
    for the variance)::

        mu_i  = sum_j alpha[i,j] * mean_j
        var_i = sum_j alpha[i,j] * (var_j + (mean_j - mu_i)^2)

    Each stream ``x_i`` is then normalised by its own ``(mu_i, var_i)`` and gets a
    shared per-channel affine. ``q_proj`` is zero-initialised so ``attn == 0`` at
    init: ``alpha`` is uniform ``1/N`` and the layer starts as a plain
    equal-weighted joint normalisation (``k_proj`` keeps its default init so
    gradients flow into ``q_proj`` from the first step).

    Drop-in for :class:`CrossNormNd`: same ``(num_features, eps, ...)`` constructor
    and the same ``forward(a, b)`` / ``forward((a, b, ...))`` tuple-in/tuple-out
    contract, but accepts any ``N >= 1``. Real dtypes only; ``track_running_stats``
    is unsupported (statistics are always per-batch, the pool being
    input-conditioned). The affine is shared across streams (N-agnostic), unlike
    CrossNorm's per-side affine.
    """
    _allowed_ranks = None

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 qk_dim=None, track_running_stats=False, device=None,
                 dtype=_torch.float):
        super().__init__()
        if dtype is None: dtype = _torch.float
        assert not _is_complex_dtype(dtype), "SACrossNorm supports real dtypes only"
        assert not track_running_stats, "SACrossNorm: track_running_stats not supported"
        self.num_features = num_features
        self.qk_dim = qk_dim or num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = False
        self.dtype = dtype
        self.scale = self.qk_dim ** -0.5
        # LayerNorm the per-stream mean tokens (over the channel axis) BEFORE q/k
        # projection — the standard transformer norm-before-QKV. Without it the dot
        # product scales with ||stream-mean||^2, which the fixed 1/sqrt(qk_dim) does not
        # control: a single high-magnitude batch element (un-normalized, growing NCA
        # state) overflows fp32 inside the einsum -> inf -> nan. Normalizing the tokens
        # makes 1/sqrt(qk_dim) valid so logits stay O(1) regardless of state magnitude.
        self.token_norm = LayerNorm(num_features, dtype=dtype)

        self.q_proj = _nn.Linear(num_features, self.qk_dim, device=device, dtype=dtype)
        self.k_proj = _nn.Linear(num_features, self.qk_dim, device=device, dtype=dtype)
        # Small-normal init on q (weight+bias), like k's default-scale init: small
        # queries -> small logits -> alpha starts NEAR-uniform (≈ plain joint norm) but
        # not exactly, so stream symmetry is broken and the q/token_norm path is live
        # (receives gradient) from step one. Zero-init would make the whole QK path
        # inert at init.
        _nn.init.normal_(self.q_proj.weight, mean=0.0, std=0.01)
        _nn.init.normal_(self.q_proj.bias, mean=0.0, std=0.01)

        if affine:
            self.weight = _nn.Parameter(_torch.ones(num_features, dtype=dtype, device=device))
            self.bias = _nn.Parameter(_torch.zeros(num_features, dtype=dtype, device=device))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, *inputs):
        if len(inputs) == 1 and isinstance(inputs[0], (tuple, list)):
            inputs = tuple(inputs[0])
        assert len(inputs) >= 1, "SACrossNorm expects at least one stream"
        x0 = inputs[0]
        B, C = x0.shape[0], x0.shape[1]
        assert C == self.num_features, f"expected {self.num_features} channels, got {C}"
        for x in inputs:
            assert x.shape[0] == B and x.shape[1] == C, \
                "all streams must share batch/channel dims"
            if self._allowed_ranks is not None:
                assert x.dim() in self._allowed_ranks, \
                    f"expected input rank in {self._allowed_ranks}, got {x.dim()}"

        # per-stream per-channel moments over the spatial dims -> (B, C) each
        means, varis, flats, spatials = [], [], [], []
        for x in inputs:
            xf = x.reshape(B, C, -1)
            means.append(xf.mean(2))
            varis.append(xf.var(2, unbiased=False))
            flats.append(xf)
            spatials.append(x.shape[2:])

        M = _torch.stack(means, dim=-1)          # (B, C, N)  == adaptive_avg_pool(1)
        Vv = _torch.stack(varis, dim=-1)         # (B, C, N)

        # transformer-style attention over the stream axis (channels contracted).
        # LayerNorm the tokens (channel axis, per stream) BEFORE projecting so the dot
        # product logits stay O(1) and 1/sqrt(qk_dim) is valid regardless of ||M||.
        Mt = self.token_norm(M.transpose(1, 2))              # (B, N, C) normed over C
        q = self.q_proj(Mt).transpose(1, 2)                  # (B, qk, N)
        k = self.k_proj(Mt).transpose(1, 2)                  # (B, qk, N)
        attn = _torch.einsum("bci,bcj->bij", q, k) * self.scale   # (B, N, N) [dest, src]
        alpha = attn.softmax(dim=-1)                          # weighted pool over source j

        mu = _torch.einsum("bij,bcj->bci", alpha, M)         # (B, C, N) per-dest mean
        diff = M.unsqueeze(-1) - mu.unsqueeze(-2)            # (B, C, j, i)
        var = (_torch.einsum("bij,bcj->bci", alpha, Vv)
               + _torch.einsum("bij,bcji->bci", alpha, diff.pow(2)))   # (B, C, N)

        out = []
        for i, xf in enumerate(flats):
            mi = mu[..., i].unsqueeze(-1)
            inv = (var[..., i].unsqueeze(-1) + self.eps).rsqrt()
            xn = (xf - mi) * inv
            if self.affine:
                xn = xn * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)
            out.append(xn.reshape(B, C, *spatials[i]))
        return tuple(out)

    def extra_repr(self):
        r = f"{self.num_features}, qk_dim={self.qk_dim}, eps={self.eps}, affine={self.affine}"
        if self.dtype != _torch.float:
            r += f", dtype={self.dtype}"
        return r


class SACrossNorm1d(SACrossNormNd):
    _allowed_ranks = (3,)

class SACrossNorm2d(SACrossNormNd):
    _allowed_ranks = (4,)

class SACrossNorm3d(SACrossNormNd):
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


class BlendedRunningGroupNorm2d(_nn.Module):
    """Per-channel norm that blends live spatial stats with slowly-evolving running stats.

    Used identically in train and eval modes — no self.training branching.

    For each forward call, computes per-instance per-channel spatial mean/std
    (live stats), then blends them with a slow exponential moving average:

        mu_blend  = alpha * mu_live  + (1 - alpha) * running_mu
        sig_blend = alpha * sig_live + (1 - alpha) * running_sigma
        y = (x - mu_blend) / (sig_blend + eps)

    Running buffers are updated every forward pass (train and eval alike) with
    the batch-averaged live stats:

        running_mu  <- (1 - beta) * running_mu  + beta * mean_B(mu_live)
        running_var <- (1 - beta) * running_var + beta * mean_B(var_live)

    Limit cases:
    - blend_alpha=1  →  standard InstanceNorm (no running stats used)
    - blend_alpha=0, decay_beta→0  →  fully slow multiplier (Berlin–Kac-like)
    - blend_alpha in (0,1)  →  partial slowing of the spatial constraint

    DSL token: ``K[alpha,beta]``  (e.g. ``K[0.5,0.05]``).
    If brackets are omitted, defaults are alpha=0.5, beta=0.05.
    """

    def __init__(self, channels: int, blend_alpha: float = 0.5,
                 decay_beta: float = 0.05, eps: float = 1e-5,
                 affine: bool = True, dtype=None):
        super().__init__()
        if dtype is None:
            dtype = _torch.float
        self.channels = channels
        self.alpha = float(blend_alpha)
        self.beta = float(decay_beta)
        self.eps = eps
        self.dtype = dtype
        self.register_buffer("running_mu",  _torch.zeros(channels, dtype=dtype))
        self.register_buffer("running_var", _torch.ones(channels,  dtype=dtype))
        self.register_buffer("initialized", _torch.tensor(False))
        if affine:
            self.weight = _nn.Parameter(_torch.ones(channels,  dtype=dtype))
            self.bias   = _nn.Parameter(_torch.zeros(channels, dtype=dtype))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias",   None)

    def forward(self, x: _torch.Tensor) -> _torch.Tensor:  # x: (B, C, H, W)
        # Per-instance per-channel live spatial stats
        mu_live  = x.mean(dim=(2, 3))                     # (B, C)
        var_live = x.var(dim=(2, 3), unbiased=False)       # (B, C)
        sig_live = var_live.sqrt()

        # Batch-averaged live stats for the running-buffer update (detached)
        mu_batch  = mu_live.detach().mean(0)               # (C,)
        var_batch = var_live.detach().mean(0)              # (C,)

        if not self.initialized:
            self.running_mu.copy_(mu_batch)
            self.running_var.copy_(var_batch)
            self.initialized.fill_(True)

        # Blended normalisation statistics — per-instance live + broadcast running
        run_sig = self.running_var.sqrt()
        mu_blend  = self.alpha * mu_live  + (1.0 - self.alpha) * self.running_mu[None, :]   # (B,C)
        sig_blend = self.alpha * sig_live + (1.0 - self.alpha) * run_sig[None, :]            # (B,C)

        y = (x - mu_blend[:, :, None, None]) / (sig_blend[:, :, None, None] + self.eps)

        if self.weight is not None:
            y = y * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        # Update running buffers AFTER computing y so the next step sees a
        # marginally updated stat. No self.training gate: same in train and eval.
        with _torch.no_grad():
            self.running_mu.mul_(1.0 - self.beta).add_(self.beta * mu_batch)
            self.running_var.mul_(1.0 - self.beta).add_(self.beta * var_batch)

        return y

    def extra_repr(self) -> str:
        return (f"{self.channels}, alpha={self.alpha}, beta={self.beta}, "
                f"eps={self.eps}")


class DampenedGroupNorm2d(_nn.Module):
    """Per-channel spatial GroupNorm (num_groups=C) with dampened mean/std subtraction.

    Standard InstanceNorm projects out the spatial zero-mode every step, which
    enforces the Berlin-Kac spherical constraint.  This module lets you dial that
    projection continuously:

        y = (x - gamma_mean * mu) / (sigma^gamma_std + eps)

    where mu, sigma are the standard per-instance per-channel spatial mean / std.

    gamma_mean = 1, gamma_std = 1  →  standard IN (spherical regime, eta ~ 0.04)
    gamma_mean = 0, gamma_std = 1  →  no mean subtraction (ferromagnetic accessible)
    gamma_mean = 1, gamma_std = 0  →  no scale normalization (likely unstable)

    Affine weight/bias (learnable) are applied after dampening, matching the
    N token in the body DSL.
    """

    def __init__(self, num_channels: int, gamma_mean: float = 1.0,
                 gamma_std: float = 1.0, eps: float = 1e-5,
                 affine: bool = True, device=None, dtype=_torch.float):
        super().__init__()
        if dtype is None:
            dtype = _torch.float
        self.num_channels = num_channels
        self.gamma_mean = float(gamma_mean)
        self.gamma_std = float(gamma_std)
        self.eps = eps
        self.affine = affine
        self.dtype = dtype

        if affine:
            self.weight = _nn.Parameter(_torch.ones(num_channels, dtype=dtype, device=device))
            self.bias = _nn.Parameter(_torch.zeros(num_channels, dtype=dtype, device=device))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: _torch.Tensor) -> _torch.Tensor:  # x: (B, C, H, W)
        assert x.shape[1] == self.num_channels, (
            f"DampenedGroupNorm2d: expected {self.num_channels} channels, got {x.shape[1]}")
        mu = x.mean(dim=(2, 3), keepdim=True)          # per-instance per-channel spatial mean
        var = x.var(dim=(2, 3), unbiased=False, keepdim=True)
        sigma = (var + self.eps).sqrt()
        # Dampened denominator: sigma^gamma_std + eps.
        # When gamma_std=1 → sigma + eps (standard).
        # When gamma_std=0 → 1 + eps ≈ 1 (no scale norm).
        denom = sigma.pow(self.gamma_std) + self.eps
        out = (x - self.gamma_mean * mu) / denom
        if self.affine:
            view = [1, self.num_channels] + [1] * (x.dim() - 2)
            out = out * self.weight.view(view) + self.bias.view(view)
        return out

    def extra_repr(self) -> str:
        return (f"{self.num_channels}, gamma_mean={self.gamma_mean}, "
                f"gamma_std={self.gamma_std}, eps={self.eps}, affine={self.affine}")


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




class ScaledTanh(_nn.Module):
    """Saturating cap k*tanh(x/k); k=1 is a plain tanh, larger k saturates later."""

    def __init__(self, k=1.0):
        super().__init__()
        self.k = float(k)

    def forward(self, x):
        return self.k * _torch.tanh(x / self.k)

    def extra_repr(self):
        return f"k={self.k}"


class EnergyConserving(_nn.Module):
    """Runs the wrapped ops, then rescales the output to the energy of its own input (mean over channels,
    summed over space). For complex states the energy is |x|^2."""

    def __init__(self, *ops):
        super().__init__()
        self.inner = ops[0] if len(ops) == 1 else Sequential(*ops)

    def forward(self, x):
        dims = tuple(range(2, x.dim()))
        e_in = x.abs().pow(2).mean(1, keepdim=True).sum(dims, keepdim=True).clamp(min=1e-12)
        y = self.inner(x)
        e_y = y.abs().pow(2).mean(1, keepdim=True).sum(dims, keepdim=True).clamp(min=1e-12)
        return y * (e_in / e_y).sqrt()


class Cubic(_nn.Module):
    """Gross-Pitaevskii interaction term |psi|^2 psi; equivariant under a global phase, so U(1) survives it."""

    def forward(self, x):
        return x * x.abs().pow(2)


class ChannelPool(_nn.Module):
    """Adaptive average pool along the channel axis (dim=1).

    Reduces ``(B, C_in, *spatial)`` to ``(B, out_channels, *spatial)`` by
    treating the channel axis as a 1D length and applying
    ``F.adaptive_avg_pool1d``.
    """

    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = int(out_channels)

    def forward(self, x):
        if x.dim() < 2:
            raise ValueError(f"ChannelPool needs ≥2 dims, got {x.dim()}")
        # Move channel to the end → flatten everything else as the batch →
        # add a singleton "channel" so adaptive_avg_pool1d treats the original
        # channel axis as the length-to-pool.
        c_in = x.size(1)
        y = x.movedim(1, -1)                       # (B, ..., C_in)
        spatial = y.shape[:-1]
        flat = y.reshape(-1, 1, c_in)              # (N, 1, C_in)
        flat = _F.adaptive_avg_pool1d(flat, self.out_channels)  # (N, 1, C_out)
        flat = flat.squeeze(1)                     # (N, C_out)
        y = flat.view(*spatial, self.out_channels)
        return y.movedim(-1, 1)

    def extra_repr(self):
        return f"out_channels={self.out_channels}"


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

class Parse3d(_nn.Sequential):
    def __init__(self, string, hidden_dim=None, kernel_size=3):
        self.string = string
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        super().__init__(*_parse_block(string, hidden_dim, dim=3, kernel_size=kernel_size))

    def extra_repr(self):
        if self.hidden_dim is not None:
            return f"string=\"{self.string}\", hidden_dim={self.hidden_dim}"
        else:
            return f"string=\"{self.string}\""

def _parse_block(config, hidden_dim=None, dim=2, kernel_size=3):
    ConvNd = getattr(_nn, f"Conv{dim}d")
    InstanceNormCls = globals()[f"InstanceNorm{dim}d"]
    BatchNormCls = globals()[f"BatchNorm{dim}d"]
    CrossNormCls = globals()[f"CrossNorm{dim}d"]
    SACrossNormCls = globals()[f"SACrossNorm{dim}d"]
    ks = kernel_size
    pad = (ks - 1) // 2
    s = [[]]
    par = []        # stack of parallel blocks; each is a list of finalized branch modules
    par_enter = []  # stack of (i, o) channel counts at each '<' so branches start aligned
    par_chan = []   # stack of per-branch (i, o) end-channel counts (consumed by '@n')
    last_stream_chans = None  # per-stream (i,o) of the current tuple value, for a later '@n'
    i = o = hidden_dim

    def _finalize_branch(seq):
        if not seq: return _nn.Identity()
        return seq[0] if len(seq) == 1 else Sequential(*seq)
    dtype = _torch.float
    # Dampened-IN default params (used when 'D' token has no brackets)
    _d_gamma_mean = 1.0
    _d_gamma_std = 1.0
    # Tracking index for bracket skipping.  Python for-loops don't allow
    # mutating idx to skip characters, so we track a skip counter instead.
    _skip_until = -1
    for idx in range(len(config)):
        if idx <= _skip_until:
            continue
        c = config[idx]
        d = config[idx+1] if idx < len(config)-1 else None

        # --- D token: DampenedGroupNorm2d[gm,gs] ---
        # Parse float bracket args before the shared bracket handler below,
        # because the shared handler always tries int conversion.
        if c == "D":
            gm, gs = 1.0, 1.0
            if d == "[":
                e = config[idx+1:].index("]")
                v = config[idx+2:idx+1+e]
                parts = v.split(",")
                gm = float(parts[0]) if len(parts) >= 1 else 1.0
                gs = float(parts[1]) if len(parts) >= 2 else 1.0
                _skip_until = idx + 1 + e  # skip '[', content, ']'
            s[-1].append(DampenedGroupNorm2d(i, gamma_mean=gm, gamma_std=gs,
                                             affine=True, dtype=dtype))
            continue

        # --- K token: BlendedRunningGroupNorm2d[alpha,beta] ---
        # Same float-bracket parsing as D. Defaults: alpha=0.5, beta=0.05.
        if c == "K":
            ka, kb = 0.5, 0.05
            if d == "[":
                e = config[idx+1:].index("]")
                v = config[idx+2:idx+1+e]
                parts = v.split(",")
                ka = float(parts[0]) if len(parts) >= 1 else 0.5
                kb = float(parts[1]) if len(parts) >= 2 else 0.05
                _skip_until = idx + 1 + e
            s[-1].append(BlendedRunningGroupNorm2d(i, blend_alpha=ka,
                                                   decay_beta=kb,
                                                   affine=True, dtype=dtype))
            continue

        # --- h / s tokens: ExpHardCap[cap] / ExpSoftCap[cap], default cap 6 ---
        if c in {"h", "s"}:
            cap, tau = 6.0, 1.0
            if d == "[":
                e = config[idx+1:].index("]")
                parts = config[idx+2:idx+1+e].split(",")
                cap = float(parts[0])
                if len(parts) >= 2: tau = float(parts[1])
                _skip_until = idx + 1 + e
            s[-1].append(ExpHardCap(cap) if c == "h" else ExpSoftCap(cap, tau))
            continue

        if c in { "C", "c", "L", "l" , "N", "n", "P"}:
            if d == "[":
                e = config[idx+1:].index("]")
                v = config[idx+2:idx+1+e]
                if "x" in v:
                    o = i * int(v[1:])
                elif "/" in v:
                    o = i // int(v[1:])
                elif "," in v:
                    i,o = [int(vv) for vv in v.split(",")]
                else:
                    o = int(v)
                idx = idx + 1 + e

        if idx == 0 and c == "[":
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
        elif c == "Q":
            s[-1].append(ChannelNorm(i, elementwise_affine=True)),
        elif c == "q":
            s[-1].append(ChannelNorm(i, elementwise_affine=False)),
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
        elif c == "P":
            s[-1].append(ChannelPool(o))
            i = o
        elif c == "U":
            s[-1].append(_nn.GLU(dim=1))
            o = i//2
            i = o
        elif c == "E":
            assert dtype not in {_torch.cfloat, _torch.cdouble}
            s[-1].append(_nn.ELU())
        elif c == "e":
            s[-1].append(Exp())
        elif c == "R":
            assert dtype not in {_torch.cfloat, _torch.cdouble}
            s[-1].append(_nn.ReLU())
        elif c == "G":
            assert dtype not in {_torch.cfloat, _torch.cdouble}
            s[-1].append(_nn.GELU())
        elif c == "S":
            s[-1].append(_nn.Sigmoid())
        elif c == "T":
            if d == "[":
                e = config[idx+1:].index("]")
                v = config[idx+2:idx+1+e]
                s[-1].append(ScaledTanh(float(v[1:]) if v.startswith("x") else float(v)))
                _skip_until = idx + 1 + e
            else:
                s[-1].append(_nn.Tanh())
        elif c == "=":
            s[-1] = [EnergyConserving(*s[-1])] if s[-1] else []
        elif c == "V":
            s[-1].append(Cubic())
        elif c == "(":
            s.append([])
        elif c == ")":
            r = s.pop(-1)
            r = Skip(*r)
            s[-1].append(r)
        elif c == "<":
            # parallel start: open a new branch block + a fresh sequence for branch 0
            par.append([])
            par_enter.append((i, o))
            par_chan.append([])
            s.append([])
        elif c == "|":
            # next branch: finalize current branch, record its end channels, reset to entry
            par[-1].append(_finalize_branch(s.pop(-1)))
            par_chan[-1].append((i, o))
            i, o = par_enter[-1]
            s.append([])
        elif c == ">":
            # parallel end. One path (no '|') -> ForEach: the shared sequence is mapped
            # over every input stream. Multiple paths -> Parallel: distinct weights per
            # stream. (For a single tensor input both yield a 1-tuple; they differ when
            # fed a multi-stream tuple, e.g. after a 'Y'.)
            par[-1].append(_finalize_branch(s.pop(-1)))
            par_chan[-1].append((i, o))
            branches = par.pop(-1)
            chans = par_chan.pop(-1)
            par_enter.pop(-1)
            s[-1].append(ForEach(*branches) if len(branches) == 1 else Parallel(*branches))
            # channels left at a branch's end (all equal under ForEach). Remember the
            # per-stream channels so a later '@n' can restore the selected stream's count.
            last_stream_chans = chans
            i, o = chans[-1]
        elif c == "@":
            # '@n' selects stream n from a tuple/list value (standalone Index). Can appear
            # anywhere a tuple is present: after <...>, after a Y, or a tuple-in Skip.
            j = idx + 1
            if j < len(config) and config[j] == "-": j += 1
            while j < len(config) and config[j].isdigit(): j += 1
            sel = int(config[idx+1:j])
            s[-1].append(Index(sel))
            if last_stream_chans is not None and -len(last_stream_chans) <= sel < len(last_stream_chans):
                i, o = last_stream_chans[sel]
                last_stream_chans = None
            _skip_until = j - 1
        elif c == "Y":
            s[-1].append(CrossNormCls(i))
        elif c == "X":
            s[-1].append(SACrossNormCls(i))
    assert len(s) == 1
    return s[-1]
    #if len(s[-1]) == 1: return s[-1][0]
    #else: return _nn.Sequential(*s[-1])
            
