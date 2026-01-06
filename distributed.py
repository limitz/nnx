import math as _math
import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _F
import torch.distributed as _dist
import torchvision as _torchvision
import numpy as _np
import collections as _collections
from . import functional as _Fx
from torch.distributed import ReduceOp


class all_gather_with_grad(_torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            _dist.is_available()
            and _dist.is_initialized()
            and (_dist.get_world_size() > 1)
        ):
            outputs = [_torch.zeros_like(x) for _ in range(_dist.get_world_size())]
            _dist.all_gather(outputs, x)
            return _torch.cat(outputs, 0)
        return x

    @staticmethod
    def backward(ctx, grads):
        if (
            _dist.is_available()
            and _dist.is_initialized()
            and (_dist.get_world_size() > 1)
        ):
            s = (grads.shape[0] // _dist.get_world_size()) * _dist.get_rank()
            e = (grads.shape[0] // _dist.get_world_size()) * (_dist.get_rank() + 1)
            grads = grads.contiguous()
            _dist.all_reduce(grads)
            return grads[s:e]
        return grads


class all_reduce_with_grad(_torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, op=ReduceOp.SUM):
        if (
            _dist.is_available()
            and _dist.is_initialized()
            and (_dist.get_world_size() > 1)
        ):
            x = x.contiguous()
            _dist.all_reduce(x, op)
            ctx["op"] = op
        return x

    @staticmethod
    def backward(ctx, grads):
        if "op" in ctx:
            if ctx["op"] in { ReduceOp.SUM, ReduceOp.AVG }:
                return grads
            else:
                assert False, "implementation needs to be checked"
        else:
            return grads

class AllGather(_nn.Module):
    def forward(self, x):
        return all_gather_with_grad(x)

class AllReduce(_nn.Module):
    def __init__(self, op=ReduceOp.SUM):
        super().__init__()
        self.op = op

    def forward(self, x):
        return all_reduce_with_grad(x, self.op)

class SyncGroupNorm(_nn.Module):
    def __init__(self, groups, channels=None, affine=True, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.channels = channels
        self.groups = groups
        self.affine = affine
        if channels:
            assert channels % groups == 0
        if affine:
            assert channels is not None
            self.weight = _nn.parameter.Parameter(_torch.randn(channels))
            self.bias = _nn.parameter.Parameter(_torch.randn(channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
            
    def forward(self, x):
        groups = x.view(x.shape[0], self.groups, -1)
        if (
            _dist.is_available() 
            and _dist.is_enabled() 
            and (_dist.world_size() > 1)
        ):
            x1 = groups.mean(-1)
            x2 = groups.pow(2).mean(-1)
            xs = _torch.stack([x1,x2])
            all_reduce_with_grad(xs, ReduceOp.AVG)
            mean, x2 = xs.unbind(0)
            var = x2 - mean.pow(2)
            std = var.sqrt()
            std = _torch.where(_torch.isnan(std),
                               _torch.zeros_like(std)
                               std)
        else:
            std, mean = torch.std_mean(groups,-1,correction=0,
                                       keepdim=True)
            
        groups = groups.sub(mean).div(std.add(self.eps))
        x = groups.view(x.shape)
        if self.affine:
            w = self.weight.view(self.channels, [1] * (x.dim()-2))
            b = self.bias.view(self.channels, [1]*(x.dim()-2))
            x = x * w + b
        return x
