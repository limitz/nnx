import torch as _torch
import torch.nn.functional as _F
import torch.nn.functional as legacy
from torch.nn.functional import *
import cv2 as _cv2
import math as _math


def expand(t, shape):
    shape = [1 if t.dim() <= i and s == -1 else s for i,s in enumerate(reversed(shape))]
    return t.expand(list(reversed(shape)))
        
def tensor(x, shape=None, device=None, dtype=None):
    if not isinstance(x, _torch.Tensor): 
        if isinstance(x, (list, tuple, _torch.Size)):
            if any(isinstance(v,_torch.Tensor) for v in x):
                for t in x:
                    if isinstance(t, _torch.Tensor):
                        device = device or t.device
                        dtype = dtype or t.dtype
                        break
                assert shape is not None
                x = [expand(v,shape).to(device).to(dtype) 
                     if isinstance(v, _torch.Tensor) 
                     else expand(_torch.tensor(v, dtype=dtype, device=device), shape)
                     for v in x]
                x = _torch.cat(x, shape.index(-1))
            else:    
                x = _torch.tensor(x, device=device)
        else:
            if shape is None: shape = ()
            elif isinstance(shape, int): shape = (shape,)
            x = _torch.full(tuple(shape), x, dtype=dtype, device=device)
    
    if device is not None:
        x = x.to(device)
    if dtype is not None:
        x = x.to(dtype)
    return x

def crop_view(input, box):
    x,y,w,h = box
    if isinstance(x,float): x = int(x * input.shape[-1])
    if isinstance(y,float): y = int(y * input.shape[-2])
    if isinstance(w,float): w = int(w * input.shape[-1])
    if isinstance(h,float): h = int(h * input.shape[-2])
    x0 = min(max(0,x), input.shape[-1])
    y0 = min(max(0,y), input.shape[-2])
    x1 = min(max(0,x+w), input.shape[-1])
    y1 = min(max(0,y+h), input.shape[-2])
    return input[...,y0:y1,x0:x1]
    
def pad_to(input, size, mode="center", *args, **kwargs):
    if isinstance(input, (list, tuple)):
        return _torch.stack([pad_to(v, size, mode, *args, **kwargs) for v in input])
        
    padding = [(a-b) for a,b in zip(reversed(size), reversed(input.shape))]
    padding = tensor([padding,[0]*len(size)])
    
    if isinstance(mode, str):
        mode = [mode] * len(size)
    for dim,(p,m) in enumerate(zip(padding.split(1,-1),mode)):
        if m == "near":
            p[:] = p.flip(-2)
        elif m == "center":
            p[:] -= p[0]//2
            p[1,:] *= -1
        if p[0] < 0:
            input = input.transpose(-dim-1,-1)
            input = input[...,-p[0]:]
            input = input.transpose(-dim-1,-1)
            p[0] = 0    
        if p[1] < 0:
            input = input.transpose(-dim-1,-1)
            input = input[...,:p[1]]
            input = input.transpose(-dim-1,-1)
            p[1] = 0
    return _F.pad(input, padding.mT.reshape(-1).unbind(), *args, **kwargs)
        

def padstack(tensors, dim=0, pad_mode="constant", pad_value=0):
    assert len(tensors) > 0
    ndims = tensors[0].dim()
    if dim < 0: dim += ndims
    for t in tensors[1:]: assert t.dim() ==  ndims
    new_shape = [max([t.shape[d] for t in tensors]) for d in range(ndims)]
    padding = [[new_shape[d//2] - t.shape[d//2] if (d & 1) else 0 for d in range(ndims*2,0,-1)] for t in tensors]
    return _torch.stack([_F.pad(t,p,pad_mode,pad_value) for t,p in zip(tensors,padding)], dim=dim)

def padcat(tensors, dim=0, pad_mode="constant", pad_value=0):
    assert len(tensors) > 0
    ndims = tensors[0].dim()
    if dim < 0: dim += ndims
    for t in tensors[1:]: assert t.dim() ==  ndims
    new_shape = [max([t.shape[d] for t in tensors]) if d != dim else -1 for d in range(ndims)]
    padding = [[new_shape[d//2] - t.shape[d//2] if (d & 1 and d//2 != dim) else 0 for d in range(ndims*2,0,-1)] for t in tensors]
    return _torch.cat([_F.pad(t,p,pad_mode,pad_value) for t,p in zip(tensors,padding)], dim=dim)

def broadcast_shape(input, ignore_dim=None):
    assert isinstance(input, (list, tuple))
    assert len(input) > 0
    s = _torch.tensor(input[0].shape)
    if ignore_dim is not None:
        s[ignore_dim] = -1
    for v in input[1:]:
        vs = _torch.tensor(v.shape)
        if vs.numel() < s.numel(): 
            vs = _F.pad(vs, s.numel()-vs.numel(), "constant", 1)
        elif s.numel() < vs.numel():
            s = _F.pad(s, vs.numel()-s.numel(), "constant", 1)
        if ignore_dim is not None:
            vs[ignore_dim] = -1
        vs[vs==1] = s[vs==1]
        s[s==1] = vs[s==1]
        if vs != s: return None
    return s

def interpolated_cat(input, size=None, dim=0):
    assert isinstance(input, (list, tuple))
    assert len(input) > 0
    size = size or input[0].shape[dim]
    ...

def interpolate(input, size=None, scale_factor=None, mode="nearest"):
    if -1 in size:
        assert isinstance(input, (list, tuple))
        r = []
        for x in input:
            scales = [v/w for v,w in zip(size, x.shape[-len(size):]) 
                      if v != -1]
            if len(scales) > 1:
                scale = _math.exp(sum(_math.log(scale) 
                                      for scale in scales) / len(scales))
            else:
                scale = scales[0]
            s = [int(w*scale) if v == -1 else v 
                    for v,w in zip(size, x.shape[-len(size):])]
            y = _F.interpolate(x[None], size=s, mode=mode)[0]
            r.append(y)
        return r
        
    if mode == "tricubic":
        s = input.shape
        x = input.flatten(0,-4)
        for i in range(3):
            x = interpolate(x, (x.shape[-2],size[-i]), mode="bicubic")
            x = x.permute(0,2,3,1).contiguous()
        return xs.view(*s[:-3], *size)
    elif input.dtype in { _torch.cfloat, _torch.cdouble }:
        return _torch.complex(
            interpolate(input.real, size, scale_factor, mode),
            interpolate(input.imag, size, scale_factor, mode))
    
    else:
        r = _F.interpolate(input, size=size, 
                           scale_factor=scale_factor, 
                           mode=mode)
        return r
        
def _extrapolate2d(r):
    while r[:,-1].lt(1).any():
        mask = r[:,-1:].lt(1).bool()
        rm = _torch.where(mask, _torch.full_like(r, -_math.inf), r)
        q = _torch.where(mask, _F.max_pool2d(rm, 3, stride=1, padding=1), r)
        dx = _torch.gradient(q[:,0], dim=-1)[0]
        dy = _torch.gradient(q[:,1], dim=-2)[0]
        delta = _torch.stack((dx,dy),1)/8
        f = r[:,:-1]
        f = _torch.where(mask, _torch.zeros_like(f), f)
        delta = _torch.where(mask, _torch.zeros_like(delta), delta)
        k = _torch.tensor([[[1,0,-1],[1,0,-1],[1,0,-1]],
                          [[1,1,1],[0,0,0],[-1,-1,-1.]]],
                         device = f.device).view(2,1,3,3)
        kd = _F.conv2d(delta, k, padding=1, groups=2)
        k = _torch.ones(1,1,3,3, device=f.device)
        ks = _F.conv2d(f, k.expand(2,1,3,3), padding=1, groups=2)
        km = _F.conv2d((~mask).float(), k, padding=1)
        f = (ks+kd*0.9) / km
        dst = km.gt(0) * (mask.float())
        f = _torch.cat((f, dst.float()),1)
        r = _torch.where(dst.gt(0), f, r)
    return r

def _extrapolate3d(r):
    while r[:,-1].lt(1).any():
        mask = r[:,-1:].lt(1).bool()
        rm = _torch.where(mask, _torch.full_like(r, -_math.inf), r)
        q = _torch.where(mask, _F.max_pool3d(rm, 3, stride=1, padding=1), r)
        dx = _torch.gradient(q[:,0], dim=-1)[0]
        dy = _torch.gradient(q[:,1], dim=-2)[0]
        dz = _torch.gradient(q[:,2], dim=-3)[0]
        delta = _torch.stack((dx,dy,dz),1) / 8
        f = r[:,:-1]
        f = _torch.where(mask, _torch.zeros_like(f), f)
        delta = _torch.where(mask, _torch.zeros_like(delta), delta)
        k = _torch.tensor([[1,0,-1,1,0,-1,1,0,-1.]*3,
                          [1,1,1,0,0,0,-1,-1,-1.]*3,
                          [1,1,1]*3+[0,0,0]*3+[-1,-1,-1]*3],
                         device = f.device).view(3,1,3,3,3)
        kd = _F.conv3d(delta, k, padding=1, groups=3)
        k = _torch.ones(1,1,3,3,3, device=f.device)
        ks = _F.conv3d(f, k.expand(3,1,3,3,3), padding=1, groups=3)
        km = _F.conv3d((~mask).float(), k, padding=1)
        f = (ks+kd*0.9) / km
        dst = km.gt(0) * (mask.float())
        f = _torch.cat((f, dst.float()),1)
        r = _torch.where(dst.gt(0), f, r)
    return r

def grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=None):
    if padding_mode == "extrapolate":
        x = _torch.cat([input, _torch.ones_like(input[:,[0]])],1)
        x = grid_sample(x, grid, mode, "zeros", align_corners)
        assert x.dim() in {4,5}
        
        if x.dim() == 4: x = _extrapolate2d(x)
        elif x.dim() == 5: x = _extrapolate3d(x)
        return x[:,:-1]
    elif input.dtype in { _torch.cfloat, _torch.cdouble }:
        return _torch.complex(
            _F.grid_sample(input.real,grid,mode,padding_mode,align_corners),
            _F.grid_sample(input.imag,grid,mode,padding_mode,align_corners)
        )
    else:
        return _F.grid_sample(input,grid,mode,padding_mode,align_corners)

def flow_to_grid(flow):
    dims = [v for v in range(flow.dim()) if v != 1] + [1]
    return flow.permute(dims)

def grid_to_flow(grid):
    dims = [0,-1] + list(range(1,grid.dim()-1))
    return grid.permute(dims)
    
def flow_sample(input, flow, *args, **kwargs):
    return grid_sample(input,flow_to_grid(flow), **args, **kwargs)   

def linfield(*size, device="cpu"):
    assert len(size) > 0
    if isinstance(size[0], (tuple, list)): 
        assert len(size) == 1
        size = size[0]
    return _torch.stack(
        _torch.meshgrid(*[_torch.linspace(-1,1,s,device=device) * s / (s+1)
                         for s in reversed(size)], indexing="xy"), -1)

def norm(x, dim=None, eps=1e-8):
    return x.sub(x.mean(dim,keepdim=True)).div(x.std(dim,keepdim=True) + eps)

def center(x, dim=None):
    return x.sub(x.mean(dim,keepdim=True))

def n_between(a,b,n=(),inclusive=True):
    if isinstance(a, float) or isinstance(b, float):
        r = _torch.rand(n) * (b-a) + a
    else:
        b = (b+1) if inclusive else b
        n = n if isinstance(n, (tuple, list)) else (n,)
        r = _torch.randint(a,b,n)
    return r.item() if r.dim() == 0 else r
    
between = n_between

def constant_or_between(v,a,b,n): 
    return n_between(a,b,n) if v is None else v

def random_clipping_mask_3d(shape, strength=.25, device="cpu"):
    s = strength
    mask = _torch.ones(1,1,*shape[-3:],device=device)
    rotate = (_torch.randn(3) * _math.pi) * s
    #vect = 1 #rotate.sin().abs() + rotate.cos().abs()
    translate = _torch.randn(3) * s / 3
    scale = _torch.randn(3).mul(s).exp()
    affines = _torch.tensor([
        [
            [_math.cos(rotate[0]), _math.sin(rotate[0]), 0, 0],
            [-_math.sin(rotate[0]), _math.cos(rotate[0]), 0, 0],
            [0,0,1,0], 
            [0,0,0,1]
        ],
        [
            [_math.cos(rotate[1]), 0, _math.sin(rotate[1]), 0],
            [0,1,0,0], 
            [-_math.sin(rotate[1]), 0, _math.cos(rotate[1]),  0],
            [0,0,0,1]
        ],
        [
            [1,0,0,0], 
            [0,_math.cos(rotate[2]), _math.sin(rotate[2]),  0],
            [0,-_math.sin(rotate[2]), _math.cos(rotate[2]), 0],
            [0,0,0,1]
        ],
        [
            [scale[0],0,0, translate[0]],
            [0,scale[1],0, translate[1]],
            [0,0,scale[2], translate[2]],
            [0,0,0,1]
        ],
    ], device=device)
    a = affines[0] @ affines[1] @ affines[2] @ affines[3]
    field = linfield(*shape[-3:], device=device)
    field = _F.pad(field, (0,1), "constant", 1) @ a.mT
    field = field[...,:3]
    mask = _F.grid_sample(mask, field[None], mode="bilinear", align_corners=True)[0]
    return mask

def _window(*size, mode="cos", device="cpu"):
    if "cos" in mode:
        window = _torch.stack(_torch.meshgrid(*[_torch.linspace(-_math.pi/2+1e-2,_math.pi/2-1e-2,s, device=device).cos().pow(0.2) for s in size]))
    else:
        window = _torch.ones(1,*size)    
    if "prod" in mode: window = window.prod(0,keepdim=True)
    return window
    
def random_flow_field_3d(nps=((3,5),(4,6),(9,14)), 
                         strengths=(1e-1, 4e-2, 1e-2), 
                         size=(32,32,32), no_field=False, 
                         iterations=2, device="cpu", affine=0, 
                         window_mode="cos+prod+global"):
    if not isinstance(nps, (list, tuple)): nps = [nps]
    if not isinstance(strengths, (list, tuple)): strengths = [strengths]
    assert len(nps) == len(strengths)
    field = linfield(size, device=device).permute(3,0,1,2)
    w1 = _window(*size, mode=window_mode, device=device)
    w2 = _window(*size, mode=window_mode.replace("prod",""), device=device)
    
    for i in range(iterations):
        subfield = linfield(size, device=device).permute(3,0,1,2)
        for np, strength in zip(nps, strengths):
            if isinstance(np, tuple):
                np = _torch.randint(np[1]-np[0], (3,))+np[0]
                np = [n.item() for n in np]
            else:
                np = (np,np,np)
            ptss = _torch.randn(3, np[0],np[1],np[2], device=device)*strength
            for j in range(3):
                fs = []
                for pts in ptss.split(1,1):
                    f = _F.interpolate(pts, (pts.shape[-2],size[-j]), mode="bicubic")
                    f = _torch.nan_to_num(f)
                    fs.append(f[:,0])
                ptss = _torch.stack(fs,-1)
            subfield += ptss * w1
        #subfield = subfield 
        field = _F.grid_sample(field[None], subfield.permute(1,2,3,0)[None], 
                                   align_corners=True, mode="bilinear",
                                   padding_mode="border")[0]
        
        if affine > 0:
            field = field.permute(1,2,3,0)
            s = affine #strengths[0] if "global" in window_mode else 0.2
            rotate = (_torch.randn(3) * _math.pi) * s
            vect = rotate.sin().abs() + rotate.cos().abs()
            translate = _torch.randn(3) * s
            scale = _torch.ones(3)#.mul(vect)
            affines = _torch.tensor([
                [
                    [_math.cos(rotate[0]), _math.sin(rotate[0]), 0, 0],
                    [-_math.sin(rotate[0]), _math.cos(rotate[0]), 0, 0],
                    [0,0,1,0], 
                    [0,0,0,1]
                ],
                [
                    [_math.cos(rotate[1]), 0, _math.sin(rotate[1]), 0],
                    [0,1,0,0], 
                    [-_math.sin(rotate[1]), 0, _math.cos(rotate[1]),  0],
                    [0,0,0,1]
                ],
                [
                    [1,0,0,0], 
                    [0,_math.cos(rotate[2]), _math.sin(rotate[2]),  0],
                    [0,-_math.sin(rotate[2]), _math.cos(rotate[2]), 0],
                    [0,0,0,1]
                ],
                [
                    [scale[0],0,0, translate[0]],
                    [0,scale[1],0, translate[1]],
                    [0,0,scale[2], translate[2]],
                    [0,0,0,1]
                ],
            ], device=device)
            a = affines[0] @ affines[1] @ affines[2] @ affines[3]
            field = _F.pad(field, (0,1), "constant", 1) @ a.mT
            field = field[...,:3]
            #field = field.squeeze(-2) 
        
            field = field - linfield(size, device=device)
            if "iter" in window_mode and i < iterations-1:
                field = field * w1.permute(1,2,3,0)
            field = field + linfield(size, device=device)
            field = field.permute(-1,0,1,2)

    field = field.permute(1,2,3,0)
    #field = field.permute(1,2,3,0)
    if False and affine:
        #s = 1
        rotate = (_torch.rand(1).sub(0.5) * _math.pi/4)
        s = (rotate.sin().abs() + rotate.cos().abs())
        
        affines = _torch.tensor([
             [
                    [s,0,0,0], 
                    [0,_math.cos(rotate[0])*s, _math.sin(rotate[0])*s,  0],
                    [0,-_math.sin(rotate[0])*s, _math.cos(rotate[0])*s, 0],
                    [0,0,0,1]
                ],
                
        ], device=device)
        a = affines[0]# @ affines[1] @ affines[2] @ affines[3]
        field = _F.pad(field, (0,1), "constant", 1) @ a.mT
        field = field[...,:3]
        #field = field.squeeze(-2) 
    
    field = field - linfield(size, device=device)
    if "global" in window_mode:
        field = field * w2.permute(1,2,3,0)
    field = field + linfield(size, device=device)
    
    
    if no_field:
        field = field - linfield(size, device=device)
    return field

def gaussian_at(kernel_size, position=0, sigma=1, sym=True):
    n = position + _torch.linspace(-1,1,kernel_size)
    sig2 = (2 * sigma * sigma)
    w = _torch.exp(-(n ** 2 / sig2))
    return w

def invert_permutation(p):
    p = [v if v >= 0 else (v+len(p)) for v in p]
    return [p.index(i) for i in range(len(p))]

def reduce(input, reduction="mean", dim=None, keepdim=False, 
           **kwargs):
    if isinstance(reduction, dict):
        squeeze = []
        r = input
        for dim, reduction in reduction.items():
            if dim is None: dim = list(range(r.dim()))
            if not isinstance(dim, (tuple, list)):
                dim = [dim]
            dim = [d if d >=
                   0 else d + r.dim() for d in dim]
            r = reduce(r, reduction, dim, keepdim=True)
            dim = [d for d in dim if d not in squeeze]
            squeeze.extend(dim)
    
        for d in reversed(sorted(squeeze)):
            r = r.squeeze(d)
        return r
    else:
        if dim is not None:
            if isinstance(dim, (tuple, list)):
                if len(dim) == 0: return input
                if len(dim) == 1: dim = dim[0]
        if "mean" in reduction:
            r = input.mean(dim, keepdim=keepdim)
        elif "sum" in reduction:
            r = input.sum(dim, keepdim=keepdim)
        elif "prod" in reduction:
            if isinstance(dim, (tuple, list)):
                permute = [v for v in range(input.dim()) if v not in dim]
                permute = permute + list(dim)
                input = input.permute(permute)
                input = input.flatten(-len(dim))
                r = input.prod(-1)
                if keepdim:
                    r = r.view(*r.shape, *[1 for _ in dim])
                    r = r.permute(invert_permutation(permute))
            elif dim is not None:
                r = input.prod(dim, keepdim=keepdim)
            else:
                r = input.prod()
                if keepdim: r = r.view([1]*input.dim())
        elif "max" in reduction:
            if isinstance(dim, (tuple, list)):
                permute = [v for v in range(len(dim)) if v not in dim]
                permute = permute + list(dim)
                input = input.permute(permute)
                input = input.flatten(-len(dim))
                r = input.max(-1)[0]
                if keepdim:
                    r = r.view(*r.shape, *[1 for _ in dim])
                    r = r.permute(invert_permutation(permute))
                
            else:
                r = input.max(dim, keepdim=keepdim)[0]
        elif "min" in reduction:
            if isinstance(dim, (tuple, list)):
                permute = [v for v in range(len(dim)) if v not in dim]
                permute = permute + list(dim)
                input = input.permute(permute)
                input = input.flatten(-len(dim))
                r = input.min(-1)[0]
                if keepdim:
                    r = r.view(*r.shape, *[1 for _ in dim])
                    r = r.permute(invert_permutation(permute))
            else:
                r = input.min(dim, keepdim=keepdim)[0]
        elif "l2" in reduction:
            r = input.pow(2).sum(dim, keepdim=keepdim).sqrt()
            r = _torch.where(_torch.isnan(r), _torch.zeros_like(r), r)
        elif "none" in reduction:
            r = input
        else:
            assert False, "invalid reduction"
        return r

def gaussian(size, position=0, *, sigma=1, dims=None, reduction="prod", **kwargs):
    if dims is None: dims = len(size)
    size = tensor(size, dims)
    sigma = tensor(sigma, dims)
    position = tensor(position, dims)
    #if position.dtype not in { _torch.float, _torch.double }:
    #    position = (position / size) * 2 - 1
    #if sigma.dtype not in { _torch.float, _torch.double }:
    #    sigma = (sigma / size) * 2 - 1
    args = [gaussian_at(ks, p, s) for ks,p,s in zip(size, position, sigma)]
    mesh = _torch.meshgrid(*args, indexing="ij")
    r = _torch.stack(mesh)
    return reduce(r, 0, reduction)

def window(size, type="gaussian", pow=1, **kwargs):
    if type in {"gaussian"}: r = gaussian(size, **kwargs)
    else:
        args = [_torch.linspace(-1, 1, ks) for ks,s in zip(size)]
        mesh = _torch.meshgrid(*args, indexing="ij")
        r = _torch.stack(mesh)
        if type in {"cos", "cosine"}:
            r = r.mul(_math.pi).cos().add(1).div(2)
        elif type in {"exp","exponential"}:
            r = r.abs().neg().exp()
        elif type in {"pyr", "pyramid","pyramidal"}:
            r = r.abs().neg().add(1)
        elif type in {"square", "box"}:
            r = _torch.ones_like(r[:1])
        elif type in {"circle", "circular", "sphere", "spherical"}:
            r = r.pow(2).sum(0, keepdim=True).sqrt().lt(1).float()
        elif type in {"cone", "conical"}:
            r = r.pow(2).prod(0,keepdim=True).sqrt().lt(1).float()
        r = reduce(r, 0, **kwargs)
    r = r.pow(pow)
    r = _torch.nan_to_num(r)
    return r
    
def distance_to_mask(pt, masks):
    brs = []
    pt = _torch.view_as_complex(pt)
            
    for mask in masks:
        #print(mask.shape)
        contours,_ = cv2.findContours(mask.view(*mask.shape[-2:]).gt(0.3).byte().cpu().numpy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #contours = sorted(contours, key = lambda v: cv2.moments(v)["m00"])
        rs = []
        for c in contours:
            pts = _torch.from_numpy(c).to(masks.device).float()
            pts = _torch.stack((
                2 * pts[...,0] / mask.shape[-1] - 1,
                2 * pts[...,1] / mask.shape[-2] - 1),
                              -1)
            r = []
            pts = _torch.view_as_complex(pts).squeeze(-1)
            #print(pts.shape)
            for p0, p1 in zip(pts, pts.roll(1,-1)):
                pq = pt - p0
                at = p1 - p0
                pq = pq * (at.conj() / at.abs().add(1e-8))

                sabs = lambda z: z.abs() #* z.imag.sign()
                v = _torch.where(pq.real < 0,
                            sabs(pq),
                            _torch.where(pq.real > at.abs(),
                                sabs(pq.sub(at.abs())),
                                pq.imag.abs()))
                r.append(v)
            r = _torch.stack(r)
            rs.append(r)
        if not len(rs): rs = _torch.zeros_like(mask)
        else: rs = _torch.cat(rs)
        #idxs = rs.abs().argmin(0, keepdim=True)
        #rs = F.relu(torch.gather(rs, 0, idxs))
        rs = _torch.nan_to_num(rs.min(0)[0])
        rs = rs * (1-mask)
        brs.append(rs)
    r = _torch.stack(brs)
    assert not _torch.isnan(r).any()
    return r

def coords_to_positions(coords, shapes, ranges=None, indexing="xy"):
    
    shapes = tensor(shapes)
    coords = tensor(coords)
    if indexing == "xy":
        shapes = shapes.flip(-1)
        
    if shapes.dim() > 1:
        assert coords.shape[0] == shapes.shape[0]
        return _torch.stack([coords_to_positions(c,s,ranges)
                            for c,s in zip(coords, shapes)])
    if coords.dtype in { _torch.float, _torch.double }:
        if ranges is None: ranges = (0,1)
        ranges = tensor(ranges)
        ranges = ranges.expand(*shapes.shape, 2)
        b = ranges[...,:1]
        a = ranges[...,1:] - b
        assert coords.shape[-1] == shapes.shape[-1]
        return (((coords-b)/a) * shapes).long()
    elif coords.dtype in { _torch.cfloat, _torch.cdouble }:
        assert shapes.shape[-1] == 2
        if ranges is None: ranges = (-1,1)
        ranges = tensor(ranges)
        ranges = ranges.expand(*shapes.shape, 2)
        b = ranges[...,:1]
        a = ranges[...,1:] - b
        coords = _torch.view_as_real(coords)
        return (((coords-b)/a) * shapes).long()
    elif coords.dtype in { _torch.long }:
        return coords
    else:
        assert False, "invalid dtype"        
    
def voxel_unshuffle(x, kernel_size):
    s = x.shape
    x = x.view(-1, kernel_size, *s[-2:])  # [N...] * Z//k, k, H, W
    x = _F.pixel_unshuffle(x, kernel_size) # [N...] * Z//k, k**3, H//k, W//k
    x = x.view(*s[:-3], -1, *x.shape[-3:])        # [N...], Z//k, k**3, H//k, W//k
    x = x.transpose(-4,-3)                     # [N...], k**3, Z//k, H//k, W//k
    x = x.reshape(*x.shape[:-5], -1, *x.shape[-3:])
    return x

def voxel_shuffle(x, kernel_size):
    s = x.shape
    x = x.view(*s[:-4],-1, kernel_size**3, *s[-3:])
    x = x.transpose(-4, -3)                  # [N...], Z//k, k**3, H//k, W//k
    x = x.reshape(-1, *x.shape[-3:])         # [N...] * Z//k, k**3, H//k, W//k
    x = _F.pixel_shuffle(x, kernel_size) # [N...] * Z//k, k, H, W
    x = x.reshape(*s[:-4], -1, s[-3]*kernel_size,  *x.shape[-2:])  # [N...], Z, H, W
    return x

def vector_length(x, dim=1, eps=1e-16):
    return x.mul(x).sum(dim, keepdim=True).add(eps).sqrt()

def vector_norm(x, dim=1, eps=1e-16):
    return x / vector_length(x, eps=eps)

def guided_filter(x,guide=None,kernel_size=31,eps=1e-5):
    if guide is None: guide = x
    pool = lambda v,k: _F.avg_pool2d(v, k, stride=1) * k * k
    g = guide
    k = kernel_size
    e = eps
    pad = (k//2, k-k//2-1)
    y = _F.pad(x, pad*2, "reflect")
    gp = _F.pad(g, pad*2, "reflect")
    mean_y1 = pool(y, k)
    mean_g1 = pool(gp, k)
    mean_g2 = pool(gp*gp, k)
    mean_gy = pool(gp*y, k)
    var_g2 = (mean_g2 - mean_g1 * mean_g1) / (k*k)
    cov_gy = (mean_gy - mean_g1 * mean_y1) / (k*k)
    a = cov_gy / (var_g2 + e)    
    b = mean_y1 - a * mean_g1
    a = _F.pad(a, pad*2, "reflect")
    b = _F.pad(b, pad*2, "reflect")
    a = pool(a, k) / (k*k)
    b = pool(b, k) / (k*k)
    q = (a * g + b)
    return q