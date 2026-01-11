import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _F
import numpy as _np

def embed(dst, src, bits=1, in_place=False):
    assert src.dtype == _torch.uint8
    
    if not in_place:
        dst = dst.clone()
    
    is_complex = dst.dtype in {_torch.cfloat, _torch.cdouble}
    if is_complex:
        dst = _torch.view_as_real(dst)
        
    dtype = dst.dtype
    if dtype == _torch.float:
        dst = dst.view(_torch.uint32)
    elif dtype == _torch.double:
        dst = dst.view(_torch.uint64)
    elif dtype == _torch.bool:
        dst = dst.byte()

    # todo embed shape
    
    shape = dst.shape
    dst = dst.view(-1)
    idx = 0
    mask = _torch.tensor((1<<bits) - 1).to(dst.dtype)
    for byte in src.view(-1):
        for b in range(0,8,bits):
            v = (byte >> b).to(mask.dtype) & mask
            d = (dst[idx] & mask) ^ v
            dst[idx] ^= d
            idx += 1
            
    dst = dst.view(shape)
    dst = dst.view(dtype)
    if is_complex:
        dst = _torch.view_as_complex(dst)

    return dst

def unembed(dst, shape=None, bits=1):
    if shape is None:
        shape = [dst.numel() // 8]
    
    # todo embed shape
    
    length = _np.prod(shape)
    is_complex = dst.dtype in {_torch.cfloat, _torch.cdouble}
    if is_complex:
        dst = _torch.view_as_real(dst)
        
    dtype = dst.dtype
    if dtype == _torch.float:
        dst = dst.view(_torch.uint32)
    elif dtype == _torch.double:
        dst = dst.view(_torch.uint64)
    elif dtype == _torch.bool:
        dst = dst.byte()

    b = 0
    v = 0
    r = []
    mask = _torch.tensor((1<<bits) - 1).to(dst.dtype)
    for element in dst.view(-1):
        v += (element & mask).byte() << b
        b += bits
        if b >= 8:
            r.append(v)
            if len(r) >= length:
                break
            b = 0
            v = 0
    return _torch.tensor(r).view(shape)