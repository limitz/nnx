import torch as _torch
import torch.nn.functional as _F
import numpy as _np
import cv2 as _cv2
import math as _math
import matplotlib.pyplot as _plt
from .. import functional as _Fx
from . import text as _text

def render_sudoku(n=3, difficulty=0.4, font=_text.NUMERIC_5X3, spacing=(1,3), padding=1, device="cpu", colorspace="idx"):
    digits = {2:"1234",
              3:"123456789", 
              4:"0123456789ABCDEF",
              5:"ABCDEFGHIJKLMNOPQRSTUVWXY",
              6:"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"}
    assert n in digits
    assert all(d in font for d in digits[n])
    
    d = digits[n]
    lines = _torch.stack([_torch.arange(n**2).roll(i).roll(j*n) for i in range(n) for j in range(n)])
    for i in range(n):
        for r in range(0,n**2,n):
            lines[r:r+n,:] = lines[_torch.randperm(n)+r,:]
        for c in range(0,n**2,n):
            lines[:,c:c+n] = lines[:,_torch.randperm(n)+c]
    
    if not isinstance(spacing, (list, tuple)):
        spacing = [spacing] * 2
    if not isinstance(padding, (list, tuple)):
        padding = [padding] * 4
    if len(padding) == 2:
        padding = [padding[0],padding[0],padding[1],padding[1]]
    assert len(padding) == 4
    max_length = max(len(line) for line in lines)
    rs, ts = [],[]
    for j,line in enumerate(lines):
        r,t = [],[]
        for i,c in enumerate(line):
            tc = d[c.item()]
            rc = " " if _torch.rand(1) < difficulty else tc
            rc = _torch.tensor(font[rc], dtype=_torch.long, device=device)
            rc = _F.pad(rc, (0, spacing[1] if i < len(line)-1 else 0, 
                            0, spacing[0] if j < len(lines)-1 else 0))
            tc = _torch.tensor(font[tc], dtype=_torch.long, device=device)
            tc = _F.pad(tc, (0, spacing[1] if i < len(line)-1 else 0, 
                            0, spacing[0] if j < len(lines)-1 else 0))
            r.append(rc)
            t.append(tc)
        rs.append(_Fx.padcat(r, -1))
        ts.append(_Fx.padcat(t, -1))
    rs = _Fx.padcat(rs, -2)
    rs = _F.pad(rs, padding)
    ts = _Fx.padcat(ts, -2)
    ts = _F.pad(ts, padding)
    
    if colorspace == "rgb":
        rs = _text.idx_to_rgb(rs)
        ts = _text.idx_to_rgb(ts)
        return rs.to(device), ts.to(device)
    else:
        return rs[None].to(device), ts[None].to(device)
