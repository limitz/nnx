import torch as _torch
import torch.nn.functional as _F
import numpy as _np
import cv2 as _cv2
import math as _math
import matplotlib.pyplot as _plt
from .. import functional as _Fx
from . import text as _text

def maze(size=(15,15), colorspace="idx"):
    if isinstance(size, int):
        size = (size-1, size-1)
    else:
        assert isinstance(size, (list, tuple))
        size = tuple(v-1 for v in size)
        
    def step(maze, y, x, y_end, x_end):
        h,w = size
        visited = lambda v: v[4]
        clamp = lambda v,s: min(max(0, v),size[s]-1)
        nb = lambda dy,dx: (maze[clamp(y+dy,0),clamp(x+dx,1)], dy, dx)
        maze[y,x,4] = 1
        if y == y_end and x == x_end:
            maze[y,x,5] = 1
        t,b = nb(-1,0), nb(1,0)
        l,r = nb(0,-1), nb(0,1)
        nbs = [t,l,r,b]
        for j in _torch.randperm(4):
            j = j.item()
            cell,dy,dx = nbs[j]
            if visited(cell): continue
            maze[y,x,j] = 1
            maze[y+dy,x+dx,(~j)&3] = 1
            maze, c = step(maze, y+dy, x+dx, y_end, x_end)
            if c: maze[y,x,5] = 1
        return maze, maze[y,x,5]
    
    h,w = size 
    start = _torch.randint(h,(1,)).item(), _torch.randint(w, (1,)).item()
    end = _torch.randint(h,(1,)).item(), _torch.randint(w, (1,)).item()
    length = (h*w)//8
    target = _torch.zeros(h,w,6, dtype=_torch.long)
    challenge = _torch.zeros(h,w,6, dtype=_torch.long)

    target,_ = step(target,*start, *end)
    challenge = target.clone()[None]
    challenge = _torch.stack((challenge[...,4], 
                             challenge[...,2], 
                             challenge[...,3], 
                             _torch.zeros_like(challenge[...,0])), 1)
    challenge = _F.pixel_shuffle(challenge, 2).squeeze(1)
    target = target[None]
    target = target[...,[5]].repeat(1,1,1,4).permute(0,3,1,2)
    target[:,1:] = 0
    target = _F.pixel_shuffle(target, 2).squeeze(1)
    target[...,1:-1:2,:] = target[...,0:-2:2,:] * target[...,2::2,:]
    target[...,1:-1:2] = target[...,0:-2:2] * target[...,2::2]
    target = challenge + ((target) * challenge)
    challenge[:,start[0]*2,start[1]*2] = 2
    challenge[:,end[0]*2,end[1]*2] = 2
    challenge = _F.pad(challenge,(1,0,1,0))
    target = _F.pad(target,(1,0,1,0))
    if colorspace == "rgb":
        return _text.idx_to_rgb(challenge), _text.idx_to_rgb(target)
    else:
        return challenge, target