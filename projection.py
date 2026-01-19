import math as _math
import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _F
import cv2 as _cv2
import torchvision as _torchvision
import torchvision.transforms.v2 as _T
import torch.linalg as _L
import numpy as _np
from . import functional as _Fx
from . import graphics as _G

grid_sample = _Fx.grid_sample
flow_sample = _Fx.flow_sample
flow_to_grid = _Fx.flow_to_grid
grid_to_flow = _Fx.grid_to_flow
linfield = _Fx.linfield

def unshear(matrix):
    assert matrix.shape[-1] == matrix.shape[-2]
    if matrix.shape[-1] == 3:
        x,y = matrix[...,0:2,0:2].pow(2).sum(-2).sqrt().unbind(-1)
        mask = _torch.zeros_like(matrix).bool()
        mask[...,:2,1:2] = 1
        value = _torch.zeros_like(matrix)
        value[...,0,1] = matrix[...,1,0] * y / x * -1
        value[...,1,1] = matrix[...,0,0] * y / x
        return _torch.where(mask, value, matrix)
    else:
        assert False, "currently only 3x3 matrices are supported"
        
def perspective_grid(theta, size, homogenous=False):
    n,c,h,w = size
    ls = lambda s: _torch.linspace(-1, 1, s,device=theta.device) * (s-1) / s
    grid = _torch.zeros(n,h,w,3, device=theta.device)
    grid.select(-1, 0).copy_(ls(w))
    grid.select(-1, 1).copy_(ls(h).unsqueeze(-1))
    grid.select(-1, 2).fill_(1)
    grid = grid.view(n,h*w,3) @ theta.transpose(1,2)
    grid = grid.view(n,h,w,3)
    if not homogenous:
        grid = grid[:,:,:,:2] / grid[:,:,:,[2]]
    return grid

def perspective_matrix(src_points, dst_points, mode="solve"):
    assert src_points.shape == dst_points.shape
    if dst_points.dim() > 2:
        return _torch.stack([perspective_matrix(s,d, mode) 
                            for s,d in zip(src_points, dst_points)])

    if _torch.isnan(src_points).any() or _torch.isnan(dst_points).any(): 
        return _torch.full((3,3), float("nan"), device=dst_points.device)

    s0,s1,s2,s3 = src_points.float()
    d0,d1,d2,d3 = dst_points.float()
    x0,y0 = s0
    x1,y1 = s1
    x2,y2 = s2
    x3,y3 = s3
    u0,v0 = d0
    u1,v1 = d1
    u2,v2 = d2
    u3,v3 = d3
    o = _torch.tensor(1, device=src_points.device)
    z = _torch.tensor(0, device=src_points.device)
    A = _torch.stack([
        _torch.stack([x0, y0, o,  z,  z, z, -x0*u0, -y0*u0]),
        _torch.stack([x1, y1, o,  z,  z, z, -x1*u1, -y1*u1]),
        _torch.stack([x2, y2, o,  z,  z, z, -x2*u2, -y2*u2]),
        _torch.stack([x3, y3, o,  z,  z, z, -x3*u3, -y3*u3]),
        _torch.stack([ z,  z, z, x0, y0, o, -x0*v0, -y0*v0]),
        _torch.stack([ z,  z, z, x1, y1, o, -x1*v1, -y1*v1]),
        _torch.stack([ z,  z, z, x2, y2, o, -x2*v2, -y2*v2]),
        _torch.stack([ z,  z, z, x3, y3, o, -x3*v3, -y3*v3])])
    B = _torch.stack([u0,u1,u2,u3,v0,v1,v2,v3])

    if mode == "solve":
        x = _L.solve(A,B)
    elif mode == "pseudo":
        x = A.pinverse() @ B
    else:
        x = A.inverse() @ B

    x = _F.pad(x, [0,1], "constant", 1)
    x = x.view(3,3)
    return x

def project(x, thetas, size=None, mode="bilinear", padding_mode="zeros"):
    r = []
    if isinstance(x,list):
        for vx, vt in zip(x, thetas.split(1,1)):
            size = size or vx.shape[-2:]
            grid = perspective_grid(vt, (vt.shape[0], 1, *size))
            r.append(grid_sample(vx.expand(vt.shape[0],-1,-1,-1), grid, align_corners=True, mode=mode, padding_mode=padding_mode))
        r = _torch.stack(r)
    else:
        size = size or x.shape[-2:]
        n = x.shape[0]
        thetas = _torch.nan_to_num(thetas)
        if thetas.dim() > 3:
            original_shape = thetas.shape[:-2]
            if True:
                for theta in thetas.reshape(-1,n,3,3):
                    grid = perspective_grid(theta, (theta.shape[0],1,*size))
                    r.append(_Fx.grid_sample(x, grid, align_corners=True, mode=mode, padding_mode=padding_mode))
                r = _torch.stack(r)
            else:
                thetas = thetas.flatten(0,-4)
                grid = perspective_grid(thetas, (thetas.shape[0],1,*size))
                grid = _Fx.grid_sample(x.repeat(thetas.shape[0],1,1,1), grid, align_corners=True, 
                                       mode=mode, padding_mode=padding_mode)
                r = grid
            r = r.view(*original_shape, *r.shape[-3:])
        else:
            grid = perspective_grid(thetas, (thetas.shape[0],1,*size)).to(x.device)
            r = _Fx.grid_sample(x, grid, align_corners=True, mode=mode, padding_mode=padding_mode)
    return r


def matrix_field_nd(m, size):
    d = len(size)
    field = linfield(size)
    field = _F.pad(field, (0,1), "constant", 1)
    field = field.unsqueeze(-2) @ m.view(*m.shape[:-2],1,1,1,d+1,d+1).mT
    field = field.squeeze(-2)
    field = field[...,:-1] / field[...,-1:]
    return field
    
def project_nd(x, thetas, size=None, mode="bilinear", padding_mode="zeros", split_dim=None):
    r = []
    d = thetas.shape[-1]
    if isinstance(x,list):
        for v, t in zip(x, thetas.split(1,-3)):
            y = project_3d(v, t, size=size, mode=mode, padding_mode=padding_mode, split_dim=split_dim)
            r.append(y)
        return _torch.cat(r,-d-2)
    else:
        x = x.expand(*(thetas.shape[:-2] + [-1] * (d+1)))
        if split_dim is not None:
            assert split_dim > 0
            for v, t in zip(x.split(1,split_dim), thetas.split(1, split_dim)):
                y = project_3d(v, t, size=size, mode=mode, padding_mode=padding_mode, split_dim=None)
                r.append(y)
            return _torch.cat(r, split_dim)
        else:
            thetas = _torch.nan_to_num(thetas)
            size = size or x.shape[-d:]
            field = matrix_field_nd(thetas, size)
            ...
            y = grid_sample(x.flatten(0,-d-2), grid, 
                            align_corners=True, mode=mode, 
                            padding_mode=padding_mode)
            return _torch.stack(r)

def gaussian_splat_2d(thetas, size, weights=None, sigma=1, reduction="none"):
    h,w = size
    p,n,*_ = thetas.shape
    pt = _torch.stack(
        _torch.meshgrid(
            _torch.linspace(-1,1,w),
            _torch.linspace(-1,1,h),
            indexing="xy"),-1).to(thetas.device)
    fields = _F.pad(pt, (0,1), "constant", 1).expand(p, n, -1, -1, -1)
    fields = fields.unsqueeze(-2)
    fields = fields @ thetas[...,None,None,:,:].inverse().mT
    fields = fields[...,:-1] / fields[...,-1:]
    fields = fields.squeeze(-2)
    r = fields.pow(2).mul(-2*sigma).exp().prod(-1)
    if weights is not None:
        #print(r.shape, weights.shape)
        r = r * weights
    return _Fx.reduce(r, reduction, dim=0)
    
def gaussian_splat_3d(thetas, size, weights=None, sigma=1, reduction="none"):
    d,h,w = size
    p,n,*_ = thetas.shape
    pt = _torch.stack(
        _torch.meshgrid(
            _torch.linspace(-1,1,h),
            _torch.linspace(-1,1,d),
            _torch.linspace(-1,1,w),
            indexing="xy"),-1).to(thetas.device)
    #print(pt.shape)
    fields = _F.pad(pt, (0,1), "constant", 1).expand(p, n, -1, -1, -1, -1)
    fields = fields.unsqueeze(-2)
    fields = fields @ thetas[...,None,None,None,:,:].inverse().mT
    fields = fields[...,:-1] / fields[...,-1:]
    fields = fields.squeeze(-2)
    r = fields.pow(2).mul(-2*sigma).exp().prod(-1)
    if weights is not None:
        r = r * weights
    return _Fx.reduce(r, reduction, dim=0)

def masks_to_thetas(masks, margin=0.1):
    batch_thetas = []
    
    masks = _F.max_pool2d(masks, 5, stride=1, padding=2)
    masks = _F.max_pool2d(masks, 5, stride=1, padding=2)
    masks = -_F.max_pool2d(-masks, 5, stride=1, padding=2)
    masks = -_F.max_pool2d(-masks, 5, stride=1, padding=2)
    
    for mask in masks:
        thetas = []
        if mask.eq(0).all():
            thetas.append(torch.eye(3, device=mask.device))
        else:
            contours,_ = _cv2.findContours(mask.view(*mask.shape[-2:]).byte().cpu().numpy(), 
                                           _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
            #todo multi target
            contours = sorted(contours, key = lambda v: _cv2.moments(v)["m00"])
            if len(contours) > 1:
                contours = [_cv2.convexHull(_np.vstack(contours))]
            for c in contours:
                quad = _cv2.approxPolyN(c,4)
                quad = _torch.from_numpy(quad).to(mask.device)
                first = quad.pow(2).sum(-1).argmin(-1)
                quad = quad.roll(-first.item(), -2)[0]
                quad = quad / _torch.tensor([mask.shape[-1]//2, mask.shape[-2]//2], device=quad.device) - 1
                quad = quad + _torch.tensor([-margin,-margin,margin,-margin,margin,margin,-margin,margin], device=quad.device).view(4,2)
                dest = _torch.tensor([-1,-1,1,-1,1,1,-1,1.], device=quad.device).view(4,2).expand_as(quad)
                theta = perspective_matrix(dest, quad)
                thetas.append(theta)
        thetas = _torch.stack(thetas)
        batch_thetas.append(thetas)
    return _torch.stack(batch_thetas).squeeze(-3)

def logits_to_thetas_2d( ps, q=0):
    scale,ratio,angle,distort,tx,ty = ps.split([1,1,1,4,1,1],-1)
    tx,ty = tx, ty
    scale = scale.exp()
    ratio = ratio.exp()
    dx,dy,sx,sy = distort.mul(q).split(1,-1)
    sx,sy = sx*0.1, sy*0.1
    angle = angle.mul(_math.pi)
    theta = _torch.cat([
        scale * angle.cos() + sx * sy, ratio * -scale * angle.sin() + sy, tx,
        scale * angle.sin() + sx, ratio * scale * angle.cos(), ty,
        dx, dy, _torch.ones_like(scale)], -1).unflatten(-1,(3,3))
    return theta

logits_to_thetas = logits_to_thetas_2d

def logits_to_thetas_3d( ps, q=0):
    scale,ratio,angle,distort,t = ps.split([1,2,3,12,3],-1)
    scale = _torch.cat([scale,scale+ratio[...,[0]],scale+ratio[...,[1]]],-1)
    ratio = _torch.zeros_like(scale)
    dx,dy,sx,sy = distort.split(3,-1)
    tt = _torch.zeros_like(scale)#t.unsqueeze(-2) * _torch.eye(3,device=t.device)
    ms = _torch.stack([scale, ratio, angle, dx, dy, sx, sy, tt, tt], -1)
    x,y,z = logits_to_thetas_2d(ms).unbind(-3)
    insert_row = lambda m, i: _torch.cat([m[...,:i,:], 
                                        _torch.zeros_like(m[...,[0],:]),
                                        m[...,i:,:]],-2)
    insert_col = lambda m, i: _torch.cat([m[...,:i], 
                                        _torch.zeros_like(m[...,[0]]),
                                        m[...,i:]],-1)                                                     
    insert = lambda m, i: insert_row(insert_col(m,i),i)
    x = insert(x,2) + _torch.tensor([[0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,0.]], device=x.device)
    y = insert(y,1) + _torch.tensor([[0,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0.]], device=y.device)
    z = insert(z,0) + _torch.tensor([[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0.]], device=z.device)
    theta = x @ y @ z
    theta = _torch.cat([
        _torch.cat([theta[...,:-1,:-1], t.unsqueeze(-1)], -1),
        theta[...,-1:,:]], -2)
    
    return theta

def thetas_to_xywh(theta, image_size):
    assert theta.dim() in { 3, 4 }
    assert theta.shape[-1] == 3
    assert theta.shape[-2] == 3
    if theta.dim() == 4: 
        theta = theta.transpose(0,1)
    lshape = theta.shape[:-2]
    src = _torch.tensor([[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]], device=theta.device, dtype=_torch.float)
    dst = src.expand_as(theta) @ theta.mT
    dst = dst[...,:2] / dst[...,[2]]
    dst = dst * 0.5 + 0.5
    dst[...,0] = dst[...,0] * image_size[-1]
    dst[...,1] = dst[...,1] * image_size[-2]
    l,r = dst[...,0].min(), dst[...,0].max()
    t,b = dst[...,1].min(), dst[...,1].max()
    box = _torch.stack((l,t,r-l,b-t), -1)
    return box
    
def xywh_to_thetas(boxes, image_size):
    assert boxes.dim() in { 2, 3 }
    assert boxes.shape[-1] == 4
    lshape = boxes.shape[:-1]
    x = 2 * boxes[...,0] / image_size[-1] - 1
    y = 2 * boxes[...,1] / image_size[-2] - 1
    w = 2 * boxes[...,2] / image_size[-1]
    h = 2 * boxes[...,3] / image_size[-2]
    pts = _torch.stack([
        _torch.stack((x,y),-1),
        _torch.stack((x+w,y),-1),
        _torch.stack((x+w,y+h),-1),
        _torch.stack((x,y+h),-1)], -2)
    
    src = _torch.tensor([[-1,-1],[1,-1],[1,1],[-1,1]], device=boxes.device, dtype=_torch.float)
    pts = pts.view(-1,4,2)
    theta = perspective_matrix(src.repeat(pts.shape[0],1,1),pts, mode="pseudo")
    theta = theta.reshape(*lshape,3,3)
    if boxes.dim() == 3: theta = theta.transpose(0,1)
    return theta

def ltrb_to_thetas(boxes, image_size):
    assert boxes.dim() in { 2, 3 }
    assert boxes.shape[-1] == 4
    lshape = boxes.shape[:-1]
    l = 2 * boxes[...,0] / image_size[-1] - 1
    t = 2 * boxes[...,1] / image_size[-2] - 1
    r = 2 * boxes[...,2] / image_size[-1] - 1
    b = 2 * boxes[...,3] / image_size[-2] - 1
    pts = _torch.stack([
        _torch.stack((l,t),-1),
        _torch.stack((r,t),-1),
        _torch.stack((r,b),-1),
        _torch.stack((l,b),-1)], -2)
    
    src = _torch.tensor([[-1,-1],[1,-1],[1,1],[-1,1]], device=boxes.device, dtype=_torch.float)
    pts = pts.view(-1,4,2)
    theta = perspective_matrix(src.repeat(pts.shape[0],1,1),pts, mode="pseudo")
    theta = theta.reshape(*lshape,3,3)
    if boxes.dim() == 3: theta = theta.transpose(0,1)
    return theta
    
def quad_to_thetas(quads, image_size):
    assert boxes.dim() in { 2, 3 }
    assert boxes.shape[-1] == 4
    lshape = quads.shape[:-2]
    s = _torch.tensor(image_size).flip(-1)
    pts = 2 * quads / s - 1
    src = _torch.tensor([[-1,-1],[1,-1],[1,1],[-1,1]], device=boxes.device, dtype=_torch.float)
    pts = pts.view(-1,4,2)
    theta = perspective_matrix(src.repeat(pts.shape[0],1,1),pts)
    theta = theta.reshape(*lshape,3,3)
    if boxes.dim() == 3: theta = theta.transpose(0,1)
    return theta

@_torch.no_grad()
def draw_thetas(image, thetas, weights=None, thickness=2):
    ''' Utility function to draw quads onto `dst` for perspective matrices `theta`
        image is in the format NCHW
        thetas is in the format TN33 ([thetas per image] x num_batches x 3 x 3)
        weights is in the format TN ([thetas per image] x num_batches)
    '''
    if isinstance(image, list):
        dst = _torch.stack(image)
        thetas = thetas.transpose(0,1)
    else:
        dst = image.clone()
    assert dst.dim() == 4
    assert thetas.dim() == 4
    assert thetas.shape[-1] == 3
    assert thetas.shape[-2] == 3
    assert weights is None or thetas.shape[:-2] == weights.shape
    size = dst.shape[-2:]
    n = dst.shape[0]
    s = _torch.tensor(size, device=thetas.device)
    sflipped = s.flip(-1)
    for i,(d,t) in enumerate(zip(dst, thetas.detach().transpose(0,1))):
        t = t[~_torch.isnan(t.flatten(-2)).any(-1)]
        if t.shape[0] <= 0: continue
        pts0 = _torch.tensor([-1,-1,1,1,-1,1,1,1,1,-1,1,1], device=t.device).view(4,3).repeat(t.shape[0],1,1).float()
            
        #if torch.isnan(t).any(): continue
        mask = _np.zeros((*size, 1), dtype=_np.uint8)
        pts1 = pts0 @ t.mT
        pts2 = _np.int32(((pts1[:,:,:2]/pts1[:,:,[2]]) * sflipped/2 + sflipped/2).cpu().numpy())
        if weights is None:
            _cv2.polylines(mask, pts=pts2, isClosed=True, color=1, thickness=thickness)
            color = mask * 255
        else:
            ws = weights[:,i]
            color = _np.zeros((*size, 3), dtype=_np.uint8)
            for pts,w in zip(pts2,ws):
                #if w <= 0.5: continue
                red = (1-w.item())  ** 0.5
                green = (w.item()) ** 0.5
                _cv2.polylines(mask, pts=[pts], isClosed=True, color=1, thickness=thickness)
                _cv2.polylines(color, pts=[pts], isClosed=True, color=(int(red*255), int(green*255), 0), thickness=thickness)
        mask = _torch.from_numpy(mask).to(dst.device).permute(2,0,1)
        color = _torch.from_numpy(color).to(dst.device).permute(2,0,1)
        d[mask.expand_as(d) > 0] = 0
        d += color / 255
    return list([d[0] for d in dst.split(1)]) if isinstance(image, list) else dst

@_torch.no_grad()
def draw_attention(image, thetas, weights, power=1):
    ''' Utility function to draw quads onto `dst` for perspective matrices `theta`
        image is in the format NCHW
        thetas is in the format T*N33 ([any dim] x num_batches x 3 x 3)
        weights is in the format T*N ([any dim] x num_batches)
    '''
    if isinstance(image, list):
        dst = _torch.stack(image)
        thetas = thetas.transpose(0,1)
    else:
        dst = image.clone()
    assert dst.dim() == 4
    assert thetas.dim() == 4
    assert thetas.shape[-1] == 3
    assert thetas.shape[-2] == 3
    assert weights is None or thetas.shape[:-2] == weights.shape
    n,_,*size = dst.shape
    thetas = thetas.reshape(-1,n,3,3)
    weights = weights.reshape(-1,n).cpu().numpy()
    
    s = _torch.tensor(size, device=thetas.device)
    sflipped = s.flip(-1)
    pts0 = _torch.tensor([-1,-1,1,1,-1,1,1,1,1,-1,1,1], device=thetas.device).view(4,3).repeat(thetas.shape[0],1,1).float()
    
    for i,(d,t) in enumerate(zip(dst, thetas.detach().transpose(0,1))):
        if _torch.isnan(t).any(): continue
        pts1 = pts0 @ t.mT
        pts2 = _np.int32(((pts1[:,:,:2]/pts1[:,:,[2]]) * sflipped/2 + sflipped/2).cpu().numpy())
        ws = weights[:,i]
        mask = _np.zeros((*size, 1), dtype=_np.float)
        attention = _np.zeros((*size, 1), dtype=_np.float)
        for pts,w in zip(pts2,ws):
            _cv2.fillPoly(mask, pts=[pts], color=1)
            attention += mask * w ** power
            mask[:] = 0
        attention = _torch.from_numpy(attention).float().to(dst.device).permute(2,0,1)
        attention /= attention.max()
        y = attention / 2 + image[i].mean(0) / 2
        u = -attention.mul(_math.pi/2).sin() * (attention)
        v = attention.mul(_math.pi/2).cos() * (attention)
        rgb = _G.yuv_to_rgb(torch.cat((y,u,v), -3))
        d[:] = rgb
    return list(dst.split(1)) if isinstance(image, list) else dst


@_torch.no_grad()
def project_patches(x, thetas, weights=None, border=0, margin=0, cols=None, rows=None, size=None, mode="bilinear", padding_mode="zeros"):
    '''thetas is in format TN33'''
    projections = []
    if cols is None and rows is None:
        if size is None:
            size = x.shape[-2:]
    else:
        if cols is not None and rows is not None:
            assert cols * rows >= thetas.shape[0]
            if size is None:
                size_x = x.shape[-1]//cols
                size_y = x.shape[-2]//rows
                size = (size_y, size_x)
        elif cols is not None:
            rows = (thetas.shape[0] + cols - 1) // cols
            if size is None:
                size_x = x.shape[-1]//cols
                size = (size_x, size_x)
        elif rows is not None:
            cols = (thetas.shape[0] + rows - 1) // rows
            if size is None:
                size_y = x.shape[-2]//rows
                size = (size_y, size_y)

    for t in thetas:
        projection = project(x, t[None], size, mode, padding_mode)
        projections.append(projection)
    projections = _torch.stack(projections) # TNCHW
    if weights is not None and border > 0:
        for p,w in zip(projections, weights):
            #color = torch.stack(((1-w),w,torch.zeros_like(w)), 1).sqrt().view(-1,3,1,1)
            color = _torch.stack(((1-w),w,_torch.zeros_like(w)), 1).view(-1,3,1,1)
            p[...] *= w[:,None,None,None]
            p[...,:border+margin,:] = color
            p[...,:,:border+margin] = color
            p[...,-(border+margin):,:] = color
            p[...,:,-(border+margin):] = color
    if margin > 0:
        projections[...,:margin,:] = 0
        projections[...,:,:margin] = 0
        projections[...,-margin:,:] = 0
        projections[...,:,-margin:] = 0
    if cols is not None:
        projections = _Fx.padcat(projections.chunk(cols), -1)
        projections = _Fx.padcat(projections.chunk(rows), -2)
        r = projections[0,0]
    else:
        r = projections[:,0]

    return r


@_torch.no_grad()
def unproject_patches(patches_or_x, thetas, weights=None, size=None, mode="bilinear", padding_mode="zeros", reduction="none", eps=1e-5, return_counts=False):
    '''patches is in format TNCHW, thetas is in format TN33'''
    if patches_or_x.dim() == 5:
        patches = patches_or_x
    elif patches_or_x.dim() == 4:
        patches = project(x, thetas, mode="nearest")
    else:
        assert False, "expected 4d or 5d input"
        
    projections = [] if reduction == "none" else None
    counts = None
    size = size or patches.shape[-2:]
    thetas = thetas.inverse()
    assert patches.shape[0] == thetas.shape[0]
    if weights is not None:
        assert patches.shape[0] == weights.shape[0]
        
    #for i, (p,t) in enumerate(zip(patches,thetas)):
    #    projection = project(p, t, size, mode)[None[
    group = 16
    for i in range(0,len(patches),group):
        p,t = patches[i:i+group], thetas[i:i+group]
        l = p.shape[0]
        p = p.view(-1, *p.shape[-3:])
        t = t.view(-1, 3, 3)
        projection = project(p, t, size, mode, padding_mode)
        projection = projection.view(l, -1, *projection.shape[-3:])
        count = None
        if projections is None:
            projections = _torch.zeros_like(projection)
        if weights is not None:
            projection = projection * weights[i:i+group,:,None,None,None]
        if reduction == "realmean" or return_counts:
            count = project(_torch.ones_like(p), t, size, mode, padding_mode)
            if weights is not None:
                count = count * weights[i:i+group,:,None,None,None]
            if counts is None:
                counts = _torch.zeros_like(projections)
            counts += count.sum(0, keepdim=True)
        
        if reduction == "none":
            projections.append(projection)
        elif reduction == "sum" or reduction == "mean" or reduction =="realmean":
            projections = projections + projection.sum(0, keepdim=True)
        elif reduction == "max":
            projections = torch.maximum(projections[0], projection.max(0)[0])[None]
        
    if reduction == "none":
        return _torch.cat(projections) # TNCHW
    if reduction == "mean":
        if weights is not None:
            projections = projections / weights.sum(0)[:,None,None,None].add(eps)
        else:
            projections = projections / patches.shape[0]
    elif reduction == "realmean":
        projections = projections / counts.add(1e-5)
    if return_counts:
        return projections[0], counts[0]
    else:
        return projections[0]

def matrix_remove_col(m, i):
    return _torch.cat([
        m[...,:i],
        m[...,i+1:]],-1)

def matrix_insert_col(m, i, eye=False):
    if eye: 
        insert = _torch.arange(m.shape[-2],device=m.device)
        insert = insert.eq(i).to(m.dtype)
        insert = insert.unsqueeze(-1)
        insert = insert.expand_as(m[...,[0]])
    else: 
        insert = _torch.zeros_like(m[...,[0]], device=m.device)
    return _torch.cat([
        m[...,:i],
        insert,
        m[...,i:]],-1)
    
def matrix_remove_row(m, i):
    return _torch.cat([
        m[...,:i,:],
        m[...,i+1:,:]],-2)
    
def matrix_insert_row(m, i, eye=False):
    if eye: 
        insert = _torch.arange(m.shape[-1],device=m.device)
        insert = insert.eq(i).to(m.dtype)
        insert = insert.unsqueeze(-2)
        insert = insert.expand_as(m[...,[0],:])
    else: 
        insert = _torch.zeros_like(m[...,[0],:], device=m.device)
    return _torch.cat([
        m[...,:i,:],
        insert,
        m[...,i:,:]],-2)

def matrix_rotate_2d(rad):
    return _torch.stack([
        rad.cos(), -rad.sin(), _torch.zeros_like(rad),
        rad.sin(), rad.cos(), _torch.zeros_like(rad),
        _torch.zeros_like(rad), _torch.zeros_like(rad),
        _torch.ones_like(rad)],-1).view(*rad.shape, 3,3)

def matrix_rotate_3d(rad, dim=2):
    rot2d = matrix_rotate_2d(rad)
    return matrix_insert_dim(rot2d,dim)
    
def matrix_remove_dim(m, i):
    m = matrix_remove_row(m,i)
    m = matrix_remove_col(m,i)
    return m

def matrix_insert_dim(m, i):
    m = matrix_insert_col(m,i,eye=False)
    m = matrix_insert_row(m,i,eye=True)
    return m

class TranslationLoss(_nn.Module):
    def __init__(self, gamma=12, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target=None):
        pos = pred[...,:-1,-1]
        r = pos.abs().pow(self.gamma).sum(-1,keepdim=True)
        return _Fx.reduce(r, self.reduction)

class RatioLoss(_nn.Module):
    def __init__(self, gamma=2, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target=None):
        if pred.shape[-1] > 3:
            a = self.forward(matrix_remove_dim(pred, 0))
            b = self.forward(matrix_remove_dim(pred, 1))
            return a+b       
        else:
            x = pred[...,1].pow(2).sum(-1).sqrt() 
            y = pred[...,0].pow(2).sum(-1).sqrt()
            r = (x/y).log().abs().pow(self.gamma)
            return _Fx.reduce(r, self.reduction)

class ShearingLoss(_nn.Module):
    def __init__(self, gamma=2, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target=None):
        if pred.shape[-1] > 3:
            a = self.forward(matrix_remove_dim(pred, 0))
            b = self.forward(matrix_remove_dim(pred, 1))
            return a+b       
        else:
            x = pred[...,1]
            y = pred[...,0]
            r = (x * y).sum(-1)
            r = r.abs().pow(self.gamma)
            return _Fx.reduce(r, self.reduction)


def random_flow_field_3d(nps=((3,5),(4,6),(9,14)), strengths=(1e-1, 4e-2, 1e-2), size=(32,32,32), no_field=False, iterations=2, device="cpu", affine=0, window_mode="cos+prod+global"):
    if not isinstance(nps, (list, tuple)): nps = [nps]
    if not isinstance(strengths, (list, tuple)): strengths = [strengths]
    assert len(nps) == len(strengths)
    field = linfield(size, device=device).permute(3,0,1,2)
    w1 = window(*size, mode=window_mode, device=device)
    w2 = window(*size, mode=window_mode.replace("prod",""), device=device)
    
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
            rotate = (_torch.randn(3) * math.pi) * s
            vect = rotate.sin().abs() + rotate.cos().abs()
            translate = _torch.randn(3) * s
            scale = _torch.ones(3)#.mul(vect)
            affines = _torch.tensor([
                [
                    [math.cos(rotate[0]), math.sin(rotate[0]), 0, 0],
                    [-math.sin(rotate[0]), math.cos(rotate[0]), 0, 0],
                    [0,0,1,0], 
                    [0,0,0,1]
                ],
                [
                    [math.cos(rotate[1]), 0, math.sin(rotate[1]), 0],
                    [0,1,0,0], 
                    [-math.sin(rotate[1]), 0, math.cos(rotate[1]),  0],
                    [0,0,0,1]
                ],
                [
                    [1,0,0,0], 
                    [0,math.cos(rotate[2]), math.sin(rotate[2]),  0],
                    [0,-math.sin(rotate[2]), math.cos(rotate[2]), 0],
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
    if True and affine:
        #s = 1
        rotate = (_torch.rand(1).sub(0.5) * math.pi/2)
        s = 1 #(rotate.sin().abs() + rotate.cos().abs())
        
        affines = _torch.tensor([
             [
                    [s,0,0,0], 
                    [0,math.cos(rotate[0])*s, math.sin(rotate[0])*s,  0],
                    [0,-math.sin(rotate[0])*s, math.cos(rotate[0])*s, 0],
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
