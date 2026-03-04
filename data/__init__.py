import torch as _torch
import torchvision as _torchvision
import os as _os
import glob as _glob
import tifffile as _tifffile
from .widerface import WIDERface as WIDERface
from .random import RandomNd as RandomNd
from .. import functional as _Fx

class DatasetProxyWithIndex(_torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        r = self.dataset[idx]
        if isinstance(r, (list, tuple)):
            return (*r, idx)
        else:
            return r, idx

class ImageDataset(_torch.utils.data.Dataset):
    def __init__(self, pattern, target_types=None, transform=None):
        assert _Fx.all_in_set(target_types, {"index", "path"}) 
        self.paths = sorted(_glob.glob(pattern, recursive=True))
        self.target_types = target_types or []
        self.transform = transform
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        _, ext = _os.path.splitext(path)
        if ext == ".tiff":
            image = _tifffile.imread(path)
            image = _torch.from_numpy(image).permute(-1,0,1)
            if image.dtype == _torch.uint16:
                image = image / ((1<<16)-1)
            elif image.dtype == _torch.uint8:
                image = image / ((1<<8)-1)
        else:
            image = _torchvision.io.read_image(path) / 255
        if self.transform is not None:
            image = self.transform(image)
        r = [image]
        if "index" in self.target_types:
            r.append(idx)
        if "path" in self.target_types:
            r.append(path)
        return r[0] if len(r) == 1 else tuple(r)
        
class VideoDataset(_torch.utils.data.Dataset):
    def __init__(self, path, start=0, length=10, transform=None):
        self.transform = transform
        self.frames, *_  = _torchvision.io.read_video(path, 
                                                      start_pts=start, 
                                                      end_pts=start+length, 
                                                      pts_unit="sec", 
                                                      output_format="TCHW")[:-1]

    def __len__(self):
        return self.frames.shape[0]

    def __getitem__(self, idx):
        frame = self.frames[idx]/255
        if self.transform is not None:
            frame = self.transform(frame)
        return frame


class ZipDataset(_torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        
    def __len__(self):
        return min(len(d) for d in self.datasets)

    def __getitem__(self, idx):
        return tuple(d[idx] for d in self.datasets)

class CollateWithFallback:
    def __init__(self, fallback="list"):
        assert fallback in { "list" }
        self.fallback = fallback

    def __call__(self, xs):
        assert len(xs) > 0
        assert all(len(x) == len(xs[0]) for x in xs)
        zs = zip(*xs)
        rs = []
        for z in zs:
            if all(type(v) == type(z[0]) for v in z):
                if isinstance(z[0], _torch.Tensor):
                    if all(v.shape == z[0].shape for v in z):
                        rs.append(_torch.stack(z))
                        continue
                elif isinstance(z[0], (int, float, complex, bool)):
                    rs.append(_torch.tensor(list(z)))
                    continue
            
            # fallback  
            if self.fallback == "list":
                rs.append(list(z))
        
        return tuple(rs)

collate_with_list_fallback = default_collate = CollateWithFallback()


class WindowDataset(_torch.utils.data.Dataset):
    def __init__(self, dataset, radius=2, collate_fn=collate_with_list_fallback):
        self.dataset = dataset
        self.radius = radius
        self.collate_fn = collate_fn
        
    def __len__(self):
        return max(0, (len(self.dataset) - 2 * self.radius))

    def __getitem__(self, idx):
        rs = []
        for i in range(idx, idx+2*self.radius+1):
            r = self.dataset[i]  
            rs.append(r)
        
        rs = self.collate_fn(rs)
        return rs
        
class SkippableBatchSampler(_torch.utils.data.Sampler):
    def __init__(self, data_source, batch_size, skip=0):
        self.data_source = data_source
        self.batch_size = batch_size
        self.indices = None
        self.to_skip = skip

    def skip(self, n):
        self.to_skip = n
        
    def __len__(self):
        return (len(self.data_source) // self.batch_size) - self.to_skip
        
    def __iter__(self):
        l = len(self.data_source)
        self.indices = _torch.randperm(l)[:(l//self.batch_size)*self.batch_size]
        self.indices = self.indices.split(self.batch_size,0)[self.to_skip:]
        self.to_skip = 0
        for i in self.indices:
            yield i
        
