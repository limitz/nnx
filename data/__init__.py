import torch as _torch
import torchvision as _torchvision

from .widerface import WIDERface as WIDERface
from .random import RandomNd as RandomNd

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
        
