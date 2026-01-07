import torch as _torch
import torch.nn.functional as _F
from ..modules import Between as _Between
from .. import functional as _Fx

class RandomNd(_torch.utils.data.IterableDataset):
    def __init__(self, size, channels=1,
                 granularity=_Between(2,9), 
                 threshold=_Between(0.25,0.75),
                 mode=None):
        self.channels = channels
        self.size = size
        self.granularity = granularity
        self.threshold = threshold
        if mode is None:
            if len(self.size) == 3: self.mode = "tricubic"
            elif len(self.size) == 2: self.mode = "bicubic"
            elif len(self.size) == 1: self.mode="linear"
            else: self.mode="nearest"
        else:
            self.mode = mode
        
    def __iter__(self):
        return self
        
    def __next__(self):
        r = _torch.empty((self.channels, *self.size), dtype=_torch.float)
        for i in range(self.channels):
            shape = _Fx.tensor(self.granularity, shape=(len(self.size),))
            threshold = _Fx.tensor(self.threshold)
            src = _torch.rand(1,*shape)
            src = _F.pad(src,(2,2)*len(shape))
            src = _Fx.interpolate(src[None], self.size, mode=self.mode)[0]
            src = src.gt(threshold).float()
            r[[i]] = src
        return r
