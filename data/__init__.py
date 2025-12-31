import torch as _torch

from .widerface import WIDERface as WIDERface

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
        
