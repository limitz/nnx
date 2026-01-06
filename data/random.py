import torch as _torch
from ..modules import Between as _Between

class RandomNd(_torch.utils.data.IterableDataset):
    def __init__(self, size, 
                 granularity=_Between(2,9), 
                 threshold=_Between(0.25,0.75)):
        self.size = size
        self.min_cells = min_cells
        self.max_cells = max_cells

    def __iter__(self):
        return self
        
    def __next__(self):
        r,c = torch.randint(self.min_cells,self.max_cells+1,(len(self.size),))
        threshold = torch.rand(()) * 0.5 + 0.25
        src = torch.rand(1,1,r.item(),c.item())
        src = F.pad(src,(2,2,2,2))
        src = F.interpolate(src, self.size, mode="bicubic").gt(threshold).long()[0]
        return src.float(), src.float()
