import math as _math
import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _F
import torchvision as _torchvision
import torchvision.transforms.v2 as _T
import torchvision.transforms.v2.functional as _TF
import numpy as _np
import collections as _collections
from .. import functional as _Fx

class WIDERface(_torch.utils.data.Dataset):
    def __init__(self, root, split="train", transform=None, target_transform=None):
        assert split in {"train", "eval"}
        self.split = split
        self.category_names = {}
        self.categories = {}
        self.files = []
        self.annotations = {}
        self.transform = transform
        self.target_transform = target_transform
        
        if split == "eval": split = "val"
        gt_path = os.path.join(root, "widerface","wider_face_split")
        gt_path = os.path.join(gt_path, f"wider_face_{split}_bbx_gt.txt")
        self.root = os.path.join(root, "widerface", f"WIDER_{split}", "images")
        with open(gt_path) as stream:
            while True:
                path = stream.readline()[:-1]
                if not path: break
                category,*_ = path.split("/")
                cid, cname = category.split("--",1)
                cname = cname.replace("--","_")
                cid = int(cid)
                self.category_names[cid] = cname
                self.categories[path] = cid
                self.files.append(path)
                self.annotations[path] = []
                num_boxes = int(stream.readline())
                if not num_boxes: stream.readline()
                for _ in range(num_boxes):
                    fields = [int(field) 
                              for field in stream.readline()[:-1].split(" ") 
                              if field]
                    if len(fields) < 4: continue
                    self.annotations[path].append(fields)
                if not len(self.annotations[path]):
                    self.files.remove(path)
                    
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        path = self.files[idx]
        boxes = self.annotations[path]
        boxes = _torch.tensor(boxes)[:,:4]
        image = _torchvision.io.read_image(os.path.join(self.root, path))/255
        boxes[...,[0,2]] /= image.shape[-1]
        boxes[...,[1,3]] /= image.shape[-2]
        mask = _torch.zeros(1,SIZE,SIZE)
        for box in boxes:
            area = box[-2] * box[-3]
            Fx.crop_view(mask, box)[:] = area ** -0.5 / len(boxes)
        
        #cid = self.categories[path]
        
        rng = _torch.get_rng_state()
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            _torch.set_rng_state(rng)
            mask = self.target_transform(mask)
        return image, mask

    @staticmethod
    def collate(data):
        images = [v[0] for v in data]
        targets = _torch.stack([v[1] for v in data])
        return images, targets
        
