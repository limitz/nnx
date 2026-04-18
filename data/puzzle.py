import torch as _torch
import torchvision as _torchvision
from ..graphics.maze import render_maze as _render_maze


class MazeDataset(_torch.utils.data.Dataset):
    def __init__(self, length=1024, size=(16, 16), colorspace="palette"):
        self.length = length
        self.size = size
        self.colorspace = colorspace

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        challenge, target = _render_maze(size=self.size, colorspace=self.colorspace)
        if self.colorspace == "palette":
            challenge = challenge.squeeze(0)
            target = target.squeeze(0)
        return challenge, target


class TeleportDataset(_torch.utils.data.Dataset):
    def __init__(self, dataset=None, size=(40, 80), margin=6):
        if dataset is None:
            dataset = _torchvision.datasets.MNIST(
                root="data", train=True, download=True,
                transform=_torchvision.transforms.ToTensor()
            )
        self.dataset = dataset
        self.size = size
        self.margin = margin

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        r = self.dataset[idx]
        image = r[0] if isinstance(r, (list, tuple)) else r

        C, H, W = image.shape
        canvas_h, canvas_w = self.size
        m = self.margin
        mid_x = canvas_w // 2

        avail_h = canvas_h - 2 * m
        avail_w = mid_x - 2 * m

        if H > avail_h or W > avail_w:
            scale = min(avail_h / H, avail_w / W)
            H, W = int(H * scale), int(W * scale)
            image = _torch.nn.functional.interpolate(
                image.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
            ).squeeze(0)

        y = m + (avail_h - H) // 2
        x_left = m + (avail_w - W) // 2
        x_right = mid_x + m + (avail_w - W) // 2

        source = _torch.zeros(C, canvas_h, canvas_w)
        source[:, y:y + H, x_left:x_left + W] = image

        target = _torch.zeros(C, canvas_h, canvas_w)
        target[:, y:y + H, x_right:x_right + W] = image

        return source, target
