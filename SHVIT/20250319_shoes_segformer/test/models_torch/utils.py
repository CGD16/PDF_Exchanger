import torch
import torch.nn as nn
import torch.nn.functional as F

class ResizeLayer(nn.Module):
    def __init__(self, height, width):
        super(ResizeLayer, self).__init__()
        self.height = height
        self.width = width

    def forward(self, inputs):
        resized = F.interpolate(inputs, size=(self.height, self.width), mode='bilinear', align_corners=False)
        return resized

class DropPath(nn.Module):
    def __init__(self, drop_path):
        super(DropPath, self).__init__()
        self.drop_path = drop_path

    def forward(self, x, training=False):
        if training:
            shape = (x.size(0),) + (1,) * (len(x.size()) - 1)
            keep_prob = 1 - self.drop_path
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()  # In-place floor operation
            x = (x / keep_prob) * random_tensor
        return x