import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import ResizeLayer


class MLP(nn.Module):
    def __init__(self, decode_dim):
        super(MLP, self).__init__()
        self.decode_dim = decode_dim
        self.proj = None  # Initialize as None

    def forward(self, x):
        if self.proj is None:
            # Initialize the Linear layer with the correct input dimension
            input_dim = x.shape[-1]
            self.proj = nn.Linear(input_dim, self.decode_dim)
        x = self.proj(x)
        return x


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.9)
        self.relu = nn.ReLU()

    def forward(self, x, training=False):
        x = self.conv(x)
        x = self.bn(x) if training else x
        x = self.relu(x)
        return x


class SegFormerHead(nn.Module):
    def __init__(self, num_mlp_layers=4, decode_dim=768, num_classes=19):
        super(SegFormerHead, self).__init__()
        self.decode_dim = decode_dim
        self.linear_layers = nn.ModuleList([MLP(decode_dim) for _ in range(num_mlp_layers)])
        self.linear_fuse = None # 
        self.dropout = nn.Dropout(0.1)
        self.linear_pred = nn.Conv2d(decode_dim, num_classes, kernel_size=1)

    def forward(self, inputs, training=False):
        H = inputs[0].shape[2]
        W = inputs[0].shape[3]      
        #print([inp.shape for inp in inputs])
        outputs = []

        for x, mlp in zip(inputs, self.linear_layers):
            x = x.permute(0, 2, 3, 1)
            x = mlp(x)
            x = x.permute(0, 3, 1, 2)
            x = ResizeLayer(H, W)(x)
            outputs.append(x)

        x = torch.cat(outputs[::-1], dim=1)
        if self.linear_fuse is None:
            # Initialize the Linear layer with the correct input dimension
            input_dim = x.shape[1]
            self.linear_fuse = ConvModule(in_channels=input_dim, out_channels=self.decode_dim)
        x = self.linear_fuse(x, training=training)
        x = self.dropout(x) if training else x
        x = self.linear_pred(x)

        return x
