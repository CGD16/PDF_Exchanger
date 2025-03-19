import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import MixVisionTransformer
from .head import SegFormerHead
from .utils import ResizeLayer
# from .shvit import SHViT_S4

MODEL_CONFIGS = {
    'mit_b0': {
        'embed_dims': [32, 64, 160, 256],
        'depths': [2, 2, 2, 2],
        'decode_dim': 256,
    },
    'mit_b1': {
        'embed_dims': [64, 128, 320, 512],
        'depths': [2, 2, 2, 2],
        'decode_dim': 256,
    },
    'mit_b2': {
        'embed_dims': [64, 128, 320, 512],
        'depths': [3, 4, 6, 3],
        'decode_dim': 768,
    },
    'mit_b3': {
        'embed_dims': [64, 128, 320, 512],
        'depths': [3, 4, 18, 3],
        'decode_dim': 768,
    },
    'mit_b4': {
        'embed_dims': [64, 128, 320, 512],
        'depths': [3, 8, 27, 3],
        'decode_dim': 768,
    },
    'mit_b5': {
        'embed_dims': [64, 128, 320, 512],
        'depths': [3, 6, 40, 3],
        'decode_dim': 768,
    },
    'mit_shvit_B0': {
        'decode_dim': 256,
    },
    'mit_shvit_B2': {
        'decode_dim': 768,
    },
}

class SegFormer_B0(nn.Module):
    def __init__(self, input_shape, num_classes=7, num_stages=None, flgUseResize=True):
        super(SegFormer_B0, self).__init__()
        self.flgUseResize = flgUseResize
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.mix_vision_transformer = MixVisionTransformer(
            img_size=input_shape[1], img_channels=input_shape[0],
            embed_dims=MODEL_CONFIGS['mit_b0']['embed_dims'],
            depths=MODEL_CONFIGS['mit_b0']['depths'],
        )

        self.seg_former_head = SegFormerHead(
            num_classes=num_classes,
            decode_dim=MODEL_CONFIGS['mit_b0']['decode_dim'],
        )

        self.resize_layer = ResizeLayer(input_shape[1], input_shape[2])

    def forward(self, x, training=False):
        x = self.mix_vision_transformer(x, training=training)
        x = self.seg_former_head(x, training=training)
        if (self.flgUseResize):
            x = self.resize_layer(x)
        x = F.softmax(x, dim=1).to(torch.float32)  # Softmax and cast to float32
        return x