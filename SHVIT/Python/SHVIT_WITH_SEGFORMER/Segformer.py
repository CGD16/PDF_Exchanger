import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict

# from mmseg.ops import resize # steht unten
# from ..builder import HEADS
# from .decode_head import BaseDecodeHead
# from mmseg.models.utils import *
# import attr

from IPython import embed
from SingleHeadVisiontransformer import SHViT




from timm.layers import SqueezeExcite


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768): 
        """
        Args:
            c3: input_dim
            c_out = embed_dim
            F_i = x (computed tensor from SHVIT)
        """


        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x



# embedding_dim = 448 # 768
class SegFormerHead(nn.Module):
    def __init__(self, num_classes=150, embedding_dim = 448, feature_strides=None, channels=None,  in_channels=448,  dropout_ratio=0.1, **kwargs):
        super(SegFormerHead, self).__init__(**kwargs)
        self.in_channels = in_channels
        c1_in_channels = self.in_channels
        self.num_classes = num_classes
        self.feature_strides = feature_strides  

        

        self.dropout_ratio = dropout_ratio
        if self.dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None


        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        self.linear_fuse = ConvModule(
            in_channels=embedding_dim, # 4C --> C --> embedding_dim
            out_channels=embedding_dim,
            kernel_size=1,
            # norm_cfg=dict(type='SyncBN', requires_grad=True)
        )
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)




    def forward(self, inputs):
        _, c_shvit, h_shvit, w_shvit = inputs.shape
        shvit_bibi = SHViT()
        x = shvit_bibi(inputs)#(inputs)#.to(device)
        print("Outputshape of SHViT: ", x.shape)
        print("=======================================================================================")
        print("Inputshape for the Segformer: ", x.shape)

        c1 = x
        n, _, _, _ = c1.shape

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])
        _c1 = resize(_c1, size=(int(h_shvit/4), int(w_shvit/4)),mode='bilinear',align_corners=False)
        _c = self.linear_fuse(torch.cat([_c1.to('cpu')], dim=1))
        
        x = self.dropout(_c)
        x = self.linear_pred(x)
        print("Outputshape of Segformer Head: ", x.shape)
        return x



