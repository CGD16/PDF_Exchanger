import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .attention import Attention
from .utils import DropPath


class DWConv(nn.Module):
    def __init__(self, hidden_features=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features)

    def forward(self, x, H, W):
        x = x.view(x.size(0), H, W, x.size(-1)).permute(0, 3, 1, 2)  # B, C, H, W
        x = self.dwconv(x)
        x = x.flatten(2).permute(0, 2, 1)  # B, H*W, C
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W, training=False):
        x = self.fc1(x)
        x = self.dwconv(x, H=H, W=W)
        x = self.act(x)
        x = self.drop(x) if training else x
        x = self.fc2(x)
        x = self.drop(x) if training else x
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0, drop_path=0.0, sr_ratio=1):
        super(Block, self).__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-05)
        self.attn = Attention(dim, num_heads, sr_ratio, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim, eps=1e-05)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x, H, W, training=False):
        attn_output_norm = self.norm1(x)
        attn_output = self.attn(attn_output_norm, H=H, W=W, training=training)
        attn_output_with_drop = self.drop_path(attn_output, training=training)
        x = x + attn_output_with_drop

        # Apply LayerNormalization and MLP layer
        mlp_output_norm = self.norm2(x)
        mlp_output = self.mlp(mlp_output_norm, H=H, W=W, training=training)
        mlp_output_with_drop = self.drop_path(mlp_output, training=training)
        x = x + mlp_output_with_drop

        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, img_channels=3, patch_size=7, stride=4, filters=768):
        super(OverlapPatchEmbed, self).__init__()
        self.pad = nn.ZeroPad2d(patch_size // 2)
        self.conv = nn.Conv2d(in_channels=img_channels, out_channels=filters, kernel_size=patch_size, stride=stride, padding=0)
        self.norm = nn.LayerNorm(filters, eps=1e-05)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        C, H, W = x.shape[1], x.shape[2], x.shape[3]
        x = x.reshape(-1, H * W, C)  # B, H*W, C
        x = self.norm(x)
        return x, H, W


class MixVisionTransformer(nn.Module):
    def __init__(self, img_size=224, img_channels=3, embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], 
                 qkv_bias=False, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, depths=[3, 4, 6, 3], 
                 sr_ratios=[8, 4, 2, 1]):
        super(MixVisionTransformer, self).__init__()
        self.depths = depths
        
        # Patch embedding configurations
        patch_sizes = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        img_sizes = [img_size, img_size // 4, img_size // 8, img_size // 16]
        channels = [img_channels] + embed_dims[:-1]
        
        self.patch_embeds = nn.ModuleList([
            OverlapPatchEmbed(img_size=img_sizes[i], img_channels=channels[i], patch_size=patch_sizes[i], stride=strides[i], filters=embed_dims[i])
            for i in range(len(embed_dims))
        ])
        
        dpr = [x.item() for x in torch.linspace(0.0, drop_path_rate, sum(depths))]
        
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()
        cur = 0
        
        for i in range(len(embed_dims)):
            block = nn.ModuleList([
                Block(dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, 
                      drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], sr_ratio=sr_ratios[i])
                for j in range(depths[i])
            ])
            self.blocks.append(block)
            self.norms.append(nn.LayerNorm(embed_dims[i], eps=1e-05))
            cur += depths[i]
                    
    def forward_features(self, x, training=False):
        B = x.size(0)
        outs = []

        for i in range(len(self.blocks)):
            x, H, W = self.patch_embeds[i](x)
            for blk in self.blocks[i]:
                x = blk(x, H=H, W=W, training=training)
            x = self.norms[i](x)
            x = x.view(B, H, W, -1).permute(0, 3, 1, 2)
            outs.append(x)

        return outs

    def forward(self, x, training=False):
        x = self.forward_features(x, training)
        return x
