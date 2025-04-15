import torch
from torch import sqrt
import torch.nn as nn

from models_torch_with_sw.attention import Attention
from models_torch_with_sw.utils import DropPath
from torch.nn.init import trunc_normal_



class DWConv(nn.Module):
    """
    2D Depth-Wise Convolution Layer.

    Args:
        hidden_features (int, optional): The number of hidden features/channels. Defaults to 768.

    Inputs:
        x (torch.Tensor): Input tensor of shape (B, N, C), where B is the batch size, 
                          N is the sequence length, and C is the number of features.
        H (int): Height dimension of the input tensor.
        W (int): Width dimension of the input tensor.

    Outputs:
        torch.Tensor: Output tensor of shape (B, H*W, C) after applying the depth-wise convolution.
    """
    def __init__(self, hidden_features: int=768, sw_flag: bool=False):
        super(DWConv, self).__init__()
        self.sw_flag = sw_flag
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        if self.sw_flag == True: 
            B, N, C = x.shape
            x = x.transpose(1, 2).view(B, C, H, W)
            x = self.dwconv(x)
            x = x.flatten(2).transpose(1, 2)
        else:
            x = x.view(x.size(0), H, W, x.size(-1)).permute(0, 3, 1, 2)  # B, C, H, W
            x = self.dwconv(x)
            x = x.flatten(2).permute(0, 2, 1)  # B, H*W, C
        return x


# ===================================================================================================
# https://github.com/ucasligang/SimViT/blob/main/classification/simvit.py # Sliding Window
class CenterAttention(nn.Module):
    """
    """
    # self.attn = CenterAttention(
    #     dim=dim,
    #     num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
    #     attn_drop=attn_drop, proj_drop=drop)
    def __init__(self,
                 dim,
                 num_heads=1,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 stride=1,
                 padding=True,
                 kernel_size=3):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.k_size = kernel_size  # kernel size
        self.stride = stride  # stride
        # self.pat_size = patch_size  # patch size

        self.in_channels = dim  # origin channel is 3, patch channel is in_channel
        self.num_heads = num_heads
        self.head_channel = dim // num_heads
        # self.dim = dim # patch embedding dim
        # it seems that padding must be true to make unfolded dim matchs query dim h*w*ks*ks
        self.pad_size = kernel_size // 2 if padding is True else 0  # padding size
        self.pad = nn.ZeroPad2d(self.pad_size)  # padding around the input
        self.scale = qk_scale or (dim // num_heads)**-0.5
        self.unfold = nn.Unfold(kernel_size=self.k_size, stride=self.stride, padding=0, dilation=1)

        self.qkv_bias = qkv_bias
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # print("1) Centerattention (input): ", x.shape) # [1, 16384, 72]
        B, N, C = x.shape
        x = x.reshape(B, H, W, C)
        # print("2) Centerattention (x.shape): ", x.shape) # [1, 128, 128, 72]
        assert C == self.in_channels

        self.pat_size_h = (H+2 * self.pad_size-self.k_size) // self.stride+1
        self.pat_size_w = (W+2 * self.pad_size-self.k_size) // self.stride+1
        self.num_patch = self.pat_size_h * self.pat_size_w
        # print("3) Centerattention (patch size): ", self.pat_size_h, self.pat_size_w, self.num_patch, self.pad_size, self.num_patch, self.pad_size, self.k_size, self.stride)
        # 128 128 16384 1 16384 1 3 1

        print("================================== Noch ok!! ==================================") 

        # (B, NumHeads, H, W, HeadC)
        q = self.q_proj(x).reshape(B, H, W, self.num_heads, self.head_channel).permute(0, 3, 1, 2, 4)
        print("================================== ", q.shape)
        # q = self.pad(q).permute(0, 1, 3, 4, 2)  # (B, NumH, H, W, HeadC)
        # query need to be copied by (self.k_size*self.k_size) times
        q = q.unsqueeze(dim=4)
        q = q * self.scale
        # if stride is not 1, q should be masked to match ks*ks*patch
        # ...

        # (2, B, NumHeads, HeadsC, H, W)
        kv = self.kv_proj(x).reshape(B, H, W, 2, self.num_heads, self.head_channel).permute(3, 0, 4, 5, 1, 2)

        kv = self.pad(kv)  # (2, B, NumH, HeadC, H, W)
        kv = kv.permute(0, 1, 2, 4, 5, 3)
        H, W = H + self.pad_size * 2, W + self.pad_size * 2

        # unfold plays role of conv2d to get patch data
        # print("4) kv (bbbefore): ", kv.shape) # [2, 1, 1, 130, 130, 72]
        kv = kv.permute(0, 1, 2, 5, 3, 4).reshape(2 * B, -1, H, W)
        # print("5) kv (before): ", kv.shape) # [2, 72, 130, 130]
        kv = self.unfold(kv)
        # print("6) kv (after): ", kv.shape) # [2, 648, 16384]
        kv = kv.reshape(2, B, self.num_heads, self.head_channel, self.k_size**2,
                        self.num_patch)  # (2, B, NumH, HC, ks*ks, NumPatch)
        kv = kv.permute(0, 1, 2, 5, 4, 3)  # (2, B, NumH, NumPatch, ks*ks, HC)
        # print("7) kv (aaafter): ", kv.shape) # [2, 1, 1, 16384, 9, 72]
        k, v = kv[0], kv[1]

        # (B, NumH, NumPatch, 1, HeadC)
        q = q.reshape(B, self.num_heads, self.num_patch, 1, self.head_channel)
        # print("8) k, q, v: ", k.shape, q.shape, v.shape) # torch.Size([1, 1, 16384, 9, 72]) torch.Size([1, 1, 16384, 1, 72]) torch.Size([1, 1, 16384, 9, 72])
        # print("9) k.transpose(-2, -1): ", k.transpose(-2, -1).shape) # [1, 1, 16384, 72, 9]
        attn = (q @ k.transpose(-2, -1))  # (B, NumH, NumPatch, ks*ks, ks*ks)
        # print("10) attn: ", attn.shape) # [1, 1, 16384, 1, 9]
        attn = self.softmax(attn)  # softmax last dim
        attn = self.attn_drop(attn)

        out = (attn @ v).squeeze(3)  # (B, NumH, NumPatch, HeadC)
        # print("11) out: ", out.shape) # [1, 1, 16384, 72]
        out = out.permute(0, 2, 1, 3).reshape(B, self.pat_size_h, self.pat_size_w, C)  # (B, Ph, Pw, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        # print("12) out (before reshape): ", out.shape) # [1, 128, 128, 72]
        out = out.reshape(B, -1, C)
        # print("13) out (after reshape): ", out.shape) # [1, 16384, 72]
        # print()
        # print("========================"*5)
        # print()
        return out

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.trunc_normal_
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ===================================================================================================




class Mlp(nn.Module):
    """
    Multi-Layer Perceptron (MLP) with Depth-Wise Convolution.

    Args:
        in_features (int): The number of input features.
        hidden_features (int, optional): The number of hidden features. Defaults to in_features.
        out_features (int, optional): The number of output features. Defaults to in_features.
        drop (float, optional): Dropout rate. Defaults to 0.0.

    Inputs:
        x (torch.Tensor): Input tensor of shape (B, N, C), where B is the batch size, 
                          N is the sequence length, and C is the number of input features.
        H (int): Height dimension of the input tensor.
        W (int): Width dimension of the input tensor.

    Outputs:
        torch.Tensor: Output tensor of shape (B, N, out_features) after linear projection, 
                      depth-wise convolution, activation, and dropout.
    """
    def __init__(self, in_features: int, hidden_features: int=None, out_features: int=None, drop: float=0.0):
        super(Mlp, self).__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """
    2D Transformer Block with multi-head self-attention and MLP.

    Args:
        dim (int): The number of input dimensions.
        num_heads (int): The number of attention heads.
        mlp_ratio (float, optional): Ratio of hidden dimension to input dimension for the MLP. Defaults to 4.0.
        qkv_bias (bool, optional): Whether to include bias in query, key, and value projections. Defaults to False.
        drop (float, optional): Dropout rate. Defaults to 0.0.
        attn_drop (float, optional): Dropout rate on attention weights. Defaults to 0.0.
        drop_path (float, optional): Dropout rate on the residual path. Defaults to 0.0.
        sr_ratio (int, optional): Spatial reduction ratio for the attention layer. Defaults to 1.

    Inputs:
        x (torch.Tensor): Input tensor of shape (B, N, C), where B is the batch size, 
                          N is the sequence length, and C is the number of input dimensions.
        H (int): Height dimension of the input tensor.
        W (int): Width dimension of the input tensor.

    Outputs:
        torch.Tensor: Output tensor of shape (B, N, C) after attention and MLP operations.
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float=4.0, qkv_bias: bool=False, drop: float=0.0, 
                 attn_drop: float=0.0, drop_path: float=0.0, sr_ratio: int=1):
        super(Block, self).__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn = Attention(dim, num_heads, sr_ratio, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    """
    2D Patch Embedding with overlapping patches.

    Args:
        img_size (int, optional): Size of the input image. Defaults to 224.
        img_channels (int, optional): Number of channels in the input image. Defaults to 3.
        patch_size (int, optional): Size of the patch. Defaults to 7.
        stride (int, optional): Stride of the convolution. Defaults to 4.
        filters (int, optional): Number of output filters. Defaults to 768.

    Inputs:
        x (torch.Tensor): Input tensor of shape (B, img_channels, H, W), where B is the batch size, 
                          img_channels is the number of input channels,
                          H is the height, and W is the width.

    Outputs:
        torch.Tensor: Output tensor of shape (B, N, filters), where N is the reshaped sequence length.
        int: Height dimension of the output tensor.
        int: Width dimension of the output tensor.
    """
    def __init__(self, img_size: int=224, img_channels: int=3, patch_size: int=7, stride: int=4, filters: int=768):
        super(OverlapPatchEmbed, self).__init__()
        self.conv = nn.Conv2d(img_channels, filters, kernel_size=patch_size, stride=stride, padding=patch_size // 2)
        self.norm = nn.LayerNorm(filters, eps=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)  # B, H*W, C
        x = self.norm(x)
        return x, H, W


class MixVisionTransformer(nn.Module):
    """
    2D Vision Transformer with mixed patch embeddings and multiple Transformer blocks.

    Args:
        img_size (int, optional): Size of the input image. Defaults to 224.
        img_channels (int, optional): Number of input channels. Defaults to 3.
        embed_dims (list of int, optional): Embedding dimensions for each stage. Defaults to [64, 128, 256, 512].
        num_heads (list of int, optional): Number of attention heads for each stage. Defaults to [1, 2, 4, 8].
        mlp_ratios (list of int, optional): MLP ratios for each stage. Defaults to [4, 4, 4, 4].
        qkv_bias (bool, optional): Whether to use bias in query, key, and value projections. Defaults to False.
        drop_rate (float, optional): Dropout rate. Defaults to 0.0.
        attn_drop_rate (float, optional): Dropout rate for attention layers. Defaults to 0.0.
        drop_path_rate (float, optional): Dropout rate for stochastic depth. Defaults to 0.0.
        depths (list of int, optional): Depth (number of blocks) for each stage. Defaults to [3, 4, 6, 3].
        sr_ratios (list of int, optional): Spatial reduction ratios for each stage. Defaults to [8, 4, 2, 1].

    Inputs:
        x (torch.Tensor): Input tensor of shape (B, img_channels, H, W), where B is the batch size, 
                          img_channels is the number of input channels,
                          H is the height, and W is the width.

    Outputs:
        list of torch.Tensor: List of output tensors from each stage, where each tensor has shape (B, C, H, W).
    """
    def __init__(self, img_size: int=224, img_channels: int=3, embed_dims: list=[64, 128, 256, 512], 
                 num_heads: list=[1, 2, 4, 8], mlp_ratios: list=[4, 4, 4, 4], qkv_bias: bool=False, 
                 drop_rate: float=0.0, attn_drop_rate: float=0.0, drop_path_rate: float=0.0,
                 depths: list=[3, 4, 6, 3], sr_ratios: list=[8, 4, 2, 1]):
        super(MixVisionTransformer, self).__init__()
        self.depths = depths

        patch_sizes = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        img_sizes = [img_size, img_size // 4, img_size // 8, img_size // 16]
        channels = [img_channels] + embed_dims[:-1]

        self.patch_embeds = nn.ModuleList([
            OverlapPatchEmbed(img_size=img_sizes[i], img_channels=channels[i], patch_size=patch_sizes[i],
                              stride=strides[i], filters=embed_dims[i])
            for i in range(len(embed_dims))
        ])

        dpr = torch.linspace(0.0, drop_path_rate, sum(depths)).tolist()

        self.blocks = nn.ModuleList([
            nn.ModuleList([
                Block(dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                      drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[sum(depths[:i]) + j],
                      sr_ratio=sr_ratios[i])
                for j in range(depths[i])
            ])
            for i in range(len(embed_dims))
        ])

        self.norms = nn.ModuleList([nn.LayerNorm(embed_dims[i], eps=1e-5) for i in range(len(embed_dims))])

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        outs = []

        for i, (patch_embed, block, norm) in enumerate(zip(self.patch_embeds, self.blocks, self.norms)):
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x).permute(0, 2, 1).view(B, -1, H, W)
            outs.append(x)

        return outs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)