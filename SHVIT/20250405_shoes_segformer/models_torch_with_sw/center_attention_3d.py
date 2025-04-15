import torch
from torch import sqrt
import torch.nn as nn

from models_torch_with_sw.attention_3d import Attention3D
from models_torch_with_sw.utils_3d import DropPath3D
from torch.nn.init import trunc_normal_


# ===================================================================================================
# https://github.com/ucasligang/SimViT/blob/main/classification/simvit.py # Sliding Window
class CenterAttention3D(nn.Module):
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
        self.pad = nn.ZeroPad3d(self.pad_size)  # padding around the input
        self.scale = qk_scale or (dim // num_heads)**-0.5
        # self.unfold = nn.Unfold(kernel_size=self.k_size, stride=self.stride, padding=0, dilation=1)

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
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    # Unfold(kernel_size=self.k_size, stride=self.stride, padding=0, dilation=1)
    def unfold3d(self, x: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        kd = kh = kw = kernel_size
        sd = sh = sw = stride
        
        # x = x.unfold(2, kernel_size=kd, stride=sd, padding=0, dilation=1).unfold(3, kernel_size=kh, stride=sh, padding=0, dilation=1).unfold(4, kernel_size=kw, stride=sw, padding=0, dilation=1)
        x = x.unfold(2,kd,sd).unfold(3,kh,sh).unfold(4,kw,sw)
        x = x.contiguous().view(B, C, -1, kd * kh * kw)
        return x.permute(0, 2, 3, 1)  # (B, NumPatch, ks*ks*ks, C)


    def forward(self, x, D, H, W):
        B, N, C = x.shape
        x = x.reshape(B, D, H, W, C)
        assert C == self.in_channels

        self.pat_size_d = (D+2 * self.pad_size-self.k_size) // self.stride+1
        self.pat_size_h = (H+2 * self.pad_size-self.k_size) // self.stride+1
        self.pat_size_w = (W+2 * self.pad_size-self.k_size) // self.stride+1
        self.num_patch = self.pat_size_d * self.pat_size_h * self.pat_size_w

        
        '''
        # (B, NumHeads, D, H, W, HeadC)
        q = self.q_proj(x).reshape(B, D, H, W, self.num_heads, self.head_channel).permute(0, 4, 1, 2, 3, 5) # [1, 1, 16, 16, 16, 72]
        # print("q.shape: ", q.shape) # [1, 1, 16, 16, 16, 72]
        # q = self.pad(q).permute(0, 1, 3, 4, 2)  # (B, NumH, H, W, HeadC)
        # query need to be copied by (self.k_size*self.k_size) times
        q = q.unsqueeze(dim=5)
        q = q * self.scale
        '''
        q = self.q_proj(x).view(B, D, H, W, self.num_heads, self.head_channel).permute(0, 4, 1, 2, 3, 5)  # (B, NumH, D, H, W, HeadC)
        q = q.contiguous().view(B, self.num_heads, -1, self.head_channel)  # (B, NumH, D*H*W, HeadC)
        q = q * self.scale
        q = q.view(B, self.num_heads, self.pat_size_d * self.pat_size_h* self.pat_size_w, 1, self.head_channel)
        # if stride is not 1, q should be masked to match ks*ks*patch
        # ...
        
        # (2, B, NumHeads, HeadsC, D, H, W)
        kv = self.kv_proj(x).reshape(B, D, H, W, 2, self.num_heads, self.head_channel).permute(4,0,5,6,1,2,3) # (3, 0, 4, 5, 1, 2)
        # print("kv.shape (proj): ", kv.shape) # [2, 1, 1, 72, 16, 16, 16] 
        # kv = self.pad(kv)  # (2, B, NumH, HeadC, H, W)
        # print("kv.shape (pad): ", kv.shape)
        
        k, v = kv[0], kv[1]
        k = self.pad(k.squeeze(0))
        v = self.pad(v.squeeze(0))
        
        
    
        # D, H, W = D + self.pad_size * 2, H + self.pad_size * 2, W + self.pad_size * 2
        # unfold plays role of conv3d to get patch data
        # kv = kv.permute(0,1,2,6,3,4,5).reshape(2 * B, -1, D, H, W) 
        # print("5) kv (before): ", k.shape) # [2, 72, 18, 18, 18]
        k = self.unfold3d(k, self.k_size, self.stride)  # (B, NumPatch, ks, HeadC)
        v = self.unfold3d(v, self.k_size, self.stride)
        
        # kv = kv.reshape(2, B, self.num_heads, self.head_channel, self.k_size**2, 
        #                 self.num_patch)  # (2, B, NumH, HC, ks*ks, NumPatch)
        
        k = k.reshape(B, self.num_heads, self.num_patch, self.k_size ** 3, self.head_channel)
        v = v.reshape(B, self.num_heads, self.num_patch, self.k_size ** 3, self.head_channel)

     

        attn = (q @ k.transpose(-2, -1))  # (B, NumH, NumPatch, ks*ks, ks*ks)
        attn = self.softmax(attn)  # softmax last dim
        attn = self.attn_drop(attn)

        out = (attn @ v).squeeze(3)  # (B, NumH, NumPatch, HeadC)
        out = out.permute(0, 2, 1, 3).reshape(B, self.pat_size_d, self.pat_size_h, self.pat_size_w, C)  # (B, Ph, Pw, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        out = out.reshape(B, -1, C)
        return out