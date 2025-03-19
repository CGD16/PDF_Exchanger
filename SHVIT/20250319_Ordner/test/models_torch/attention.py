import torch
import torch.nn as nn
import math

class Attention(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super(Attention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = self.dim // self.num_heads

        self.units = self.num_heads * self.head_dim
        self.sqrt_of_units = math.sqrt(self.head_dim)

        self.q = nn.Linear(dim, self.units, bias=qkv_bias)
        self.k = nn.Linear(dim, self.units, bias=qkv_bias)
        self.v = nn.Linear(dim, self.units, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)  # epsilon=1e-05 by default
            
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W, training=False):
        B, N, C = x.shape

        q = self.q(x)
        q = q.reshape(B, N, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)  # B C H W
            x = self.sr(x)
            x = x.reshape(B, self.dim, -1).permute(0, 2, 1)  # B N_DIM C
            x = self.norm(x)

        k = self.k(x)
        v = self.v(x)
        
        k = k.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 3, 1)
        v = v.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k)
        attn = attn / self.sqrt_of_units
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn) if training else attn
        
        x = torch.matmul(attn, v)
        x = x.permute(0, 2, 1, 3).reshape(B, N, self.units)
        x = self.proj(x)
        x = self.proj_drop(x) if training else x
        
        return x