import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(self.max_period) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1), value=0.0)
        return emb


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_in = x

        x = self.norm(x).reshape(B, C, H * W)  # [B, C, N]
        qkv = self.qkv(x)  # [B, 3C, N]
        q, k, v = qkv.chunk(3, dim=1)

        def reshape_heads(tensor):
            return tensor.reshape(B, self.num_heads, C // self.num_heads, H * W)

        q, k, v = map(reshape_heads, (q, k, v))
        scale = (C // self.num_heads) ** -0.5
        attn = (q.transpose(2, 3) @ k) * scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v.transpose(2, 3)).transpose(2, 3).reshape(B, C, H * W)

        out = self.proj(out).reshape(B, C, H, W)
        return x_in + out


class DoubleConv(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None
    ):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        groups = min(32, out_channels)
        mid_groups = min(32, mid_channels)

        self.use_residual = in_channels == out_channels

        self.conv1 = weight_norm(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        )
        self.norm1 = nn.GroupNorm(mid_groups, mid_channels)
        self.act1 = nn.SiLU()

        self.conv2 = weight_norm(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        )
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act2 = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
        gamma1: Optional[torch.Tensor] = None,
        beta1: Optional[torch.Tensor] = None,
        gamma2: Optional[torch.Tensor] = None,
        beta2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x

        x = self.conv1(x)
        x = self.norm1(x)
        if gamma1 is not None:
            x = gamma1[:, :, None, None] * x
        if beta1 is not None:
            x = x + beta1[:, :, None, None]
        x = self.act1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        if gamma2 is not None:
            x = gamma2[:, :, None, None] * x
        if beta2 is not None:
            x = x + beta2[:, :, None, None]
        x = self.act2(x)

        if self.use_residual:
            x = x + residual
        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv(x)
        down = self.pool(skip)
        return down, skip


class UpBlock(nn.Module):
    def __init__(
        self, in_channels: int, skip_channels: int, out_channels: int, time_emb_dim: int
    ):
        super().__init__()
        total_in = in_channels + skip_channels
        self.double_conv = DoubleConv(total_in, out_channels)

        self.film_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 4),
        )
        nn.init.zeros_(self.film_mlp[-1].weight)
        nn.init.zeros_(self.film_mlp[-1].bias)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor, time_emb: torch.Tensor
    ) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        film_params = self.film_mlp(time_emb)
        gamma1, beta1, gamma2, beta2 = torch.chunk(film_params, 4, dim=1)
        return self.double_conv(x, gamma1, beta1, gamma2, beta2)


class DiffusionUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_multipliers: List[int] = [1, 2, 4],
        time_emb_dim: int = 128,
        timesteps: int = 1000,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        self.timesteps = timesteps
        self.num_classes = num_classes

        self.time_embed = nn.Sequential(
            TimeEmbedding(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        if num_classes is not None:
            self.class_embed = nn.Embedding(num_classes, time_emb_dim)
        else:
            self.class_embed = nn.Identity()

        self.channels = [base_channels * m for m in channel_multipliers]

        self.input_conv = nn.Conv2d(
            in_channels, self.channels[0], kernel_size=3, padding=1
        )

        self.down_blocks = nn.ModuleList()
        for i in range(len(self.channels) - 1):
            self.down_blocks.append(DownBlock(self.channels[i], self.channels[i + 1]))

        # Bottleneck
        self.bottleneck = nn.Sequential(
            DoubleConv(self.channels[-1], self.channels[-1]),
            AttentionBlock(self.channels[-1]),
        )

        # Decoder
        self.up_blocks = nn.ModuleList()
        reversed_channels = list(reversed(self.channels))
        skip_channels = list(reversed(self.channels[1:]))

        for i in range(len(skip_channels)):
            in_ch = reversed_channels[i]
            skip_ch = skip_channels[i]
            out_ch = reversed_channels[i + 1]
            self.up_blocks.append(UpBlock(in_ch, skip_ch, out_ch, time_emb_dim))

        self.output_conv = nn.Conv2d(self.channels[0], out_channels, kernel_size=1)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        t_emb = self.time_embed(t)
        if self.num_classes is not None and y is not None:
            t_emb = t_emb + self.class_embed(y)

        x = self.input_conv(x)
        skips = []
        for down_block in self.down_blocks:
            x, skip = down_block(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for up_block in self.up_blocks:
            skip = skips.pop()
            x = up_block(x, skip, t_emb)

        return self.output_conv(x)
