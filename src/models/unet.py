import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    """Simple sinusoidal time embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class DoubleConv(nn.Module):
    """Improved double convolution with better normalization."""

    def __init__(
        self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None
    ):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        groups = min(8, out_channels) if out_channels >= 8 else out_channels
        mid_groups = min(8, mid_channels) if mid_channels >= 8 else mid_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(mid_groups, mid_channels),
            nn.SiLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class DownBlock(nn.Module):
    """Modular downsampling block."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv(x)
        down = self.pool(skip)
        return down, skip


class UpBlock(nn.Module):
    """Modular upsampling block with time embedding."""

    def __init__(
        self, in_channels: int, skip_channels: int, out_channels: int, time_emb_dim: int
    ):
        super().__init__()
        total_in = in_channels + skip_channels + time_emb_dim
        self.conv = DoubleConv(total_in, out_channels)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor, time_emb: torch.Tensor
    ) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        t_emb = time_emb[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        x = torch.cat([x, skip, t_emb], dim=1)
        return self.conv(x)


class DiffusionUNet(nn.Module):
    """Modular U-Net that works well for MNIST by default but is configurable."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_multipliers: List[int] = [1, 2, 4],  # Fixed for MNIST compatibility
        time_emb_dim: int = 128,
        timesteps: int = 1000,
    ):
        super().__init__()
        self.timesteps = timesteps

        # Time embedding
        self.time_embed = nn.Sequential(
            TimeEmbedding(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Calculate encoder channel sizes
        self.channels = [base_channels * m for m in channel_multipliers]

        # Input projection
        self.input_conv = nn.Conv2d(
            in_channels, self.channels[0], kernel_size=3, padding=1
        )

        # Encoder blocks
        self.down_blocks = nn.ModuleList()
        for i in range(len(self.channels) - 1):
            self.down_blocks.append(DownBlock(self.channels[i], self.channels[i + 1]))

        # Bottleneck
        self.bottleneck = DoubleConv(self.channels[-1], self.channels[-1])

        # Decoder blocks - FIXED: Use actual skip connection channels
        self.up_blocks = nn.ModuleList()
        reversed_channels = list(reversed(self.channels))
        skip_channels = list(
            reversed(self.channels[1:])
        )  # Actual skip connection channels

        for i in range(len(skip_channels)):
            in_ch = reversed_channels[i]
            skip_ch = skip_channels[i]  # Actual channel size from encoder
            out_ch = reversed_channels[i + 1]
            self.up_blocks.append(UpBlock(in_ch, skip_ch, out_ch, time_emb_dim))

        # Output projection
        self.output_conv = nn.Conv2d(self.channels[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
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
