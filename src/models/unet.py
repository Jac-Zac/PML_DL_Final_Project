import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion-style models."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # Create sinusoidal time embedding
        half_dim = self.dim // 2
        scale = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=t.device) * -scale)
        emb = t[:, None] * freqs[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class TimeMLP(nn.Module):
    """Learned MLP-based time embedding for flow-matching."""

    def __init__(self, time_emb_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(t[:, None])  # Expand shape to [B, 1]


class DoubleConv(nn.Module):
    """Two convolutional layers with optional FiLM conditioning."""

    def __init__(
        self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None
    ):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        # Use group norm with small group size
        groups = min(8, out_channels) if out_channels >= 8 else out_channels
        mid_groups = min(8, mid_channels) if mid_channels >= 8 else mid_channels

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(mid_groups, mid_channels)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
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
        # First conv + optional FiLM
        x = self.conv1(x)
        x = self.norm1(x)
        if gamma1 is not None and beta1 is not None:
            x = gamma1[:, :, None, None] * x + beta1[:, :, None, None]
        x = self.act1(x)

        # Second conv + optional FiLM
        x = self.conv2(x)
        x = self.norm2(x)
        if gamma2 is not None and beta2 is not None:
            x = gamma2[:, :, None, None] * x + beta2[:, :, None, None]
        return self.act2(x)


class DownBlock(nn.Module):
    """Downsampling block with FiLM conditioning."""

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.use_film = time_emb_dim is not None
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

        if self.use_film:
            self.film_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels * 4),
            )

    def forward(self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None):
        # Apply FiLM if available
        if self.use_film and time_emb is not None:
            film_params = self.film_mlp(time_emb)
            gamma1, beta1, gamma2, beta2 = torch.chunk(film_params, 4, dim=1)
            skip = self.conv(x, gamma1, beta1, gamma2, beta2)
        else:
            skip = self.conv(x)

        down = self.pool(skip)
        # Return both for skip connections
        return down, skip


class Bottleneck(nn.Module):
    """U-Net bottleneck with FiLM conditioning."""

    def __init__(self, channels: int, time_emb_dim: int):
        super().__init__()
        self.use_film = time_emb_dim is not None
        self.conv = DoubleConv(channels, channels)
        if self.use_film:
            self.film_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, channels * 4),
            )

    def forward(self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None):
        if self.use_film and time_emb is not None:
            film_params = self.film_mlp(time_emb)
            gamma1, beta1, gamma2, beta2 = torch.chunk(film_params, 4, dim=1)
            return self.conv(x, gamma1, beta1, gamma2, beta2)
        return self.conv(x)


class UpBlock(nn.Module):
    """Upsampling block with FiLM conditioning and skip connections."""

    def __init__(
        self, in_channels: int, skip_channels: int, out_channels: int, time_emb_dim: int
    ):
        super().__init__()
        self.double_conv = DoubleConv(in_channels + skip_channels, out_channels)
        self.film_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 4),
        )

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor, time_emb: torch.Tensor
    ) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)  # Concatenate skip connection
        film_params = self.film_mlp(time_emb)
        gamma1, beta1, gamma2, beta2 = torch.chunk(film_params, 4, dim=1)
        return self.double_conv(x, gamma1, beta1, gamma2, beta2)


class DiffusionUNet(nn.Module):
    """
    U-Net with FiLM-based time conditioning, usable for diffusion or flow-matching models.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_multipliers: List[int] = [1, 2, 4],
        time_emb_dim: int = 128,
        timesteps: int = 1000,
        num_classes: Optional[int] = None,
        time_embedding_type: str = "sinusoidal",  # "sinusoidal" for diffusion, "mlp" for flow-matching
    ):
        super().__init__()
        self.timesteps = timesteps
        self.num_classes = num_classes

        # Time embedding
        if time_embedding_type == "sinusoidal":
            self.time_embed = nn.Sequential(
                TimeEmbedding(base_channels),
                nn.Linear(base_channels, time_emb_dim),
                nn.SiLU(),
                nn.Linear(time_emb_dim, time_emb_dim),
            )
        elif time_embedding_type == "mlp":
            self.time_embed = TimeMLP(time_emb_dim)
        else:
            raise ValueError(f"Unknown time embedding type: {time_embedding_type}")

        # Optional class conditioning
        if num_classes is not None:
            self.class_embed = nn.Embedding(num_classes, time_emb_dim)

        self.channels = [base_channels * m for m in channel_multipliers]

        self.input_conv = nn.Conv2d(
            in_channels, self.channels[0], kernel_size=3, padding=1
        )

        # Downsampling path
        self.down_blocks = nn.ModuleList()
        for i in range(len(self.channels) - 1):
            self.down_blocks.append(
                DownBlock(self.channels[i], self.channels[i + 1], time_emb_dim)
            )

        self.bottleneck = Bottleneck(self.channels[-1], time_emb_dim)

        # Upsampling path
        reversed_channels = list(reversed(self.channels))
        skip_channels = list(reversed(self.channels[1:]))

        self.up_blocks = nn.ModuleList()
        for i in range(len(skip_channels)):
            in_ch = reversed_channels[i]
            skip_ch = skip_channels[i]
            out_ch = reversed_channels[i + 1]
            self.up_blocks.append(UpBlock(in_ch, skip_ch, out_ch, time_emb_dim))

        self.output_conv = nn.Conv2d(self.channels[0], out_channels, kernel_size=1)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Compute time embedding (and optionally class embedding)
        t_emb = self.time_embed(t)
        if self.num_classes is not None and y is not None:
            t_emb = t_emb + self.class_embed(y)

        x = self.input_conv(x)
        skips = []

        # Down path
        for down_block in self.down_blocks:
            x, skip = down_block(x, t_emb)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x, t_emb)

        # Up path
        for up_block in self.up_blocks:
            skip = skips.pop()
            x = up_block(x, skip, t_emb)

        return self.output_conv(x)
