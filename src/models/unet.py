import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Sinusoidal Time Embedding ---
class TimeEmbedding(nn.Module):
    """
    Creates sinusoidal embeddings for discrete timesteps (used in diffusion models).
    This is similar to positional encoding in transformers, encoding time step t as sinusoids.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim  # Dimension of the output embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # Compute sinusoidal positional embedding for timesteps t
        half_dim = self.dim // 2
        # Compute scale factor for frequencies
        scale = math.log(10000) / (half_dim - 1)
        # Generate frequency terms exponentially spaced
        freqs = torch.exp(torch.arange(half_dim, device=t.device) * -scale)
        # Shape (B, half_dim), multiply each timestep by frequencies
        emb = t[:, None] * freqs[None, :]
        # Concatenate sin and cos embeddings (shape B x dim)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb  # Shape: (batch_size, dim)


class DoubleConv(nn.Module):
    """
    Double convolutional block with GroupNorm and SiLU activation.
    Supports FiLM conditioning by applying gamma (scale) and beta (shift) parameters after normalization.
    """

    def __init__(
        self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None
    ):
        super().__init__()
        # If mid_channels not provided, use out_channels
        if mid_channels is None:
            mid_channels = out_channels

        # Set number of groups for GroupNorm, min(8, channels)
        groups = min(8, out_channels) if out_channels >= 8 else out_channels
        mid_groups = min(8, mid_channels) if mid_channels >= 8 else mid_channels

        # First conv block
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(mid_groups, mid_channels)
        self.act1 = nn.SiLU()

        # Second conv block
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
        # First conv + norm
        x = self.conv1(x)
        x = self.norm1(x)
        # Apply FiLM conditioning if provided
        if gamma1 is not None and beta1 is not None:
            # gamma and beta shape (B, C), unsqueeze to (B, C, 1, 1) for broadcast
            x = gamma1[:, :, None, None] * x + beta1[:, :, None, None]
        x = self.act1(x)

        # Second conv + norm
        x = self.conv2(x)
        x = self.norm2(x)
        # Apply FiLM conditioning if provided
        if gamma2 is not None and beta2 is not None:
            x = gamma2[:, :, None, None] * x + beta2[:, :, None, None]
        x = self.act2(x)

        return x


class DownBlock(nn.Module):
    """
    Downsampling block: applies double conv, then max-pooling.
    Returns both downsampled output and skip connection.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)  # Downsamples by factor of 2

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv(x)  # Save for skip connection in U-Net
        down = self.pool(skip)  # Downsample spatially
        return down, skip


class UpBlock(nn.Module):
    """
    Upsampling block with FiLM conditioning from time embedding.
    Upsamples input, concatenates skip connection, applies FiLM-conditioned double conv.
    """

    def __init__(
        self, in_channels: int, skip_channels: int, out_channels: int, time_emb_dim: int
    ):
        super().__init__()
        total_in = in_channels + skip_channels  # channels after concat
        self.double_conv = DoubleConv(total_in, out_channels)

        # MLP to predict FiLM parameters gamma and beta for both conv layers
        # Each of the 2 conv layers requires gamma and beta => 4 * out_channels
        self.film_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 4),
        )

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor, time_emb: torch.Tensor
    ) -> torch.Tensor:
        # Upsample x to spatial size of skip connection
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        # Concatenate skip features along channel dim
        x = torch.cat([x, skip], dim=1)

        # Get FiLM params from time embedding
        film_params = self.film_mlp(time_emb)  # shape (B, out_channels*4)
        # Split into gamma1, beta1, gamma2, beta2 each of shape (B, out_channels)
        gamma1, beta1, gamma2, beta2 = torch.chunk(film_params, 4, dim=1)

        # Apply double conv with FiLM conditioning
        return self.double_conv(x, gamma1, beta1, gamma2, beta2)


class DiffusionUNet(nn.Module):
    """
    U-Net architecture designed for diffusion models.
    Supports optional class conditioning and FiLM conditioning via time embeddings.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_multipliers: List[int] = [1, 2, 4],
        time_emb_dim: int = 128,
        timesteps: int = 1000,
        num_classes: Optional[int] = None,  # Optional class conditioning
    ):
        super().__init__()
        self.timesteps = timesteps
        self.num_classes = num_classes

        # Time embedding: map discrete timestep to high-dimensional embedding
        self.time_embed = nn.Sequential(
            TimeEmbedding(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Class embedding if using conditional diffusion
        if num_classes is not None:
            self.class_embed = nn.Embedding(num_classes, time_emb_dim)

        # Compute channels per layer based on multipliers
        self.channels = [base_channels * m for m in channel_multipliers]

        # Initial conv layer to project input to base channel size
        self.input_conv = nn.Conv2d(
            in_channels, self.channels[0], kernel_size=3, padding=1
        )

        # Create encoder blocks (DownBlocks)
        self.down_blocks = nn.ModuleList()
        for i in range(len(self.channels) - 1):
            self.down_blocks.append(DownBlock(self.channels[i], self.channels[i + 1]))

        # Bottleneck double conv at the bottom of the U-Net
        self.bottleneck = DoubleConv(self.channels[-1], self.channels[-1])

        # Create decoder blocks (UpBlocks) with FiLM conditioning
        self.up_blocks = nn.ModuleList()
        reversed_channels = list(reversed(self.channels))
        skip_channels = list(reversed(self.channels[1:]))

        for i in range(len(skip_channels)):
            in_ch = reversed_channels[i]
            skip_ch = skip_channels[i]
            out_ch = reversed_channels[i + 1]
            self.up_blocks.append(UpBlock(in_ch, skip_ch, out_ch, time_emb_dim))

        # Final output conv layer (1x1 conv to reduce to desired output channels)
        self.output_conv = nn.Conv2d(self.channels[0], out_channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,  # Optional class label conditioning
    ) -> torch.Tensor:
        # Get time embedding
        t_emb = self.time_embed(t)

        # Add class embedding if available (class conditional)
        if self.num_classes is not None and y is not None:
            y_emb = self.class_embed(y)
            # Combine time and class embeddings via addition (broadcastable)
            t_emb = t_emb + y_emb

        # Encoder path with skip connections
        x = self.input_conv(x)
        skips = []
        for down_block in self.down_blocks:
            x, skip = down_block(x)
            skips.append(skip)

        # Bottleneck conv
        x = self.bottleneck(x)

        # Decoder path: upsample and use FiLM conditioning
        for up_block in self.up_blocks:
            skip = skips.pop()  # Retrieve skip connection in reverse order
            x = up_block(x, skip, t_emb)

        # Final output projection
        return self.output_conv(x)
