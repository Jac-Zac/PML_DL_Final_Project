import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding."""

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding


class ResBlock(nn.Module):
    """Residual block with FiLM conditioning and optional attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        use_attention: bool = False,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention

        # Ensure we can create proper group norms
        groups_in = min(32, in_channels) if in_channels >= 32 else in_channels
        groups_out = min(32, out_channels) if out_channels >= 32 else out_channels

        # First conv path
        self.norm1 = nn.GroupNorm(groups_in, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        # Time embedding projection for FiLM
        self.time_proj = nn.Sequential(
            nn.SiLU(), nn.Linear(time_emb_dim, out_channels * 2)  # gamma and beta
        )

        # Second conv path
        self.norm2 = nn.GroupNorm(groups_out, out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # Skip connection
        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_connection = nn.Identity()

        # Optional attention
        if use_attention:
            self.attention = AttentionBlock(out_channels, num_heads)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        residual = x

        # First conv
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        # FiLM conditioning
        time_out = self.time_proj(time_emb)
        gamma, beta = torch.chunk(time_out, 2, dim=1)
        h = gamma[:, :, None, None] * h + beta[:, :, None, None]

        # Second conv
        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)

        # Skip connection
        out = h + self.skip_connection(residual)

        # Optional attention
        if self.use_attention:
            out = self.attention(out)

        return out


class AttentionBlock(nn.Module):
    """Multi-head self-attention block."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        groups = min(32, channels) if channels >= 32 else channels
        self.norm = nn.GroupNorm(groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        residual = x

        x = self.norm(x)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape for multi-head attention
        q = q.view(b * self.num_heads, self.head_dim, h * w)
        k = k.view(b * self.num_heads, self.head_dim, h * w)
        v = v.view(b * self.num_heads, self.head_dim, h * w)

        # Attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.bmm(q.transpose(1, 2), k) * scale
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(attn, v.transpose(1, 2))
        out = out.transpose(1, 2).contiguous().view(b, c, h, w)

        out = self.proj_out(out)
        return residual + out


class DownsampleBlock(nn.Module):
    """Downsampling block with residual connections."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_res_blocks: int = 2,
        use_attention: bool = False,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.res_blocks = nn.ModuleList()
        channels = in_channels

        for i in range(num_res_blocks):
            self.res_blocks.append(
                ResBlock(
                    channels,
                    out_channels,
                    time_emb_dim,
                    use_attention=use_attention,
                    num_heads=num_heads,
                    dropout=dropout,
                )
            )
            channels = out_channels

        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

    def forward(
        self, x: torch.Tensor, time_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for res_block in self.res_blocks:
            x = res_block(x, time_emb)

        skip = x
        x = self.downsample(x)
        return x, skip


class UpsampleBlock(nn.Module):
    """Upsampling block with skip connections."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_res_blocks: int = 2,
        use_attention: bool = False,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )

        self.res_blocks = nn.ModuleList()

        # First block takes upsampled + skip
        first_in_channels = out_channels + skip_channels
        self.res_blocks.append(
            ResBlock(
                first_in_channels,
                out_channels,
                time_emb_dim,
                use_attention=use_attention,
                num_heads=num_heads,
                dropout=dropout,
            )
        )

        # Remaining blocks
        for i in range(num_res_blocks - 1):
            self.res_blocks.append(
                ResBlock(
                    out_channels,
                    out_channels,
                    time_emb_dim,
                    use_attention=use_attention,
                    num_heads=num_heads,
                    dropout=dropout,
                )
            )

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor, time_emb: torch.Tensor
    ) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)

        for res_block in self.res_blocks:
            x = res_block(x, time_emb)

        return x


class UnifiedDiffusionUNet(nn.Module):
    """
    Unified U-Net for diffusion models with classifier-free guidance support.

    Combines the best features from both implementations:
    - FiLM conditioning for better time/class integration
    - Classifier-free guidance support
    - Modular architecture with attention
    - Flexible channel multipliers
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_multipliers: List[int] = [1, 2, 4, 8],
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = [16, 8],
        num_heads: int = 4,
        time_emb_dim: Optional[int] = None,
        num_classes: Optional[int] = None,
        class_dropout_prob: float = 0.1,
        dropout: float = 0.0,
        max_period: int = 10000,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.class_dropout_prob = class_dropout_prob

        # Time embedding dimension
        if time_emb_dim is None:
            time_emb_dim = base_channels * 4

        # Time embedding
        self.time_embed = nn.Sequential(
            TimeEmbedding(base_channels, max_period),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Class embedding for classifier-free guidance
        if num_classes is not None:
            # +1 for null/unconditional class
            self.class_embed = nn.Embedding(num_classes + 1, time_emb_dim)

        # Calculate channel sizes
        self.channels = [base_channels * mult for mult in channel_multipliers]

        # Input projection
        self.input_conv = nn.Conv2d(in_channels, self.channels[0], 3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        for i in range(len(self.channels) - 1):
            # Check if this resolution should have attention
            # Assuming input resolution and tracking downsampling
            use_attention = (64 // (2**i)) in attention_resolutions

            self.down_blocks.append(
                DownsampleBlock(
                    self.channels[i],
                    self.channels[i + 1],
                    time_emb_dim,
                    num_res_blocks,
                    use_attention,
                    num_heads,
                    dropout,
                )
            )

        # Bottleneck
        self.bottleneck = ResBlock(
            self.channels[-1],
            self.channels[-1],
            time_emb_dim,
            use_attention=True,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Decoder
        self.up_blocks = nn.ModuleList()
        reversed_channels = list(reversed(self.channels))

        for i in range(len(reversed_channels) - 1):
            in_ch = reversed_channels[i]
            skip_ch = reversed_channels[i + 1]
            out_ch = reversed_channels[i + 1]

            # Check if this resolution should have attention
            use_attention = (
                64 // (2 ** (len(reversed_channels) - 2 - i))
            ) in attention_resolutions

            self.up_blocks.append(
                UpsampleBlock(
                    in_ch,
                    skip_ch,
                    out_ch,
                    time_emb_dim,
                    num_res_blocks,
                    use_attention,
                    num_heads,
                    dropout,
                )
            )

        # Output
        groups = (
            min(32, self.channels[0]) if self.channels[0] >= 32 else self.channels[0]
        )
        self.output = nn.Sequential(
            nn.GroupNorm(groups, self.channels[0]),
            nn.SiLU(),
            nn.Conv2d(self.channels[0], out_channels, 3, padding=1),
        )

    def forward(
        self, x: torch.Tensor, timesteps: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, C, H, W]
            timesteps: Timestep tensor [B]
            y: Optional class labels [B]
        """
        # Time embedding
        t_emb = self.time_embed(timesteps)

        # Class embedding for classifier-free guidance
        if self.num_classes is not None and y is not None:
            if self.training:
                # During training, randomly drop classes for CFG
                drop_mask = (
                    torch.rand(y.shape[0], device=y.device) < self.class_dropout_prob
                )
                y = torch.where(drop_mask, self.num_classes, y)  # null class

            class_emb = self.class_embed(y)
            t_emb = t_emb + class_emb

        # Input
        h = self.input_conv(x)

        # Encoder
        skips = []
        for down_block in self.down_blocks:
            h, skip = down_block(h, t_emb)
            skips.append(skip)

        # Bottleneck
        h = self.bottleneck(h, t_emb)

        # Decoder
        for up_block in self.up_blocks:
            skip = skips.pop()
            h = up_block(h, skip, t_emb)

        # Output
        return self.output(h)

    @torch.no_grad()
    def sample_with_cfg(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        guidance_scale: float = 7.5,
    ) -> torch.Tensor:
        """
        Sample with classifier-free guidance.

        Args:
            x: Noisy input tensor [B, C, H, W]
            timesteps: Timestep tensor [B]
            y: Class labels [B] (optional)
            guidance_scale: Guidance strength (higher = more guidance)
        """
        if self.num_classes is None or y is None:
            # No guidance, regular sampling
            return self(x, timesteps, y)

        # Duplicate inputs for conditional and unconditional predictions
        x_combined = torch.cat([x, x], dim=0)
        t_combined = torch.cat([timesteps, timesteps], dim=0)

        # Create conditional and unconditional labels
        y_cond = y
        y_uncond = torch.full_like(y, self.num_classes)  # null class
        y_combined = torch.cat([y_cond, y_uncond], dim=0)

        # Forward pass
        self.eval()
        eps_combined = self(x_combined, t_combined, y_combined)

        # Split predictions
        eps_cond, eps_uncond = eps_combined.chunk(2, dim=0)

        # Apply classifier-free guidance
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        return eps

    def get_num_parameters(self, only_trainable: bool = True) -> int:
        """Get total number of parameters."""
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
