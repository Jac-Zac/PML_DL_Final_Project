import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- TimeEmbedding, PatchEmbed, and TransformerBlock classes remain the same ---


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class PatchEmbed(nn.Module):
    """Splits an image into patches and embeds them."""

    def __init__(
        self,
        img_size: int = 28,
        patch_size: int = 4,
        in_channels: int = 1,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, E, P, P)
        x = x.flatten(2)  # (B, E, N)
        x = x.transpose(1, 2)  # (B, N, E)
        return x


class TransformerBlock(nn.Module):
    """A standard Transformer block with self-attention and MLP."""

    def __init__(self, dim: int, n_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out
        return x


class DiffusionUViT(nn.Module):
    """A U-ViT model for diffusion, combining a Vision Transformer with time embeddings."""

    def __init__(
        self,
        img_size: int = 28,
        patch_size: int = 4,
        in_channels: int = 1,
        out_channels: int = 1,
        embed_dim: int = 256,
        depth: int = 6,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        time_emb_dim: int = 128,
    ):
        super().__init__()
        # Store parameters for reshaping later
        self.img_size = img_size
        self.patch_size = patch_size
        self.out_channels = out_channels

        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.n_patches, embed_dim)
        )

        self.time_embed = nn.Sequential(
            TimeEmbedding(embed_dim),
            nn.Linear(embed_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, embed_dim),
        )

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, n_heads, mlp_ratio) for _ in range(depth)]
        )

        self.norm_out = nn.LayerNorm(embed_dim)
        # The final layer projects to the flattened patch dimension
        final_out_channels = patch_size * patch_size * out_channels
        self.final_layer = nn.Linear(embed_dim, final_out_channels)

        # No longer need self.unpatchify_shape

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x) + self.pos_embed

        time_emb = self.time_embed(t)[:, None, :]
        x = x + time_emb

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm_out(x)
        x = self.final_layer(x)  # Shape: (B, N, P*P*C_out)

        # --- FIXED UNPATCHIFY LOGIC ---
        # Reshape for pixel shuffle.
        # 1. Transpose to (B, P*P*C_out, N)
        x = x.transpose(1, 2)

        # 2. Get the grid size (number of patches in height/width)
        patches_h = patches_w = self.img_size // self.patch_size

        # 3. Reshape to (B, C_out * P * P, H_patches, W_patches)
        x = x.view(
            x.shape[0],
            self.out_channels * self.patch_size * self.patch_size,
            patches_h,
            patches_w,
        )

        # 4. Use pixel shuffle to rearrange patches into an image
        # This operation takes (B, C * r^2, H, W) -> (B, C, H * r, W * r)
        # where r is the patch_size
        x = F.pixel_shuffle(x, self.patch_size)

        return x
