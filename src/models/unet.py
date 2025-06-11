import math

import torch
import torch.nn as nn


def sinusoidal_time_embedding(timesteps, dim):
    device = timesteps.device
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)


def get_num_groups(channels):
    # Pick number of groups that divides channels
    for g in [32, 16, 8, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )
        self.block1 = nn.Sequential(
            nn.GroupNorm(get_num_groups(in_channels), in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(get_num_groups(out_channels), out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.residual = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, t_emb):
        h = self.block1(x)
        time_emb = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + time_emb
        h = self.block2(h)
        return h + self.residual(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.resblock = ResidualBlock(in_channels, out_channels, time_emb_dim)
        self.down = nn.Conv2d(
            out_channels, out_channels, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x, t_emb):
        x = self.resblock(x, t_emb)
        return self.down(x), x  # output, skip


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, time_emb_dim):
        super().__init__()
        self.resblock = ResidualBlock(
            in_channels + skip_channels, out_channels, time_emb_dim
        )
        self.up = nn.ConvTranspose2d(
            out_channels, out_channels, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x, skip, t_emb):
        # If skip and x spatial dimensions don't match, interpolate
        if skip.shape[2:] != x.shape[2:]:
            skip = torch.nn.functional.interpolate(
                skip, size=x.shape[2:], mode="nearest"
            )
        x = torch.cat([x, skip], dim=1)
        x = self.resblock(x, t_emb)
        return self.up(x)


class DiffusionUNet(nn.Module):
    def __init__(
        self, in_channels=1, out_channels=1, time_emb_dim=256, base_channels=64
    ):
        super().__init__()
        self.time_embedding = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.down1 = DownBlock(base_channels, base_channels * 2, time_emb_dim)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, time_emb_dim)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8, time_emb_dim)

        self.bot = ResidualBlock(base_channels * 8, base_channels * 8, time_emb_dim)

        self.up3 = UpBlock(
            base_channels * 8, base_channels * 8, base_channels * 4, time_emb_dim
        )
        self.up2 = UpBlock(
            base_channels * 4, base_channels * 4, base_channels * 2, time_emb_dim
        )
        self.up1 = UpBlock(
            base_channels * 2, base_channels * 2, base_channels, time_emb_dim
        )

        self.out_conv = nn.Sequential(
            nn.GroupNorm(get_num_groups(base_channels), base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, t):
        x_input = x  # Save for final shape
        t_emb = sinusoidal_time_embedding(t, self.time_embedding[0].in_features)
        t_emb = self.time_embedding(t_emb)

        x = self.init_conv(x)

        x, skip1 = self.down1(x, t_emb)
        x, skip2 = self.down2(x, t_emb)
        x, skip3 = self.down3(x, t_emb)

        x = self.bot(x, t_emb)

        x = self.up3(x, skip3, t_emb)
        x = self.up2(x, skip2, t_emb)
        x = self.up1(x, skip1, t_emb)

        x = self.out_conv(x)

        if x.shape[2:] != x_input.shape[2:]:
            x = nn.functional.interpolate(
                x, size=x_input.shape[2:], mode="bilinear", align_corners=False
            )

        return x
