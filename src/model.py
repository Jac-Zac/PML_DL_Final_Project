from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class LightweightUNet(nn.Module):
    def __init__(self, time_emb_dim: int = 32, timesteps: int = 1000):
        super().__init__()
        self.timesteps = timesteps
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim), nn.ReLU(), nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.conv_down1 = DoubleConv(1, 32)
        self.conv_down2 = DoubleConv(32, 64)
        self.conv_down3 = DoubleConv(64, 128)

        self.bottleneck = DoubleConv(128, 256)

        self.conv_up3 = DoubleConv(256 + 128, 128)
        self.conv_up2 = DoubleConv(128 + 64, 64)
        self.conv_up1 = DoubleConv(64 + 32 + time_emb_dim, 32)

        self.conv_last = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Time embedding
        t = t.float() / self.timesteps
        t_emb = self.time_mlp(t.unsqueeze(-1))

        # Downsampling
        conv1 = self.conv_down1(x)
        x = F.max_pool2d(conv1, 2)
        conv2 = self.conv_down2(x)
        x = F.max_pool2d(conv2, 2)
        conv3 = self.conv_down3(x)
        x = F.max_pool2d(conv3, 2)

        # Bottleneck
        x = self.bottleneck(x)

        # Upsampling
        x = F.interpolate(x, size=conv3.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, conv3], dim=1)
        x = self.conv_up3(x)

        x = F.interpolate(x, size=conv2.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, conv2], dim=1)
        x = self.conv_up2(x)

        x = F.interpolate(x, size=conv1.shape[2:], mode="bilinear", align_corners=True)
        t_emb_expanded = (
            t_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[-2], x.shape[-1])
        )
        x = torch.cat([x, conv1, t_emb_expanded], dim=1)
        x = self.conv_up1(x)

        return self.conv_last(x)


def transform_sampled_image(image):
    image = (image.clamp(-1, 1) + 1) / 2  # rescale to [0, 1]
    return image


class Diffusion:
    def __init__(
        self,
        noise_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        img_size=64,
        device="cpu",
    ):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        self.img_size = img_size

        # set coefficients for noise schedule
        self.beta = self._prepare_noise_schedule().to(
            device
        )  # forward process variances (here we use a linear schedule)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def _prepare_noise_schedule(self):
        # linear schedule for beta
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def _sample_q(self, x, t):
        # Eq. 4 in the DDPM paper (Ho et al. 2020) after reparametrization
        # noisy images at time step t xt = x* sqrt(apha_bar) + sqrt(1-alpha_bar) * e
        # with e ~ N(0, I).

        pass

        # sample noise from standard normal distribution
        pass

        # with reparametrization trick (q(xt|x0) = N(xt; x0*sqrt(alpha_bar), sqrt(1-alpha_bar)) we get the following
        # sample from q(xt|x0) = x0*sqrt(alpha_bar) + sqrt(1-alpha_bar) * e

        pass

    def _sample_timesteps(self, n):
        # return random integers between 1 and noise_steps of size n
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    @staticmethod
    def loss_simple(epsilon, epsilon_pred) -> torch.nn.MSELoss:
        # Eq. 14 in the DDPM paper - very simple loss aka MSE between predicted and true noise
        pass

    def perform_training_step(self, model, image):
        # Step 2, Algorithm 1: sample image x0 ~q(x0)
        pass
        # Step 3, Algorithm 1: generate random timesteps
        pass
        # Step 4, Algorithm 1: sample noise for each timestep (epsilon)
        pass

        # Step 5, Algorithm 1: Take gradient descent step on
        # predict noise
        pass

        # calculate loss (here we use a simple MSE loss)
        pass

    @torch.no_grad()
    def sample_step(self, model, x_t, t):
        # Step 3, Algorithm 2: sample noise for next time step
        # Forward Pass (predict noise for current time step)
        pass

        # get alpha and beta for current time step
        pass

        # Step 3, Algorithm 2: sample noise for next time step
        if t[0] > 1:
            pass
        else:
            # last step, add no noise, otherwise it would get worse
            pass

        # Step 4, Algorithm 2: update x_t-1 (remove a little bit of noise)
        pass

    def sample(self, model, t_sample_times=None):
        # Following Algorithm 2 in the DDPM paper
        model.eval()
        # Step 1, Algorithm 2:
        # start with random noise (x_T)
        pass

        # Step 2, Algorithm 2: go over all time steps in reverse order (from noise to image)
        pass
        for i in reversed(range(1, self.noise_steps)):
            # timestep encoding
            pass

            # Perform a sampling step
            pass

            # save sampled images if requested:

        model.train()


def load_pretrained_model(args: MLPTrainingArgs, model_path: str, device: torch.device):
    """
    Load a trained model from the given path.
    """
    model = Model(hidden_sizes=args.hidden_sizes)

    # Ensure model_path is valid and remove invalid argument
    model.load_state_dict(torch.load(model_path, weights_only=True))

    return model.to(device)  # Move model to specified device
