from typing import Optional

import torch
from torch import Tensor


class Diffusion:
    def __init__(
        self,
        noise_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        img_size: int = 64,
        device: torch.device = torch.device("cpu"),
    ):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        self.img_size = img_size

        # Prepare noise schedule (linear beta schedule)
        self.beta = self._prepare_noise_schedule().to(device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def _prepare_noise_schedule(self) -> Tensor:
        """Generate a linear noise schedule (beta)"""
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def _sample_q(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Sample from q(x_t | x_0) according to Eq. 4 in DDPM paper.

        Args:
            x: original images x_0, shape (B, C, H, W)
            t: timestep tensor, shape (B,)

        Returns:
            Noisy images x_t
        """
        # TODO: Implement reparameterization trick sampling
        ...

    def _sample_timesteps(self, n: int) -> Tensor:
        """Sample random timesteps for a batch"""
        return torch.randint(
            low=1, high=self.noise_steps, size=(n,), device=self.device
        )

    @staticmethod
    def loss_simple(epsilon: Tensor, epsilon_pred: Tensor) -> Tensor:
        """
        Simple MSE loss between true noise and predicted noise (Eq. 14 in DDPM paper).

        Args:
            epsilon: true noise
            epsilon_pred: predicted noise

        Returns:
            Scalar loss
        """
        # TODO: implement loss calculation
        ...

    def perform_training_step(self, model: torch.nn.Module, images: Tensor) -> Tensor:
        """
        Perform one training step on a batch of images.

        Args:
            model: neural network model
            images: batch of original images (x_0)

        Returns:
            Loss tensor
        """
        # TODO: implement training step algorithm
        ...

    @torch.no_grad()
    def sample_step(self, model: torch.nn.Module, x_t: Tensor, t: Tensor) -> Tensor:
        """
        Perform one step of sampling (denoising) according to Algorithm 2.

        Args:
            model: neural network model
            x_t: noisy images at timestep t
            t: current timestep

        Returns:
            Less noisy images at timestep t-1
        """
        # TODO: implement one sampling step
        ...

    def sample(
        self, model: torch.nn.Module, t_sample_times: Optional[int] = None
    ) -> Tensor:
        """
        Generate samples by reversing the diffusion process.

        Args:
            model: neural network model
            t_sample_times: optional, number of timesteps to sample (default: self.noise_steps)

        Returns:
            Generated images tensor
        """
        model.eval()
        # TODO: implement full sampling loop (Algorithm 2)
        ...
        model.train()
