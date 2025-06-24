from typing import Optional

import torch
from torch import Tensor, nn


class FlowMatching:
    """
    Flow Matching method for learning velocity fields between data distributions.

    This implements the continuous-time flow matching objective where the model
    learns to predict the velocity (dx/dt) along linear paths between samples x0 and x1.

    Args:
        img_size: Spatial resolution of square images.
        device: Torch device for computations.
    """

    def __init__(self, img_size: int = 64, device: torch.device = torch.device("cpu")):
        self.img_size = img_size
        self.device = device

    def _sample_timesteps(self, batch_size: int) -> Tensor:
        """
        Sample random time points t uniformly from [0, 1].

        Args:
            batch_size: Number of samples to draw.

        Returns:
            Tensor of shape (batch_size,) with uniform samples.
        """
        return torch.rand(batch_size, device=self.device)

    def perform_training_step(
        self,
        model: nn.Module,
        x1: Tensor,  # Target data, scaled to [-1, 1]
        y: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Perform a single training step computing the MSE loss for velocity prediction.

        Args:
            model: The velocity prediction model v(x, t, y).
            x1: Target samples (end points of paths).
            y: Optional conditioning information.

        Returns:
            Scalar tensor representing the MSE loss between predicted and true velocity.
        """
        # Sample starting points x0 from prior (e.g. Gaussian noise)
        x0 = torch.randn_like(x1)

        batch_size = x0.size(0)
        t = self._sample_timesteps(batch_size)
        t_expanded = t.view(-1, 1, 1, 1)  # Broadcast to image shape

        # Linear interpolation between x0 and x1 at time t: x_t = (1 - t) x0 + t x1
        x_t = (1 - t_expanded) * x0 + t_expanded * x1

        # True velocity along the path: dx/dt = x1 - x0
        dx = x1 - x0

        # Predicted velocity by the model
        v = model(x_t, t, y=y)
        assert v.shape == dx.shape

        # Mean squared error loss on velocity prediction
        loss = ((v - dx) ** 2).mean()
        return loss

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        steps: int = 100,
        y: Optional[Tensor] = None,
        x_init: Optional[Tensor] = None,
        log_intermediate: bool = False,
        t_sample_times: Optional[list[int]] = None,
    ) -> list[Tensor]:
        """
        Generate samples by integrating predicted velocity fields over time.

        Args:
            model: Trained velocity prediction model.
            steps: Number of integration steps.
            y: Optional conditioning.
            x_init: Optional initial sample x_0; if None, sample from standard Gaussian.
            log_intermediate: Whether to save intermediate images during sampling.
            t_sample_times: Specific steps at which to log intermediate images.

        Returns:
            List of tensors with sampled images at requested intermediate steps.
        """
        model.eval()

        batch_size = (
            y.size(0)
            if y is not None
            else (x_init.size(0) if x_init is not None else 1)
        )
        channels = x_init.size(1) if x_init is not None else 1

        # Initialize x_t at time 0 (either given or standard Gaussian noise)
        x_t = (
            x_init.to(self.device)
            if x_init is not None
            else torch.randn(
                batch_size, channels, self.img_size, self.img_size, device=self.device
            )
        )

        dt = 1.0 / steps
        results = []

        for i in range(steps):
            t = torch.full((batch_size,), i / steps, device=self.device)
            v = model(x_t, t, y=y)  # predicted velocity dx/dt
            x_t = x_t + v * dt  # Euler integration step

            if log_intermediate and t_sample_times and i in t_sample_times:
                results.append(self.transform_sampled_image(x_t.clone()))

        return results

    @staticmethod
    def transform_sampled_image(image: Tensor) -> Tensor:
        """
        Transform images from [-1, 1] to [0, 1] range for visualization.

        Args:
            image: Tensor with pixel values in [-1, 1].

        Returns:
            Tensor with pixel values in [0, 1].
        """
        return (image.clamp(-1, 1) + 1) / 2
