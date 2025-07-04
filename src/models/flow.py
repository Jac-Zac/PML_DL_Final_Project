from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from tqdm.autonotebook import tqdm


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
            Tensor of shape (batch_size,) with uniform samples."""
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
        steps: int = 50,
        y: Optional[Tensor] = None,
        x_init: Optional[Tensor] = None,
        log_intermediate: bool = False,
    ) -> Tensor:
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
            x_t = x_t + v * dt  # Euler (explicit) integration step

            if log_intermediate:
                results.append(self.transform_sampled_image(x_t.clone().cpu()))

        results = torch.stack(results) if results else torch.tensor([])
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


class UQFlowMatching(FlowMatching):
    """
    Flow Matching with Uncertainty Quantification via Monte Carlo sampling.
    """

    def __init__(self, img_size: int = 64, device: torch.device = torch.device("cpu")):
        super().__init__(img_size, device)

    @torch.no_grad()
    def monte_carlo_covariance_estim(
        self,
        model: nn.Module,
        t: Tensor,
        x_mean: Tensor,
        x_var: Tensor,
        S: int = 100,
        y: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Perform Monte Carlo sampling to estimate covariance matrix.
        Args:
            x_mean: Mean of x_0 estimated by diffusion.
            x_var: Variance of x_0 estimated by propagation.
            S: Number of Monte Carlo samples.

        Returns:
            covariance: Empirical diagonal covariance
        """

        std_x = torch.sqrt(torch.clamp(x_var, min=1e-12))
        x_samples = [x_mean + std_x * torch.randn_like(x_mean) for _ in range(S)]
        v_samples = [model.accurate_forward(x_i, t, y=y) for x_i in x_samples]

        x_samples = torch.stack(x_samples, dim=0)  # [S, B, C, H, W]
        v_samples = torch.stack(v_samples, dim=0)  # [S, B, C, H, W]

        # Compute covariance with numerical stability
        x_centered = x_samples - x_mean.unsqueeze(0)
        v_centered = v_samples - torch.mean(v_samples, dim=0, keepdim=True)

        # NOTE: Compute the first term since second is 0 from the formula
        # And this avoids numerical instabilities
        covariance = torch.mean(x_centered * v_centered, dim=0)

        return covariance

    @torch.no_grad()
    def sample_with_uncertainty(
        self,
        model: nn.Module,
        channels: int = 1,
        y: Optional[Tensor] = None,
        cov_num_sample: int = 100,
        num_steps: int = 10,
        log_intermediate: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """
        Sample with uncertainty tracking and Cov(x, v) estimation.

        Returns:
            intermediates: List of sampled images at given steps.
            uncertainties: List of per-pixel variance maps at those steps.
        """
        model.eval()

        batch_size = 1 if y is None else y.size(0)

        x_t = torch.randn(
            batch_size, channels, self.img_size, self.img_size, device=self.device
        )
        x_t_mean = x_t.clone()
        x_t_var = torch.zeros_like(x_t)
        cov_t = torch.zeros_like(x_t)

        intermediates, uncertainties = [x_t.clone().cpu()], [x_t_var.clone().cpu()]
        dt = 1.0 / num_steps

        for i in tqdm(range(num_steps), desc="Steps", leave=False):
            t = torch.full((batch_size,), i * dt, device=self.device, dtype=torch.long)

            # Predict noise and its variance
            v_mean, v_var = model(x_t, t, y=y)  # mean and variance of noise

            v_t = v_mean + torch.sqrt(v_var) * torch.randn_like(v_mean)
            x_succ = x_t + dt * v_t

            # Mean
            x_succ_mean = x_t_mean + dt * v_mean

            # Variance
            x_succ_var = x_t_var + dt**2 * v_var + 2 * dt * cov_t

            # NOTE: Check if variance is negative and warn
            if (x_succ_var < -1e-12).any():
                print("Warning: Calculated variance is less then -1e-12.")

            # Covariance estimation with Monte Carlo
            covariance = self.monte_carlo_covariance_estim(
                model=model,
                t=t + dt,
                x_mean=x_succ_mean,
                x_var=x_succ_var,
                S=cov_num_sample,
                y=y,
            )

            # Log intermediate images
            if log_intermediate:
                intermediates.append(self.transform_sampled_image(x_succ.clone().cpu()))
                uncertainties.append(x_succ_var.clone().cpu())  # per-pixel variance

            x_t = x_succ
            x_t_mean = x_succ_mean
            x_t_var = x_succ_var
            cov_t = covariance

        # [num_steps, B, C, H, W]
        uncertainties = (
            torch.stack(uncertainties) if uncertainties else torch.tensor([])
        )
        # [num_steps, B, C, H, W]
        intermediates = (
            torch.stack(intermediates) if intermediates else torch.tensor([])
        )

        model.train()
        return intermediates, uncertainties
