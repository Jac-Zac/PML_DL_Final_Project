from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn


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
        self.device = device
        self.img_size = img_size

        self.beta = self._prepare_noise_schedule(beta_start, beta_end).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def _prepare_noise_schedule(self, beta_start: float, beta_end: float) -> Tensor:
        return torch.linspace(beta_start, beta_end, self.noise_steps)

    def _sample_timesteps(self, batch_size: int) -> Tensor:
        return torch.randint(1, self.noise_steps, (batch_size,), device=self.device)

    def _sample_q(self, x_0: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        sqrt_alpha_bar = self.alpha_bar[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = (1.0 - self.alpha_bar[t]).sqrt().view(-1, 1, 1, 1)
        noise = torch.randn_like(x_0)
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        return x_t, noise

    @staticmethod
    def loss_simple(epsilon: Tensor, epsilon_pred: Tensor) -> Tensor:
        return nn.functional.mse_loss(epsilon_pred, epsilon)

    def perform_training_step(
        self,
        model: nn.Module,
        x_0: Tensor,
        y: Optional[Tensor] = None,
        t: Optional[Tensor] = None,
    ) -> Tensor:
        x_0 = x_0.to(self.device)
        if t is None:
            t = self._sample_timesteps(x_0.size(0))
        x_t, noise = self._sample_q(x_0, t)
        noise_pred = model(x_t, t, y=y)
        return self.loss_simple(noise, noise_pred)

    @torch.no_grad()
    def sample_step(
        self,
        model: nn.Module,
        x_t: Tensor,
        t: Tensor,
        y: Optional[Tensor] = None,
    ) -> Tensor:
        beta_t = self.beta[t].view(-1, 1, 1, 1)
        alpha_t = self.alpha[t].view(-1, 1, 1, 1)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)

        noise_pred = model(x_t, t, y=y)
        z = torch.randn_like(x_t) if t[0] > 1 else torch.zeros_like(x_t)

        coef1 = 1.0 / alpha_t.sqrt()
        coef2 = (1.0 - alpha_t) / (1.0 - alpha_bar_t).sqrt()
        x_prev = coef1 * (x_t - coef2 * noise_pred) + beta_t.sqrt() * z
        return x_prev

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        t_sample_times: Optional[List[int]] = None,
        channels: int = 1,
        log_intermediate: bool = False,
        y: Optional[Tensor] = None,
    ) -> List[Tensor]:
        model.eval()
        batch_size = 1 if y is None else y.size(0)
        x_t = torch.randn(
            batch_size, channels, self.img_size, self.img_size, device=self.device
        )
        intermediates = []

        for i in reversed(range(self.noise_steps)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            x_t = self.sample_step(model, x_t, t, y)

            if log_intermediate and t_sample_times and i in t_sample_times:
                intermediates.append(self.transform_sampled_image(x_t.clone()))

        intermediates.append(self.transform_sampled_image(x_t))
        model.train()
        return intermediates

    @staticmethod
    def transform_sampled_image(image: Tensor) -> Tensor:
        return (image.clamp(-1, 1) + 1) / 2


#########################################################################################
#                        UNCERTAINTY QUANTIFICATION DIFFUSION                           #
#########################################################################################


class QUDiffusion(Diffusion):
    """
    Diffusion model with uncertainty estimation capabilities.
    Extends the base Diffusion class to support Laplace approximation models.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample_from_gaussian(self, mean: Tensor, var: Tensor) -> Tensor:
        """Sample from Gaussian distribution with given mean and variance."""
        std = torch.sqrt(torch.clamp(var, min=1e-8))
        return mean + std * torch.randn_like(mean)

    def perform_training_step(
        self,
        model: nn.Module,
        x_0: Tensor,
        y: Optional[Tensor] = None,
        t: Optional[Tensor] = None,
    ) -> Tensor:
        """Override to use accurate_forward during training if available."""
        x_0 = x_0.to(self.device)
        if t is None:
            t = self._sample_timesteps(x_0.size(0))
        x_t, noise = self._sample_q(x_0, t)

        noise_pred = model(x_t, t, y=y)

        return self.loss_simple(noise, noise_pred)

    @torch.no_grad()
    def sample_step(
        self,
        model: nn.Module,
        x_t: Tensor,
        t: Tensor,
        y: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Override sample_step to optionally include uncertainty.
        """
        return self._sample_step_with_uncertainty(model, x_t, t, y)

    def _sample_step_with_uncertainty(
        self,
        model: nn.Module,
        x_t: Tensor,
        t: Tensor,
        y: Optional[Tensor] = None,
    ) -> Tensor:
        """Sampling step with uncertainty estimation."""
        beta_t = self.beta[t].view(-1, 1, 1, 1)
        alpha_t = self.alpha[t].view(-1, 1, 1, 1)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)

        # Get noise prediction with uncertainty
        noise_pred, noise_var = model(x_t, t, y=y)

        # Standard diffusion coefficients
        coef1 = 1.0 / alpha_t.sqrt()
        coef2 = (1.0 - alpha_t) / (1.0 - alpha_bar_t).sqrt()

        # Compute mean of x_prev
        x_prev_mean = coef1 * (x_t - coef2 * noise_pred)

        # Add scheduled noise
        if t[0] > 1:
            scheduled_noise = torch.randn_like(x_t) * beta_t.sqrt()
        else:
            scheduled_noise = torch.zeros_like(x_t)

        x_prev = x_prev_mean + scheduled_noise
        return x_prev

    @torch.no_grad()
    def sample_with_uncertainty(
        self,
        model: nn.Module,
        t_sample_times: Optional[List[int]] = None,
        channels: int = 1,
        log_intermediate: bool = False,
        y: Optional[Tensor] = None,
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Sample with uncertainty estimation at specified timesteps.

        Args:
            model: LaplaceApproxModel instance
            uncertainty_schedule: Boolean list indicating which timesteps to use uncertainty
            t_sample_times: Timesteps to log intermediates
            channels: Number of image channels
            log_intermediate: Whether to log intermediate results
            y: Conditional labels

        Returns:
            intermediates: List of generated samples
            uncertainties: List of uncertainty estimates (if return_uncertainties=True)
        """
        model.eval()
        batch_size = 1 if y is None else y.size(0)

        # NOTE: Always use uncertainty or implement bayeskip
        # Initialize uncertainty schedule if not provided
        uncertainty_start = int(0 * self.noise_steps)
        uncertainty_schedule = [i >= uncertainty_start for i in range(self.noise_steps)]

        # Pad uncertainty schedule if too short
        while len(uncertainty_schedule) < self.noise_steps:
            uncertainty_schedule.append(False)

        # Initialize sampling
        x_t = torch.randn(
            batch_size, channels, self.img_size, self.img_size, device=self.device
        )

        intermediates = []
        uncertainties = []

        # Track uncertainty for propagation
        var_x_t = torch.zeros_like(x_t)

        for i in reversed(range(self.noise_steps)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            use_uncertainty = uncertainty_schedule[i]

            # Perform sampling step
            x_t = self.sample_step(model, x_t, t, y=y)

            # Update uncertainty if tracking
            if use_uncertainty and hasattr(model, "forward"):
                # Get uncertainty estimate
                with torch.no_grad():
                    _, noise_var = model(x_t, t, y=y)
                    if noise_var is not None:
                        # Simple uncertainty propagation
                        beta_t = self.beta[t].view(-1, 1, 1, 1)
                        var_x_t = var_x_t + noise_var + beta_t
                    else:
                        var_x_t = var_x_t + self.beta[t].view(-1, 1, 1, 1)
            else:
                # Just add scheduled noise variance
                var_x_t = var_x_t + self.beta[t].view(-1, 1, 1, 1)

            # Store total uncertainty
            uncertainties.append(var_x_t.sum(dim=(1, 2, 3)).cpu())

            # Log intermediate if requested
            if log_intermediate and t_sample_times and i in t_sample_times:
                intermediates.append(self.transform_sampled_image(x_t.clone()))

        # Add final sample
        intermediates.append(self.transform_sampled_image(x_t))
        model.train()

        return intermediates, uncertainties

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        t_sample_times: Optional[List[int]] = None,
        channels: int = 1,
        log_intermediate: bool = False,
        y: Optional[Tensor] = None,
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Override sample method to optionally use uncertainty.

        If uncertainty_schedule is provided, uses uncertainty sampling,
        otherwise falls back to deterministic sampling for backward compatibility.
        """
        intermediates, uncertainties = self.sample_with_uncertainty(
            model=model,
            t_sample_times=t_sample_times,
            channels=channels,
            log_intermediate=log_intermediate,
            y=y,
        )
        return intermediates, uncertainties
