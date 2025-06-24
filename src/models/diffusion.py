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


from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class QUDiffusion(Diffusion):
    """
    Diffusion model with uncertainty estimation.
    Extends the base Diffusion class to support predictive mean and variance propagation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample_from_gaussian(self, mean: Tensor, var: Tensor) -> Tensor:
        """Sample from a Gaussian given mean and variance."""
        std = torch.sqrt(torch.clamp(var, min=1e-8))
        return mean + std * torch.randn_like(mean)

    # ----------------- Private utility iterations -----------------

    def _exp_iteration(
        self,
        exp_xt: Tensor,
        ns,
        s: Tensor,
        t: Tensor,
        mc_eps_exp_s1: Tensor,
    ) -> Tensor:
        """
        Compute mean propagation for one diffusion step.

        Uses noise schedule to scale mean and predicted noise.
        """
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        sigma_t = ns.marginal_std(t)
        log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(
            s
        ), ns.marginal_log_mean_coeff(t)
        phi_1 = torch.expm1(h)

        return (
            torch.exp(log_alpha_t - log_alpha_s) * exp_xt
            - (sigma_t * phi_1) * mc_eps_exp_s1
        )

    def _var_iteration(
        self,
        var_xt: Tensor,
        ns,
        s: Tensor,
        t: Tensor,
        cov_xt_epst: Tensor,
        var_epst: Tensor,
    ) -> Tensor:
        """
        Compute variance propagation for one diffusion step.
        """
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        sigma_t = ns.marginal_std(t)
        log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(
            s
        ), ns.marginal_log_mean_coeff(t)
        phi_1 = torch.expm1(h)

        return (
            torch.square(torch.exp(log_alpha_t - log_alpha_s)) * var_xt
            - 2 * torch.exp(log_alpha_t - log_alpha_s) * (sigma_t * phi_1) * cov_xt_epst
            + torch.square(sigma_t * phi_1) * var_epst
        )

    # ----------------- Uncertainty-aware sampling -----------------

    def perform_training_step(
        self,
        model: nn.Module,
        x_0: Tensor,
        y: Optional[Tensor] = None,
        t: Optional[Tensor] = None,
    ) -> Tensor:
        """Training is not supported for QUDiffusion."""
        raise NotImplementedError(
            "This class is only for sampling with uncertainty, not training."
        )

    @torch.no_grad()
    def _sample_step_with_uncertainty(
        self,
        model: nn.Module,
        x_t: Tensor,
        t: Tensor,
        y: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Sample one reverse-diffusion step using predictive variance.
        """
        beta_t = self.beta[t].view(-1, 1, 1, 1)
        alpha_t = self.alpha[t].view(-1, 1, 1, 1)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)

        # Get predicted noise and its variance
        noise_pred, noise_var = model(x_t, t, y=y)

        # DDPM mean
        coef1 = 1.0 / alpha_t.sqrt()
        coef2 = (1.0 - alpha_t) / (1.0 - alpha_bar_t).sqrt()
        x_prev_mean = coef1 * (x_t - coef2 * noise_pred)

        # Only add noise if not last step
        if t[0] > 0:
            total_var = beta_t + noise_var.clamp(min=1e-8)
            noise = torch.randn_like(x_t)
            return x_prev_mean + total_var.sqrt() * noise
        return x_prev_mean

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
        Iteratively sample from the model, tracking predictive uncertainty.

        Args:
            model: LaplaceApproxModel.
            t_sample_times: Steps to store intermediate samples.
            channels: Image channels.
            log_intermediate: Whether to log intermediate samples.
            y: Optional conditional labels.

        Returns:
            intermediates: List of generated image tensors.
            uncertainties: Uncertainty estimates at each step.
        """
        model.eval()
        batch_size = 1 if y is None else y.size(0)

        # Prepare sampling
        x_t = torch.randn(
            batch_size, channels, self.img_size, self.img_size, device=self.device
        )
        intermediates, uncertainties = [], []
        var_x_t = torch.zeros_like(x_t)

        # Run reverse diffusion
        for i in reversed(range(self.noise_steps)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)

            # Step with uncertainty-aware noise
            x_t = self._sample_step_with_uncertainty(model, x_t, t, y)

            # Accumulate predictive noise variance
            with torch.no_grad():
                _, noise_var = model(x_t, t, y=y)
                var_x_t += noise_var.clamp(min=1e-8) + self.beta[t].view(-1, 1, 1, 1)

            uncertainties.append(var_x_t.sum(dim=(1, 2, 3)).cpu())
            if log_intermediate and t_sample_times and i in t_sample_times:
                intermediates.append(self.transform_sampled_image(x_t.clone()))

        # Append final image
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
        """Direct sampling is not supported. Use sample_with_uncertainty instead."""
        raise NotImplementedError(
            "QUDiffusion requires predictive variance; use sample_with_uncertainty()."
        )
