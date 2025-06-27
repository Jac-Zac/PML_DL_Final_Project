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
    def _monte_carlo_sample_final_step(
        self,
        mean_x0: Tensor,
        var_x0: Tensor,
        S: int = 10,
    ) -> Tuple[Tensor, Tensor]:
        """
        Perform Monte Carlo sampling at t=0 to estimate final uncertainty.
        Args:
            mean_x0: Mean of x_0 estimated by diffusion.
            var_x0: Variance of x_0 estimated by propagation.
            S: Number of Monte Carlo samples.

        Returns:
            mc_mean: Empirical mean of samples.
            mc_var: Empirical pixel-wise variance of samples.
        """
        std_x0 = torch.sqrt(torch.clamp(var_x0, min=1e-8))
        samples = [mean_x0 + std_x0 * torch.randn_like(mean_x0) for _ in range(S)]
        samples = torch.stack(samples, dim=0)  # [S, B, C, H, W]

        mc_mean = samples.mean(dim=0)  # [B, C, H, W]
        mc_var = samples.var(dim=0, unbiased=False)  # Pixel-wise variance

        return mc_mean, mc_var

    @torch.no_grad()
    def sample_with_uncertainty(
        self,
        model: nn.Module,
        t_sample_times: Optional[List[int]] = None,
        channels: int = 1,
        log_intermediate: bool = False,
        y: Optional[Tensor] = None,
        estimate_cov: bool = True,
    ) -> Tuple[List[Tensor], List[Tensor], Optional[List[Tensor]]]:
        """
        Iteratively sample from the model, tracking predictive uncertainty and optionally Cov(x, ε).
        """
        model.eval()
        batch_size = 1 if y is None else y.size(0)

        x_t = torch.randn(
            batch_size, channels, self.img_size, self.img_size, device=self.device
        )
        var_x_t = torch.zeros_like(x_t)

        intermediates, uncertainties = [], []
        covariances = [] if estimate_cov else None

        for i in reversed(range(self.noise_steps)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)

            # Predict noise and its variance
            eps_pred, eps_var = model(x_t, t, y=y)

            # Sample xt-1
            beta_t = self.beta[t].view(-1, 1, 1, 1)
            alpha_t = self.alpha[t].view(-1, 1, 1, 1)
            alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)

            coef1 = 1.0 / alpha_t.sqrt()
            coef2 = (1.0 - alpha_t) / (1.0 - alpha_bar_t).sqrt()
            x_prev_mean = coef1 * (x_t - coef2 * eps_pred)

            if i > 0:
                total_var = beta_t + eps_var.clamp(min=1e-8)
                noise = torch.randn_like(x_t)
                x_t = x_prev_mean + total_var.sqrt() * noise
            else:
                x_t = x_prev_mean

            # Propagate uncertainty
            var_x_t += eps_var.clamp(min=1e-8) + beta_t
            uncertainties.append(var_x_t.sum(dim=(1, 2, 3)).cpu())

            # Compute variance
            if estimate_cov:
                # Stima empirica della covarianza pixel-wise: Cov(x, ε)
                x_mean = x_t.mean(dim=(1, 2, 3), keepdim=True)
                eps_mean = eps_pred.mean(dim=(1, 2, 3), keepdim=True)
                cov_xt_eps = ((x_t - x_mean) * (eps_pred - eps_mean)).mean(
                    dim=(1, 2, 3), keepdim=True
                )
                covariances.append(cov_xt_eps.cpu())

            # Log immagini intermedie
            if log_intermediate and t_sample_times and i in t_sample_times:
                intermediates.append(self.transform_sampled_image(x_t.clone()))

        intermediates.append(self.transform_sampled_image(x_t))
        model.train()
        return intermediates, uncertainties, covariances

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
