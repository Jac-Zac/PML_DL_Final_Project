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
        num_monte_carlo_sample: int = 10,
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor], Tensor, Tensor]:
        """
        Iteratively sample from the model with predictive uncertainty.

        Returns:
            intermediates: Images at intermediate timesteps (if enabled)
            uncertainties: Accumulated predictive variances
            covariances: Cov(x_t, eps_t) per step
            mc_mean_x0: Empirical mean of final x_0 samples
            mc_var_x0: Empirical pixel-wise variance via MC
        """
        model.eval()
        batch_size = 1 if y is None else y.size(0)
        device = self.device

        # Step 1: Initialize sampling
        x_t = torch.randn(
            batch_size, channels, self.img_size, self.img_size, device=device
        )
        var_x_t = torch.zeros_like(x_t)
        cov_xt_eps = torch.zeros_like(x_t)

        intermediates, uncertainties, covariances = [], [], []

        # Step 2: Reverse diffusion
        for i in reversed(range(self.noise_steps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)

            # Get predictive mean and variance of noise
            eps_t_mean, eps_t_var = model(x_t, t, y=y)

            # Sample actual noise (step 5 in Algorithm 1)
            eps_t_sampled = eps_t_mean + torch.sqrt(
                torch.clamp(eps_t_var, min=1e-8)
            ) * torch.randn_like(x_t)

            # Estimate xt-1 (Equation 7: DDPM mean step)
            beta_t = self.beta[t].view(-1, 1, 1, 1)
            alpha_t = self.alpha[t].view(-1, 1, 1, 1)
            alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)

            coef1 = 1.0 / torch.sqrt(alpha_t)
            coef2 = (1.0 - alpha_t) / torch.sqrt(alpha_bar_t)
            x_prev_mean = coef1 * (x_t - coef2 * eps_t_sampled)

            # Estimate variance of xt-1 via Equation (8)
            total_var = beta_t + eps_t_var.clamp(min=1e-8)
            var_x_t = self._var_iteration(
                var_x_t, self.noise_schedule, t + 1, t, cov_xt_eps, eps_t_var
            )

            # Sample xt-1 from Gaussian (Equation 10)
            x_t = self.sample_from_gaussian(x_prev_mean, total_var)

            # Estimate Cov(xt-1, eps_t-1) (Equation 11)
            cov_xt_eps = (x_t - x_prev_mean) * eps_t_sampled

            # Store intermediate results
            uncertainties.append(var_x_t.sum(dim=(1, 2, 3)).cpu())
            covariances.append(cov_xt_eps.sum(dim=(1, 2, 3)).cpu())

            if log_intermediate and t_sample_times and i in t_sample_times:
                intermediates.append(self.transform_sampled_image(x_t.clone()))

        # Final MC sampling step at t=0
        mc_mean_x0, mc_var_x0 = self._monte_carlo_sample_final_step(
            x_t, var_x_t, num_monte_carlo_sample
        )

        # Append final image to intermediates
        intermediates.append(self.transform_sampled_image(mc_mean_x0))

        model.train()
        return intermediates, uncertainties, covariances, mc_mean_x0, mc_var_x0

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
