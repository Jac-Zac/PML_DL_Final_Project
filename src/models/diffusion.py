from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from tqdm.autonotebook import tqdm


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
        channels: int = 1,
        y: Optional[Tensor] = None,
        log_intermediate: bool = False,
    ) -> Tensor:
        model.eval()
        batch_size = 1 if y is None else y.size(0)
        x_t = torch.randn(
            batch_size, channels, self.img_size, self.img_size, device=self.device
        )
        intermediates = []

        for i in reversed(range(self.noise_steps)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            x_t = self.sample_step(model, x_t, t, y)

            if log_intermediate:
                intermediates.append(self.transform_sampled_image(x_t.clone()))

        intermediates.append(self.transform_sampled_image(x_t))
        intermediates = torch.stack(intermediates)  # [n_steps, B, C, H, W]
        model.train()
        return intermediates

    @staticmethod
    def transform_sampled_image(image: Tensor) -> Tensor:
        return (image.clamp(-1, 1) + 1) / 2


#########################################################################################
#                        UNCERTAINTY QUANTIFICATION DIFFUSION                           #
#########################################################################################


class UQDiffusion(Diffusion):
    """
    Diffusion model with uncertainty estimation capabilities.
    Extends the base Diffusion class to support Laplace approximation models.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        eps_samples = [model.accurate_forward(x_i, t, y=y) for x_i in x_samples]

        x_samples = torch.stack(x_samples, dim=0)  # [S, B, C, H, W]
        eps_samples = torch.stack(eps_samples, dim=0)  # [S, B, C, H, W]

        # Compute covariance with numerical stability
        x_centered = x_samples - x_mean.unsqueeze(0)
        v_centered = eps_samples - torch.mean(eps_samples, dim=0, keepdim=True)

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
        cov_num_sample: int = 10,
        log_intermediate: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """
        Iteratively sample from the model, tracking predictive uncertainty and optionally Cov(x, ε).
        """
        model.eval()
        batch_size = 1 if y is None else y.size(0)

        x_t = torch.randn(
            batch_size, channels, self.img_size, self.img_size, device=self.device
        )

        x_t_mean = x_t.clone()
        x_t_var = torch.zeros_like(x_t)
        cov_t = torch.zeros_like(x_t)

        intermediates, uncertainties = [], []

        for i in tqdm(reversed(range(self.noise_steps)), desc="Steps", leave=False):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)

            # Predict noise and its variance
            eps_mean, eps_var = model(x_t, t, y=y)  # mean and variance of noise
            eps_t = eps_mean + torch.sqrt(eps_var) * torch.randn_like(eps_mean)

            # Compute xt-1
            beta_t = self.beta[t].view(-1, 1, 1, 1)
            alpha_t = self.alpha[t].view(-1, 1, 1, 1)
            alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)

            # Mean and x_t-1
            coef1 = 1.0 / alpha_t.sqrt()
            coef2 = (1.0 - alpha_t) / (1.0 - alpha_bar_t).sqrt()
            x_prev_mean = coef1 * (x_t_mean - coef2 * eps_mean)
            x_prev = (
                coef1 * (x_t - coef2 * eps_t) + torch.randn_like(x_t) * beta_t.sqrt()
            )

            # Variance
            coef3 = 2 * (1 - alpha_t) / alpha_t * (1 - alpha_bar_t).sqrt()
            coef4 = (1 - alpha_t) ** 2 / alpha_t * (1 - alpha_bar_t)
            x_prev_var = (
                (1 / alpha_t * x_t_var) - (coef3 * cov_t) + (coef4 * eps_var) + beta_t
            )
            # (1 / alpha_t * x_t_var)
            # + (coef4 * eps_var)
            # + beta_t

            if i > 0:
                # Covariance estimation with Monte Carlo
                covariance = self.monte_carlo_covariance_estim(
                    model=model,
                    t=t - 1,
                    x_mean=x_prev_mean,
                    x_var=x_prev_var,
                    S=cov_num_sample,
                    y=y,
                )

            if i % 100 == 0 or i == self.noise_steps - 1:
                print(f"\nStep {i}")
                print(
                    f"  eps_var mean: {eps_var.mean().item():.4e}, std: {eps_var.std().item():.4e}"
                )
                print(
                    f"  Covariance mean: {cov_t.mean().item():.4e}, std: {cov_t.std().item():.4e}"
                )
                print(
                    f"  x_t_var mean: {x_t_var.mean().item():.4e}, std: {x_t_var.std().item():.4e}"
                )
                print(
                    f"  x_prev_var mean: {x_prev_var.mean().item():.4e}, std: {x_prev_var.std().item():.4e}"
                )

            # Log intermediate images
            if log_intermediate:
                intermediates.append(self.transform_sampled_image(x_t.clone()))
                uncertainties.append(x_t_var.clone().cpu())  # per-pixel variance

            x_t = x_prev
            x_t_mean = x_prev_mean
            x_t_var = x_prev_var
            cov_t = covariance

        uncertainties = torch.stack(uncertainties)  # [n_steps, B, C, H, W]
        intermediates = torch.stack(intermediates)  # [n_steps, B, C, H, W]

        model.train()
        return intermediates, uncertainties
