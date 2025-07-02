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
                intermediates.append(self.transform_sampled_image(x_t.clone().cpu()))

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

    # FIX: DDPM Very slow ... and perhaps not good results
    """
    @torch.no_grad()
    def sample_with_uncertainty(
        self,
        model: nn.Module,
        channels: int = 1,
        y: Optional[Tensor] = None,
        cov_num_sample: int = 10,
        log_intermediate: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        # Iteratively sample from the model, tracking predictive uncertainty and optionally Cov(x, ε).

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
            beta_t = self.beta[self.noise_steps - 1 - t].view(-1, 1, 1, 1)
            alpha_t = self.alpha[self.noise_steps - 1 - t].view(-1, 1, 1, 1)
            alpha_bar_t = self.alpha_bar[self.noise_steps - 1 - t].view(-1, 1, 1, 1)

            # Mean and x_t-1
            coef1 = 1.0 / alpha_t.sqrt()
            coef2 = (1.0 - alpha_t) / (1.0 - alpha_bar_t).sqrt()

            x_prev_mean = coef1 * (x_t_mean - coef2 * eps_mean)
            x_prev = (
                coef1 * (x_t - coef2 * eps_t) + torch.randn_like(x_t) * beta_t.sqrt()
            )

            # Variance
            coef3 = 2 * (1 - alpha_t) / (alpha_t * (1 - alpha_bar_t).sqrt())
            coef4 = (1 - alpha_t) ** 2 / (alpha_t * (1 - alpha_bar_t))
            x_prev_var = (
                (1 / alpha_t * x_t_var) - (coef3 * cov_t) + (coef4 * eps_var) + beta_t
            )

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

            # Log intermediate images
            if log_intermediate:
                intermediates.append(self.transform_sampled_image(x_t.clone()))
                uncertainties.append(x_t_var.clone().cpu())  # per-pixel variance

            x_t = x_prev
            x_t_mean = x_prev_mean
            x_t_var = x_prev_var
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
    """

    @torch.no_grad()
    def sample_with_uncertainty(
        self,
        model: nn.Module,
        channels: int = 1,
        y: Optional[Tensor] = None,
        cov_num_sample: int = 10,
        num_steps: int = 50,
        log_intermediate: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """
        Iteratively sample from the model using DDIM with uncertainty tracking.

        Args:
            model: Denoising model.
            channels: Number of channels in the image.
            y: Optional conditioning.
            cov_num_sample: Monte Carlo samples for covariance estimation.
            sampling_steps: Number of inference steps (≤ self.noise_steps).
            eta: Stochasticity (0 = deterministic DDIM, 1 = DDPM-like).
            log_intermediate: Whether to store intermediate outputs.

        Returns:
            Tuple of tensors: (intermediates, uncertainties)
        """
        model.eval()
        batch_size = 1 if y is None else y.size(0)

        # Define the time schedule - make sure to use proper dtype
        step_indices = (
            torch.linspace(0, self.noise_steps - 1, num_steps).long().to(self.device)
        )

        # Use alpha_bar (cumulative alphas) for DDIM
        alphas_bar = self.alpha_bar[step_indices].to(self.device)

        x_t = torch.randn(
            batch_size, channels, self.img_size, self.img_size, device=self.device
        )
        x_t_mean = x_t.clone()
        x_t_var = torch.zeros_like(x_t)
        cov_t = torch.zeros_like(x_t)

        intermediates, uncertainties = [x_t_var.clone().cpu()], [x_t_var.clone().cpu()]

        for i in tqdm(list(reversed(range(num_steps))), desc="Sampling", leave=False):
            t = step_indices[i].expand(batch_size)

            # Get noise prediction and its variance
            eps_mean, eps_var = model(x_t, t, y=y)
            eps_t = eps_mean + torch.sqrt(eps_var) * torch.randn_like(eps_mean)

            # Current timestep parameters
            alpha_t = alphas_bar[i].view(-1, 1, 1, 1)

            # HACK: This is the code provided by Bayesdiff
            sigma_t = torch.sqrt(1 - alpha_t)

            # Predict x0 using correct DDIM formula
            x0_pred = (x_t - eps_t * sigma_t) / torch.sqrt(alpha_t)

            if i > 0:
                # Next timestep parameters
                alpha_t_prev = alphas_bar[i - 1].view(-1, 1, 1, 1)
                sigma_t_prev = torch.sqrt(1 - alpha_t_prev)

                # DDIM step: x_{t-1} = sqrt(alpha_{t-1}) * x0_pred + sqrt(1-alpha_{t-1}) * eps_t
                x_prev = torch.sqrt(alpha_t_prev) * x0_pred + sigma_t_prev * eps_t

                # Coefficients for uncertainty propagation (matching reference implementation)
                coeff1 = torch.sqrt(alpha_t_prev) / torch.sqrt(
                    alpha_t
                )  # sqrt(alpha_{t-1}) / sqrt(alpha_t)
                coeff2 = (
                    sigma_t_prev - coeff1 * sigma_t
                )  # sigma_{t-1} - coeff1 * sigma_t

                # Mean iteration (expectation propagation)
                x_prev_mean = coeff1 * x_t_mean + coeff2 * eps_mean

                # Variance iteration (following the reference var_iteration formula)
                x_prev_var = (
                    (coeff1**2) * x_t_var
                    + 2 * coeff1 * coeff2 * cov_t
                    + (coeff2**2) * eps_var
                )

                # Estimate Cov(x, ε) at next step
                if i > 1:  # Don't compute covariance for the last step
                    # Use the correct timestep for covariance estimation
                    next_t = step_indices[i - 1].expand(batch_size)
                    covariance = self.monte_carlo_covariance_estim(
                        model=model,
                        t=next_t,
                        x_mean=x_prev_mean,
                        x_var=x_prev_var,
                        S=cov_num_sample,
                        y=y,
                    )
                else:
                    covariance = torch.zeros_like(cov_t)
            else:
                # Final step: denoise to clean image (alpha_0 = 1, sigma_0 = 0)
                x_prev = x0_pred  # At t=0, x_0 = x0_pred
                x_prev_mean = (x_t_mean - sigma_t * eps_mean) / torch.sqrt(alpha_t)
                x_prev_var = x_t_var / alpha_t + (sigma_t**2) * eps_var / alpha_t
                covariance = torch.zeros_like(cov_t)

            if log_intermediate:
                intermediates.append(self.transform_sampled_image(x_t.clone().cpu()))
                uncertainties.append(x_t_var.clone().cpu())

            # Update for next iteration
            x_t = x_prev
            x_t_mean = x_prev_mean
            x_t_var = x_prev_var
            cov_t = covariance

        model.train()

        uncertainties = (
            torch.stack(uncertainties) if uncertainties else torch.tensor([])
        )
        intermediates = (
            torch.stack(intermediates) if intermediates else torch.tensor([])
        )

        return intermediates, uncertainties
