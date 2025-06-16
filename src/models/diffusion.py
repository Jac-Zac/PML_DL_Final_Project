import numpy as np
import torch
from torch import Tensor, nn


class Diffusion:
    """
    Implements the DDPM (Denoising Diffusion Probabilistic Models) process
    as proposed by Ho et al., 2020. Supports both forward (noising) and
    reverse (denoising) diffusion steps. Includes DDIM sampling.
    """

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
        self.img_size = img_size
        self.device = device

        self.beta = self._prepare_noise_schedule().to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def _prepare_noise_schedule(self) -> Tensor:
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def _sample_timesteps(self, batch_size: int) -> Tensor:
        return torch.randint(1, self.noise_steps, (batch_size,), device=self.device)

    def _sample_q(self, x_0: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        sqrt_alpha_bar = self.alpha_bar[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = (1 - self.alpha_bar[t]).sqrt().view(-1, 1, 1, 1)
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
        y: Tensor | None = None,  # <-- accept labels here
    ) -> Tensor:
        """
        Executes a single DDPM training step:
        - Samples timestep
        - Noises image
        - Predicts noise (with optional class conditioning)
        - Computes MSE loss
        """
        x_0 = x_0.to(self.device)
        t = self._sample_timesteps(x_0.size(0))
        x_t, noise = self._sample_q(x_0, t)

        # pass y through to the model’s forward
        noise_pred = model(x_t, t, y=y)
        return self.loss_simple(noise, noise_pred)

    @torch.no_grad()
    def sample_step(
        self, model: nn.Module, x_t: Tensor, t: Tensor, y: torch.Tensor | None = None
    ) -> Tensor:
        beta_t = self.beta[t].view(-1, 1, 1, 1)
        alpha_t = self.alpha[t].view(-1, 1, 1, 1)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)

        # Pass conditioning y to the model
        noise_pred = model(x_t, t, y=y)
        z = torch.randn_like(x_t) if t[0] > 1 else torch.zeros_like(x_t)

        coef1 = 1 / alpha_t.sqrt()
        coef2 = (1 - alpha_t) / (1 - alpha_bar_t).sqrt()
        x_prev = coef1 * (x_t - coef2 * noise_pred) + beta_t.sqrt() * z
        return x_prev

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        t_sample_times: list[int] | None = None,
        channels: int = 1,
        log_intermediate: bool = False,
        y: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        model.eval()

        batch_size = 1 if y is None else y.shape[0]
        x_t = torch.randn(
            batch_size, channels, self.img_size, self.img_size, device=self.device
        )
        intermediates = []

        for i in reversed(range(0, self.noise_steps)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            x_t = self.sample_step(model, x_t, t, y)

            if log_intermediate and t_sample_times and i in t_sample_times:
                intermediates.append(self.transform_sampled_image(x_t.clone()))

        final_image = self.transform_sampled_image(x_t.clone())
        model.train()

        return intermediates + [final_image]

    @torch.no_grad()
    def sample_ddim(
        self,
        model: nn.Module,
        eta: float = 0.0,
        t_sample_times: list[int] | None = None,
        channels: int = 1,
        log_intermediate: bool = False,
        y: torch.Tensor | None = None,  # shape (batch,)
    ) -> torch.Tensor | list[torch.Tensor]:
        model.eval()
        batch_size = 1 if y is None else y.shape[0]

        ddim_steps = (
            self.noise_steps if t_sample_times is None else max(t_sample_times) + 1
        )
        step_indices = np.linspace(0, self.noise_steps - 1, ddim_steps, dtype=int)
        steps = list(reversed(step_indices.tolist()))

        x = torch.randn(
            batch_size, channels, self.img_size, self.img_size, device=self.device
        )
        intermediates = []

        for i, step in enumerate(steps):
            t = torch.full((batch_size,), step, device=self.device, dtype=torch.long)

            eps_pred = model(x, t, y=y) if y is not None else model(x, t)

            alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)
            alpha_bar_prev = (
                self.alpha_bar[steps[i + 1]].view(-1, 1, 1, 1)
                if i < len(steps) - 1
                else torch.ones_like(alpha_bar_t)
            )

            x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(
                alpha_bar_t
            )

            dir_xt = (
                torch.sqrt(
                    1 - alpha_bar_prev - eta**2 * (1 - alpha_bar_t / alpha_bar_prev)
                )
                * eps_pred
            )

            noise_term = (
                eta
                * torch.sqrt(1 - alpha_bar_t / alpha_bar_prev)
                * torch.sqrt(1 - alpha_bar_prev)
                / torch.sqrt(1 - alpha_bar_t)
                * torch.randn_like(x)
                if eta > 0
                else 0
            )

            x_prev = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt + noise_term

            if log_intermediate and t_sample_times and (i + 1) in t_sample_times:
                intermediates.append(self.transform_sampled_image(x_prev.clone()))

            x = x_prev

        final_image = self.transform_sampled_image(x)
        model.train()

        if log_intermediate and t_sample_times:
            return intermediates + [final_image]  # List[Tensor] of [B, C, H, W]
        return final_image  # Tensor [B, C, H, W]

    @staticmethod
    def transform_sampled_image(image: Tensor) -> Tensor:
        return (image.clamp(-1, 1) + 1) / 2
