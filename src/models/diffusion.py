import torch
from torch import Tensor, nn


class Diffusion:
    """
    Implements the DDPM (Denoising Diffusion Probabilistic Models) process
    as proposed by Ho et al., 2020. Supports both forward (noising) and
    reverse (denoising) diffusion steps.
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
        """Generates a linear noise schedule (beta)."""
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def _sample_timesteps(self, batch_size: int) -> Tensor:
        """Samples a random timestep for each element in the batch."""
        return torch.randint(1, self.noise_steps, (batch_size,), device=self.device)

    def _sample_q(self, x_0: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """
        Simulates forward diffusion: q(x_t | x_0) = N(√ᾱ_t x_0, (1-ᾱ_t)I).
        """
        sqrt_alpha_bar = self.alpha_bar[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = (1 - self.alpha_bar[t]).sqrt().view(-1, 1, 1, 1)
        noise = torch.randn_like(x_0)
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        return x_t, noise

    @staticmethod
    def loss_simple(epsilon: Tensor, epsilon_pred: Tensor) -> Tensor:
        """Computes the MSE loss between predicted and actual noise."""
        return nn.functional.mse_loss(epsilon_pred, epsilon)

    def perform_training_step(self, model: nn.Module, x_0: Tensor) -> Tensor:
        """
        Executes a single DDPM training step:
        - Samples timestep
        - Noises image
        - Predicts noise
        - Computes loss
        """
        x_0 = x_0.to(self.device)
        t = self._sample_timesteps(x_0.size(0))
        x_t, noise = self._sample_q(x_0, t)
        noise_pred = model(x_t, t)
        return self.loss_simple(noise, noise_pred)

    @torch.no_grad()
    def sample_step(self, model: nn.Module, x_t: Tensor, t: Tensor) -> Tensor:
        """
        Computes one reverse sampling step to obtain x_{t-1} from x_t.
        """
        beta_t = self.beta[t].view(-1, 1, 1, 1)
        alpha_t = self.alpha[t].view(-1, 1, 1, 1)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)

        noise_pred = model(x_t, t)
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
    ) -> Tensor | list[Tensor]:
        """
        Generates a sample image using reverse diffusion (Algorithm 2 in DDPM).

        Args:
            model: Trained noise prediction model.
            t_sample_times: Timesteps at which intermediate images should be saved.
            channels: Number of image channels (1 = grayscale, 3 = RGB).
            log_intermediate: If True, returns intermediate images.

        Returns:
            Final (or list of) sampled image(s), rescaled to [0, 1].
        """
        model.eval()
        x_t = torch.randn(1, channels, self.img_size, self.img_size, device=self.device)
        intermediates = []

        for i in reversed(range(1, self.noise_steps)):
            t = torch.full((1,), i, device=self.device, dtype=torch.long)
            x_t = self.sample_step(model, x_t, t)

            if log_intermediate and t_sample_times and i in t_sample_times:
                intermediates.append(self.transform_sampled_image(x_t.clone()))

        final_image = self.transform_sampled_image(x_t.clone())
        model.train()

        if log_intermediate and t_sample_times:
            return intermediates + [final_image]
        return final_image

    @staticmethod
    def transform_sampled_image(image: Tensor) -> Tensor:
        """Rescales image tensor values from [-1, 1] to [0, 1]."""
        return (image.clamp(-1, 1) + 1) / 2
