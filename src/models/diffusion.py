import torch
from torch import Tensor, nn


class Diffusion:
    """
    Implements the core DDPM diffusion process (Ho et al., 2020).
    Handles forward (noise) and reverse (denoise) sampling.
    """

    def __init__(
        self,
        noise_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        img_size: int = 64,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initializes diffusion schedule and parameters.
        """
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        # Linearly spaced noise schedule (used as variance terms)
        self.beta = self._prepare_noise_schedule().to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(
            self.alpha, dim=0
        )  # cumulative product of alphas

    def _prepare_noise_schedule(self) -> Tensor:
        # Create a linear beta schedule over all noise steps
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def _sample_timesteps(self, batch_size: int) -> Tensor:
        # Random timesteps ∈ [1, T) per sample
        return torch.randint(1, self.noise_steps, (batch_size,), device=self.device)

    def _sample_q(self, x_0: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """
        Forward process: q(x_t | x_0) = Normal(sqrt(alpha_bar) * x_0, (1 - alph_bar) * I)
        Adds noise to clean images to simulate the forward diffusion process.
        """

        # We reshape scalars to broadcast over 4D input (N, C, H, W)
        sqrt_alpha_bar = self.alpha_bar[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = (1 - self.alpha_bar[t]).sqrt().view(-1, 1, 1, 1)

        # Sample Gaussian noise
        noise = torch.randn_like(x_0)

        # Forward noising: x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * noise
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        return x_t, noise

    @staticmethod
    def loss_simple(epsilon: Tensor, epsilon_pred: Tensor) -> Tensor:
        # Simple MSE loss between predicted and ground-truth noise (ε)
        return nn.functional.mse_loss(epsilon_pred, epsilon)

    def perform_training_step(self, model: nn.Module, x_0: Tensor) -> Tensor:
        """
        Applies forward noising and computes loss between predicted and real noise.
        Corresponds to Algorithm 1 (steps 1–5) from DDPM.
        """
        x_0 = x_0.to(self.device)
        t = self._sample_timesteps(x_0.size(0))  # Step 3: sample t_i ~ Uniform(1, T)
        x_t, noise = self._sample_q(x_0, t)  # Step 4: sample x_t ~ q(x_t | x_0)
        noise_pred = model(x_t, t)  # Step 5: predict noise ε_θ(x_t, t)

        return self.loss_simple(noise, noise_pred)

    @torch.no_grad()
    def sample_step(self, model: nn.Module, x_t: Tensor, t: Tensor) -> Tensor:
        """
        Reverse process step: approximates x_{t-1} from x_t using ε_θ prediction.
        """
        # Reshape scalars for broadcasting over image tensor
        beta = self.beta[t].view(-1, 1, 1, 1)
        alpha = self.alpha[t].view(-1, 1, 1, 1)
        alpha_bar = self.alpha_bar[t].view(-1, 1, 1, 1)

        # Predict noise at timestep t
        noise_pred = model(x_t, t)

        # For t > 1: sample noise z ~ N(0, I), otherwise set z = 0
        z = torch.randn_like(x_t) if t[0] > 1 else torch.zeros_like(x_t)

        # DDPM reverse update rule:
        # x_{t-1} = 1/√α * (x_t - (1-α)/√(1-ᾱ) * ε_θ) + √β * z
        x_prev = (1 / alpha.sqrt()) * (
            x_t - ((1 - alpha) / (1 - alpha_bar).sqrt()) * noise_pred
        ) + beta.sqrt() * z

        return x_prev

    @torch.no_grad()
    def sample(self, model, t_sample_times=None, channels=1, log_intermediate=False):
        """
        Full image sampling from noise using the learned denoising model.
        Implements Algorithm 2 in DDPM paper (Ho et al., 2020).

        Args:
            model: Trained noise prediction model.
            t_sample_times: (Optional) List of timesteps at which to save intermediate outputs.
            channels: Image channels (default 1 for grayscale).
            log_intermediate: Enable logging of intermediate results

        Returns:
            Final sampled image tensor (rescaled for visualization).
        """
        model.eval()
        x_t = torch.randn(1, channels, self.img_size, self.img_size, device=self.device)

        intermediate_images = []
        for i in reversed(range(1, self.noise_steps)):
            t = torch.full((1,), i, device=self.device, dtype=torch.long)
            x_t = self.sample_step(model, x_t, t)

            if log_intermediate and (t_sample_times and i in t_sample_times):
                intermediate_images.append(self.transform_sampled_image(x_t.clone()))

        final_image = self.transform_sampled_image(x_t.clone())
        model.train()

        if log_intermediate and t_sample_times:
            return intermediate_images + [final_image]
        return final_image

    @staticmethod
    def transform_sampled_image(image: Tensor) -> Tensor:
        """
        Rescales image pixel values from [-1, 1] to [0, 1] for display or saving.
        Clamping prevents overflow artifacts.
        """
        return (image.clamp(-1, 1) + 1) / 2
