import torch
from torch import Tensor, nn


class FlowMatching:
    """
    Implements a flow matching training and sampling class similar in
    style to your Diffusion class.
    """

    def __init__(
        self,
        img_size: int = 64,
        device: torch.device = torch.device("cpu"),
    ):
        self.img_size = img_size
        self.device = device

    def _sample_timesteps(self, batch_size: int) -> Tensor:
        # Sample continuous timesteps uniformly in [0, 1]
        return torch.rand(batch_size, device=self.device)

    def _interpolate(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        # Linear interpolation x(t) = (1 - t)*x0 + t*x1
        t = t.view(-1, 1, 1, 1)  # broadcast over spatial dims
        return (1 - t) * x0 + t * x1

    def _velocity(self, x0: Tensor, x1: Tensor) -> Tensor:
        # Velocity of linear path = (x1 - x0), independent of t
        return x1 - x0

    @staticmethod
    def loss_simple(v_true: Tensor, v_pred: Tensor) -> Tensor:
        return nn.functional.mse_loss(v_pred, v_true)

    def perform_training_step(
        self,
        model: nn.Module,
        x0: Tensor,
        x1: Tensor,
        y: Tensor | None = None,
    ) -> Tensor:
        """
        Training step:
        - sample t ~ Uniform[0,1]
        - compute interpolated x(t)
        - compute true velocity (x1 - x0)
        - predict velocity from model(x(t), t, y)
        - compute MSE loss
        """
        batch_size = x0.size(0)
        t = self._sample_timesteps(batch_size)
        x_t = self._interpolate(x0, x1, t)
        v_true = self._velocity(x0, x1)

        # model expects t scaled to [0, timesteps] int or normalized float?
        # Your UNet takes float t for embedding, so float in [0,1] is fine
        v_pred = model(x_t, t, y=y)

        return self.loss_simple(v_true, v_pred)

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        x_init: Tensor | None = None,
        steps: int = 1000,
        y: Tensor | None = None,
        log_intermediate: bool = False,
        t_sample_times: list[int] | None = None,
    ) -> list[Tensor]:
        """
        Generate samples by integrating backward in time from x1 to x0.
        Since flow matching uses velocity fields v(x,t), use Euler integration:

        x_{t-dt} = x_t - v_pred(x_t, t)*dt

        Args:
            model: velocity field predictor
            x_init: starting point at t=1, if None, sample noise ~ N(0,1)
            steps: number of integration steps
            y: optional conditioning labels
            log_intermediate: whether to log intermediate images
            t_sample_times: indices of steps at which to log intermediate images
        Returns:
            list of sampled images (including final)
        """

        model.eval()

        batch_size = 1 if y is None else y.shape[0]
        channels = x_init.shape[1] if x_init is not None else 1

        if x_init is None:
            x_t = torch.randn(
                batch_size, channels, self.img_size, self.img_size, device=self.device
            )
        else:
            x_t = x_init.to(self.device)

        intermediates = []

        dt = 1.0 / steps
        times = torch.linspace(1, 0, steps + 1, device=self.device)  # from 1 to 0

        for i in range(steps):
            t = times[i].expand(batch_size)  # current time t in [0,1]
            v = model(x_t, t, y=y)
            x_t = x_t - v * dt  # Euler step backward in time

            if log_intermediate and t_sample_times and i in t_sample_times:
                intermediates.append(self.transform_sampled_image(x_t.clone()))

        final_image = self.transform_sampled_image(x_t.clone())

        model.train()

        return intermediates + [final_image]

    @staticmethod
    def transform_sampled_image(image: Tensor) -> Tensor:
        # Assume images are roughly in [-1, 1] range
        return (image.clamp(-1, 1) + 1) / 2
