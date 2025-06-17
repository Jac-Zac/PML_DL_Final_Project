from typing import Optional

import torch
from torch import Tensor, nn


class FlowMatching:
    def __init__(self, img_size: int = 64, device: torch.device = torch.device("cpu")):
        self.img_size = img_size
        self.device = device

    def _sample_timesteps(self, batch_size: int) -> Tensor:
        return torch.rand(batch_size, device=self.device)

    def perform_training_step(
        self,
        model: nn.Module,
        x0: Tensor,  # noise ~ N(0,I)
        x1: Tensor,  # data in [-1,1]
        y: Optional[Tensor] = None,
    ) -> Tensor:
        B = x0.size(0)
        t = self._sample_timesteps(B)
        t4 = t.view(-1, 1, 1, 1)
        x_t = (1 - t4) * x0 + t4 * x1  # linear OT path

        # True velocity & normalization
        u = x1 - x0
        norm = u.flatten(1).norm(dim=1, keepdim=True).view(-1, 1, 1, 1).clamp(min=1e-6)
        u = u / norm

        v = model(x_t, t, y=y)
        assert v.shape == u.shape

        # Time-weighted MSE loss
        w = t4.pow(2)
        return (w * (v - u).pow(2)).mean()

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        x_init: Optional[Tensor] = None,
        steps: int = 100,
        y: Optional[Tensor] = None,
        log_intermediate: bool = False,
        t_sample_times: Optional[list[int]] = None,
    ) -> list[Tensor]:
        model.eval()
        B = (
            y.shape[0]
            if y is not None
            else (x_init.shape[0] if x_init is not None else 1)
        )
        C = x_init.shape[1] if x_init is not None else 1
        x_t = (
            x_init.to(self.device)
            if x_init is not None
            else torch.randn(B, C, self.img_size, self.img_size, device=self.device)
        )

        results = []
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((B,), i / steps, device=self.device)
            v = model(x_t, t, y=y)
            x_t = x_t + v * dt
            if log_intermediate and t_sample_times and i in t_sample_times:
                results.append(self.transform_sampled_image(x_t.clone()))

        results.append(self.transform_sampled_image(x_t))
        return results

    @staticmethod
    def transform_sampled_image(image: Tensor) -> Tensor:
        return (image.clamp(-1, 1) + 1) / 2
