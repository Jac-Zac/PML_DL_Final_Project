from typing import Optional

import torch
from torch import Tensor


def transform_sampled_image(image):
    image = (image.clamp(-1, 1) + 1) / 2  # rescale to [0, 1]
    return image


class Diffusion:
    def __init__(
        self,
        noise_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        img_size=64,
        device="cpu",
    ):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        self.img_size = img_size

        # set coefficients for noise schedule
        self.beta = self._prepare_noise_schedule().to(
            device
        )  # forward process variances (here we use a linear schedule)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def _prepare_noise_schedule(self):
        # linear schedule for beta
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def _sample_q(self, x, t):
        # Eq. 4 in the DDPM paper (Ho et al. 2020) after reparametrization
        # noisy images at time step t xt = x* sqrt(apha_bar) + sqrt(1-alpha_bar) * e
        # with e ~ N(0, I).

        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t])[:, None, None, None]
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar[t])[
            :, None, None, None
        ]

        # sample noise from standard normal distribution
        e = torch.randn(x.shape, device=self.device)

        # with reparametrization trick (q(xt|x0) = N(xt; x0*sqrt(alpha_bar), sqrt(1-alpha_bar)) we get the following
        # sample from q(xt|x0) = x0*sqrt(alpha_bar) + sqrt(1-alpha_bar) * e

        q_mean = sqrt_alpha_bar * x
        q_std = sqrt_one_minus_alpha_bar
        q_t = q_mean + q_std * e
        return q_t, e

    def _sample_timesteps(self, n):
        # return random integers between 1 and noise_steps of size n
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    @staticmethod
    def loss_simple(epsilon, epsilon_pred) -> torch.nn.MSELoss:
        # Eq. 14 in the DDPM paper - very simple loss aka MSE between predicted and true noise
        loss = torch.nn.MSELoss()
        return loss(epsilon, epsilon_pred)

    def perform_training_step(self, model, image):
        # Step 2, Algorithm 1: sample image x0 ~q(x0)
        x_0 = image.to(self.device)
        # Step 3, Algorithm 1: generate random timesteps
        t = self._sample_timesteps(x_0.shape[0]).to(self.device)
        # Step 4, Algorithm 1: sample noise for each timestep (epsilon)
        x_t, epsilon = self._sample_q(x_0, t)

        # Step 5, Algorithm 1: Take gradient descent step on
        # predict noise
        epsilon_pred = model.forward(x_t, t)

        # calculate loss (here we use a simple MSE loss)
        loss = self.loss_simple(epsilon, epsilon_pred)

        return loss

    @torch.no_grad()
    def sample_step(self, model, x_t, t):
        # Step 3, Algorithm 2: sample noise for next time step
        # Forward Pass (predict noise for current time step)
        predicted_noise = model(x_t, t)

        # get alpha and beta for current time step
        alpha = self.alpha[t][:, None, None, None]
        alpha_bar = self.alpha_bar[t][:, None, None, None]
        beta = self.beta[t][:, None, None, None]

        # Step 3, Algorithm 2: sample noise for next time step
        if t[0] > 1:
            z_noise = torch.randn_like(x_t)
        else:
            # last step, add no noise, otherwise it would get worse
            z_noise = torch.zeros_like(x_t)

        # Step 4, Algorithm 2: update x_t-1 (remove a little bit of noise)
        x_t_minus_1 = (
            1
            / torch.sqrt(alpha)
            * (x_t - ((1 - alpha) / (torch.sqrt(1 - alpha_bar))) * predicted_noise)
            + torch.sqrt(beta) * z_noise
        )

        return x_t_minus_1

    def sample(self, model, t_sample_times=None):
        # Following Algorithm 2 in the DDPM paper
        model.eval()
        # Step 1, Algorithm 2:
        # start with random noise (x_T)
        x_t = torch.randn(1, 1, self.img_size, self.img_size, device=self.device)

        # Step 2, Algorithm 2: go over all time steps in reverse order (from noise to image)
        sample_images = []

        for i in reversed(range(1, self.noise_steps)):
            # timestep encoding
            t = (torch.ones(1) * i).long().to(self.device)

            # Perform a sampling step
            x_t = self.sample_step(model, x_t, t)

            # save sampled images if requested:
            if t_sample_times and i in t_sample_times:
                sample_images.append(x_t)

        model.train()

        rescaled_images = [transform_sampled_image(image) for image in sample_images]
        return rescaled_images
