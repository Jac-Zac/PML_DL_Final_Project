import os

import torch
import torch.optim as optim
from torchvision.utils import make_grid, save_image

from model import Diffusion, LightweightUNet


def train(num_epochs: int, device: str, dataloader: torch.utils.data.DataLoader):
    os.makedirs("diffusion_samples", exist_ok=True)
    os.makedirs("diffusion_process", exist_ok=True)
    model = LightweightUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    diffusion = Diffusion(img_size=28, device=device)

    length = len(dataloader)

    for epoch in range(num_epochs):

        for i, (image, _) in enumerate(dataloader):
            optimizer.zero_grad()
            loss = diffusion.perform_training_step(model, image)
            loss.backward()
            optimizer.step()

            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{length}], Loss: {loss.item():.4f}",
                end="\r",
            )

        # plots
        t_sample_times = [
            i for i in range(1, diffusion.noise_steps, 50)
        ]  # Sample every 50 steps
        sampled_diffusion_steps = diffusion.sample(model, t_sample_times=t_sample_times)
        sampled_diffusion_steps = torch.cat(
            sampled_diffusion_steps, dim=0
        )  # [20, 1, 28, 28], range 0, 255
        sampled_diffusion_steps = make_grid(
            sampled_diffusion_steps, nrow=len(t_sample_times), normalize=False
        )
        save_image(
            sampled_diffusion_steps,
            f"diffusion_process/diffusion_steps_epoch_{epoch + 1}.png",
        )
        sampled_images = []
        for i in range(16):
            sampled_images.append(diffusion.sample(model, t_sample_times=[1])[0])
        # save the sampled images
        sampled_images = torch.cat(sampled_images, dim=0)
        sampled_images = make_grid(sampled_images, nrow=4, normalize=False)
        save_image(
            sampled_images, f"diffusion_samples/sampled_images_epoch_{epoch + 1}.png"
        )
