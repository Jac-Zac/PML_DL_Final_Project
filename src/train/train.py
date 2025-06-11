import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models.diffusion import Diffusion
from models.unet import DiffusionUNet


def train(
    num_epochs: int,
    device: torch.device,
    dataloader: DataLoader,
    learning_rate: float = 1e-3,
):
    """Main training loop for diffusion model.

    Args:
        num_epochs (int): Number of epochs to train.
        device (torch.device): Computation device (e.g. "cuda" or "cpu").
        dataloader (DataLoader): DataLoader for training data.
        learning_rate (float): Learning rate for the optimizer.
    """
    model = DiffusionUNet().to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    diffusion = Diffusion(img_size=28, device=device)
    total_steps = len(dataloader)

    for epoch in range(1, num_epochs + 1):
        for step, (images, _) in enumerate(dataloader, start=1):
            images = images.to(device)

            optimizer.zero_grad()
            loss = diffusion.perform_training_step(model, images)
            loss.backward()
            optimizer.step()

            print(
                f"Epoch [{epoch}/{num_epochs}] | Step [{step}/{total_steps}] | Loss: {loss.item():.4f}",
                end="\r",
            )

    print("\nTraining complete.")
    return model
