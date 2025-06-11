import torch
import torch.optim as optim
import torchvision.utils
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.diffusion import Diffusion
from models.unet import DiffusionUNet
from utils.wandb import (
    finish_wandb,
    initialize_wandb,
    log_best_model,
    log_epoch_metrics,
    log_intermediate_steps,
    log_training_step,
    save_and_log_model_checkpoint,
)


def train(
    num_epochs: int,
    device: torch.device,
    dataloader: DataLoader,
    val_loader: DataLoader,
    learning_rate: float = 1e-3,
    use_wandb: bool = False,
):
    model = DiffusionUNet().to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    diffusion = Diffusion(img_size=28, device=device)

    if use_wandb:
        initialize_wandb(
            project="diffusion-project",
            config={
                "epochs": num_epochs,
                "lr": learning_rate,
                "model": "DiffusionUNet",
            },
        )

    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        train_bar = tqdm(
            dataloader, desc=f"Epoch {epoch}/{num_epochs} [Train]", leave=False
        )

        for images, _ in train_bar:
            images = images.to(device)
            images = images * 2 - 1

            optimizer.zero_grad()
            loss = diffusion.perform_training_step(model, images)
            loss_value = loss.item()
            epoch_loss += loss_value

            loss.backward()
            optimizer.step()
            train_bar.set_postfix(loss=loss_value)

            if use_wandb:
                log_training_step(loss_value)

        avg_epoch_loss = epoch_loss / len(dataloader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(
                val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]", leave=False
            )
            for images, _ in val_bar:
                images = images.to(device)
                images = images * 2 - 1
                loss = diffusion.perform_training_step(model, images)
                val_loss += loss.item()
                val_bar.set_postfix(loss=loss.item())

        avg_val_loss = val_loss / len(val_loader)

        # Generate 4 samples with intermediate steps
        samples_list = []
        with torch.no_grad():
            for _ in range(4):
                steps = diffusion.sample(
                    model,
                    t_sample_times=[1000, 800, 600, 400, 200, 0],
                    log_intermediate=True,
                )
                samples_list.extend(steps)

        samples_tensor = torch.cat(samples_list, dim=0)

        if use_wandb:
            log_epoch_metrics(epoch, avg_epoch_loss, avg_val_loss, samples_tensor)
            log_intermediate_steps(samples_list)
            save_and_log_model_checkpoint(model, epoch, avg_epoch_loss, avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            log_best_model(model, best_val_loss)

        print(
            f"Epoch [{epoch}/{num_epochs}] | "
            f"Train Loss: {avg_epoch_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}"
        )

    if use_wandb:
        finish_wandb()

    print("\nTraining complete.")
    return model
