import os
from typing import Optional

import torch
import torch.optim as optim
from tqdm import tqdm

from src.models.diffusion import Diffusion

# from src.models.unet import DiffusionUNet
from src.models.transformer_net import DiffusionUViT
from src.utils.wandb import (
    finish_wandb,
    initialize_wandb,
    log_best_model,
    log_epoch_metrics,
    log_intermediate_steps,
    log_training_step,
    save_and_log_model_checkpoint,
)


def train_one_epoch(model, dataloader, optimizer, diffusion, device, use_wandb):
    model.train()
    total_loss = 0.0

    for images, _ in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device) * 2 - 1  # Normalize to [-1, 1]
        optimizer.zero_grad()
        loss = diffusion.perform_training_step(model, images)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        total_loss += loss_val

        if use_wandb:
            log_training_step(loss_val)

    return total_loss / len(dataloader)


def validate(model, val_loader, diffusion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, _ in tqdm(val_loader, desc="Validating", leave=False):
            images = images.to(device) * 2 - 1
            loss = diffusion.perform_training_step(model, images)
            total_loss += loss.item()

    return total_loss / len(val_loader)


def log_samples(model, diffusion):
    samples_steps = []
    with torch.no_grad():
        for _ in range(4):
            steps = diffusion.sample(
                model,
                t_sample_times=[1000, 800, 600, 400, 200, 0],
                log_intermediate=True,
            )
            samples_steps.append(torch.cat(steps, dim=0))
    return samples_steps


def load_checkpoint(model, optimizer, checkpoint_path, device):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return 1, float("inf")

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint.get("epoch", 1) + 1
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    return epoch, best_val_loss


def train(
    num_epochs: int,
    device: torch.device,
    dataloader,
    val_loader,
    learning_rate: float = 1e-3,
    use_wandb: bool = False,
    checkpoint_path: Optional[str] = None,
):
    # model = DiffusionUNet().to(device)
    model = DiffusionUViT().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    start_epoch, best_val_loss = load_checkpoint(
        model, optimizer, checkpoint_path, device
    )
    diffusion = Diffusion(img_size=28, device=device)

    if use_wandb:
        initialize_wandb(
            project="diffusion-project",
            config={
                "epochs": num_epochs,
                "lr": learning_rate,
                # "model": "DiffusionUNet",
                "model": "DiffusionUViT",
            },
        )

    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_loss = train_one_epoch(
            model, dataloader, optimizer, diffusion, device, use_wandb
        )
        val_loss = validate(model, val_loader, diffusion, device)

        if use_wandb:
            log_epoch_metrics(epoch, train_loss, val_loss)
            log_intermediate_steps(log_samples(model, diffusion))

            save_and_log_model_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                best_val_loss=best_val_loss,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                log_best_model(
                    model=model,
                    optimizer=optimizer,
                    val_loss=best_val_loss,
                    epoch=epoch,
                )

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    if use_wandb:
        finish_wandb()

    print("\nTraining complete.")
    return model
