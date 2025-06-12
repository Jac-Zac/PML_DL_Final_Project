import os

import torch
import torch.optim as optim
from tqdm import tqdm

from src.models.diffusion import Diffusion
from src.models.unet import DiffusionUNet
from src.utils.wandb import (
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
    dataloader,
    val_loader,
    learning_rate: float = 1e-3,
    use_wandb: bool = False,
    checkpoint_path: str = None,
):
    model = DiffusionUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    start_epoch = 1
    best_val_loss = float("inf")

    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 1) + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))

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

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        train_loss = 0.0

        for images, _ in tqdm(
            dataloader, desc=f"Epoch {epoch}/{num_epochs} [Train]", leave=False
        ):
            images = images.to(device) * 2 - 1  # Normalize to [-1, 1]

            optimizer.zero_grad()
            loss = diffusion.perform_training_step(model, images)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            train_loss += loss_val
            if use_wandb:
                log_training_step(loss_val)

        avg_train_loss = train_loss / len(dataloader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, _ in tqdm(
                val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]", leave=False
            ):
                images = images.to(device) * 2 - 1
                loss = diffusion.perform_training_step(model, images)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # Generate samples for logging
        samples_steps = []
        with torch.no_grad():
            for _ in range(4):
                steps = diffusion.sample(
                    model,
                    t_sample_times=[1000, 800, 600, 400, 200, 0],
                    log_intermediate=True,
                )
                samples_steps.append(torch.cat(steps, dim=0))

        if use_wandb:
            log_epoch_metrics(epoch, avg_train_loss, avg_val_loss)
            log_intermediate_steps(samples_steps)

            # Save full checkpoint (model + optimizer + meta)
            save_and_log_model_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
                best_val_loss=best_val_loss,
            )

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            log_best_model(
                model=model,
                optimizer=optimizer,
                val_loss=best_val_loss,
                epoch=epoch,
            )

        print(
            f"Epoch [{epoch}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )

    if use_wandb:
        finish_wandb()

    print("\nTraining complete.")
    return model
