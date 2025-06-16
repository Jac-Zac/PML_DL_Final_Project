from typing import Optional

import torch
import torch.optim as optim
from tqdm import tqdm

from src.models.diffusion import Diffusion
from src.models.unet import DiffusionUNet
from src.utils.environment import load_checkpoint
from src.utils.wandb import (
    initialize_wandb,
    log_best_model,
    log_epoch_metrics,
    log_sample_grid,
    log_training_step,
    save_and_log_model_checkpoint,
)


def train_one_epoch(model, dataloader, optimizer, diffusion, device, use_wandb):
    model.train()
    total_loss = 0.0

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device) * 2 - 1  # Normalize to [-1, 1]
        y = labels.to(device)  # Class labels for conditioning

        optimizer.zero_grad()
        loss = diffusion.perform_training_step(model, images, y=y)
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
        for images, labels in tqdm(val_loader, desc="Validating", leave=False):
            images = images.to(device) * 2 - 1
            y = labels.to(device)

            loss = diffusion.perform_training_step(model, images, y=y)
            total_loss += loss.item()

    return total_loss / len(val_loader)


def train(
    num_epochs: int,
    device: torch.device,
    dataloader,
    val_loader,
    learning_rate: float = 1e-3,
    use_wandb: bool = False,
    checkpoint_path: Optional[str] = None,
    model_name: str = "unet",  # new param to specify model
    model_kwargs: dict = None,  # kwargs for model init
):
    model_kwargs = model_kwargs or {}

    model, optimizer, start_epoch, best_val_loss = load_checkpoint(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        device=device,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs={"lr": learning_rate},
        model_kwargs=model_kwargs,
    )

    diffusion = Diffusion(img_size=28, device=device)

    if use_wandb:
        wandb_run = initialize_wandb(
            project="diffusion-project",
            config={
                "epochs": num_epochs,
                "lr": learning_rate,
                "model": model_name,
                "num_classes": model_kwargs["num_classes"],
            },
        )

    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_loss = train_one_epoch(
            model, dataloader, optimizer, diffusion, device, use_wandb
        )
        val_loss = validate(model, val_loader, diffusion, device)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if use_wandb:
            log_epoch_metrics(epoch, train_loss, val_loss)
            log_sample_grid(model, diffusion, num_samples=5, num_timesteps=6)

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
                best_model_state = {
                    "model": model,
                    "optimizer": optimizer,
                    "val_loss": best_val_loss,
                    "epoch": epoch,
                }

    # Finalize
    if use_wandb and best_val_loss is not None:
        log_best_model(**best_model_state)
        wandb_run.finish()

    print("\nTraining complete.")
    return model
