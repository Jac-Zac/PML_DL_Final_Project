import os
from typing import Optional

import torch
from tqdm import tqdm

from src.utils.environment import load_checkpoint
from src.utils.wandb import (
    initialize_wandb,
    log_best_model,
    log_epoch_metrics,
    log_sample_grid,
    log_training_step,
    save_and_log_model_checkpoint,
)


def train_one_epoch(model, dataloader, optimizer, method_instance, device, use_wandb):
    model.train()
    total_loss = 0.0

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device).mul_(2).sub_(1)  # In-place normalization to [-1, 1]
        y = labels.to(device)  # Class labels for conditioning

        optimizer.zero_grad(set_to_none=True)  # Slightly more efficient memory-wise
        loss = method_instance.perform_training_step(model, images, y=y)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        total_loss += loss_val

        if use_wandb:
            log_training_step(loss_val)

    return total_loss / max(1, len(dataloader))  # avoid ZeroDivisionError


def validate(model, val_loader, method_instance, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating", leave=False):
            images = images.to(device) * 2 - 1
            y = labels.to(device)

            loss = method_instance.perform_training_step(model, images, y=y)
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
    model_name: str = "unet",
    model_kwargs: Optional[dict] = None,
    method: str = "diffusion",
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

    # Only create best_model_state if resuming
    if checkpoint_path and os.path.exists(checkpoint_path):
        best_model_state = {
            "model": model,
            "optimizer": optimizer,
            "val_loss": best_val_loss,
            "epoch": start_epoch,
            # NOTE: Potentially you can also add the number of steps
        }

    # Select method
    if method == "diffusion":
        from src.models.diffusion import Diffusion

        method_instance = Diffusion(img_size=28, device=device)
    elif method == "flow":
        from src.models.flow import FlowMatching

        method_instance = FlowMatching(img_size=28, device=device)
    else:
        raise ValueError(f"Unsupported method: {method}")

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
            model, dataloader, optimizer, method_instance, device, use_wandb
        )
        val_loss = validate(model, val_loader, method_instance, device)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if use_wandb:
            log_epoch_metrics(epoch, train_loss, val_loss)
            log_sample_grid(model, method_instance, num_samples=5, num_timesteps=6)

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
