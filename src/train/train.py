from typing import Optional

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.utils.environment import load_checkpoint
from src.utils.wandb import (
    initialize_wandb,
    log_epoch_metrics,
    log_sample_grid,
    log_training_step,
    save_best_model_artifact,
)


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    method_instance,
    device,
    use_wandb,
    grad_clip: float = 1.0,
):
    model.train()
    total_loss = 0.0

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        # NOTE: Rescaling for between [-1, 1] for training stability
        images = images.to(device).mul_(2).sub_(1)
        y = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        loss = method_instance.perform_training_step(model, images, y=y)
        loss.backward()

        if grad_clip is not None:
            clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        loss_val = loss.item()
        total_loss += loss_val

        if use_wandb:
            log_training_step(loss_val)

    return total_loss / max(1, len(dataloader))


def validate(model, val_loader, method_instance, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating", leave=False):
            images = images.to(device).mul_(2).sub_(1)
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

    # Initialize scheduler ahead of load_checkpoint
    dummy_optimizer = torch.optim.Adam([torch.zeros(1)], lr=learning_rate)
    scheduler = CosineAnnealingLR(dummy_optimizer, T_max=num_epochs, eta_min=1e-5)

    model, optimizer, _, start_epoch, best_val_loss = load_checkpoint(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        device=device,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs={"lr": learning_rate},
        model_kwargs=model_kwargs,
        scheduler=scheduler,
    )

    # Re-create scheduler with real optimizer if not loaded
    if not hasattr(scheduler, "optimizer") or scheduler.optimizer is not optimizer:
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    # Initialize method instance
    if method == "diffusion":
        from src.models.diffusion import Diffusion

        method_instance = Diffusion(img_size=28, device=device)
    elif method == "flow":
        from src.models.flow import FlowMatching

        method_instance = FlowMatching(img_size=28, device=device)
    else:
        raise ValueError(f"Unsupported method: {method}")

    wandb_run = None
    if use_wandb:
        wandb_run = initialize_wandb(
            project="diffusion-project",
            config={
                "epochs": num_epochs,
                "lr": learning_rate,
                "model": model_name,
                "num_classes": model_kwargs.get("num_classes"),
                "method": method,
            },
        )

    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_loss = train_one_epoch(
            model, dataloader, optimizer, method_instance, device, use_wandb
        )
        val_loss = validate(model, val_loader, method_instance, device)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}"
        )

        # Check if this is the best model
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        if use_wandb:
            log_epoch_metrics(epoch, train_loss, val_loss, current_lr)
            log_sample_grid(model, method_instance, num_samples=5, num_timesteps=6)

            # Save best model artifact when we find a new best
            if is_best:
                save_best_model_artifact(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    val_loss=val_loss,
                    train_loss=train_loss,
                )

    if wandb_run:
        wandb_run.finish()

    print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")
    return model
