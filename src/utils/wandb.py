import os

import torch
import torchvision.utils as vutils
from dotenv import load_dotenv

import wandb


def initialize_wandb(project="diffusion-project", run_name=None, config=None):
    if wandb.run is not None:
        print(f"WandB already initialized with project: {wandb.run.project}")
        return wandb.run

    # Load environment variables from .env if present
    load_dotenv()

    # Get the W&B API key from environment variable
    wandb_api_key = os.getenv("WANDB_API_KEY")

    # If the API key is not set, ask the user to input it manually
    if wandb_api_key is None:
        wandb_api_key = input(
            "WANDB_API_KEY environment variable not set. Please enter your WandB API key: "
        ).strip()
        if not wandb_api_key:
            raise ValueError("WandB API key is required to proceed.")

    # Login using the API key
    wandb.login(key=wandb_api_key)

    # Initialize W&B run and return it
    run = wandb.init(project=project, name=run_name, config=config)
    return run


def log_training_step(loss: float):
    wandb.log({"train/loss_step": loss})


def log_epoch_metrics(epoch, train_loss, val_loss, learning_rate):
    wandb.log(
        {
            "train/loss_epoch": train_loss,
            "val/loss": val_loss,
            "epoch": epoch,
            "learning_rate": learning_rate,
        }
    )


def save_best_model_artifact(
    model,
    optimizer,
    scheduler,
    epoch: int,
    val_loss: float,
    train_loss: float,
    checkpoint_dir="checkpoints",
):
    """Save best model and log as special 'best-model' artifact to wandb."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, "best_model.pth")

    # Save the checkpoint
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": val_loss,
            "train_loss": train_loss,
        },
        path,
    )

    # Create and log the best model artifact
    artifact = wandb.Artifact(
        "best-model",
        type="model",
        description=f"Best model checkpoint from epoch {epoch}",
        metadata={
            "epoch": epoch,
            "best_val_loss": val_loss,
            "train_loss": train_loss,
            "learning_rate": scheduler.get_last_lr()[0],
        },
    )
    artifact.add_file(path, name="best_model.pth")
    wandb.log_artifact(artifact)

    # Update wandb summary with best metrics
    wandb.summary["best_val_loss"] = val_loss
    wandb.summary["best_val_epoch"] = epoch
    wandb.summary["best_train_loss"] = train_loss
    wandb.summary["best_val_lr"] = scheduler.get_last_lr()[0]

    print(f"New best model saved! Epoch {epoch}, Val Loss: {val_loss:.4f}")


def log_sample_grid(
    model, diffusion, num_samples=5, num_timesteps=6, max_timesteps=1000
):
    """Generate and log sample grid showing diffusion process."""
    t_sample_times = torch.linspace(
        max_timesteps, 0, steps=num_timesteps, dtype=torch.int32
    ).tolist()

    # Create batch of conditioning labels [0, 1, 2, ..., num_samples-1]
    y = torch.arange(num_samples, device=diffusion.device)

    # Sample all at once in batch
    all_samples_grouped = diffusion.sample(
        model,
        t_sample_times=t_sample_times,
        log_intermediate=True,
        y=y,
    )
    # all_samples_grouped shape: (T, B, C, H, W)

    # Rearrange to (B, T, C, H, W)
    stacked = torch.stack(all_samples_grouped)  # (T, B, C, H, W)
    permuted = stacked.permute(1, 0, 2, 3, 4)  # (B, T, C, H, W)

    # For each sample in batch, create a horizontal grid of its timesteps
    rows = []
    for sample_idx in range(num_samples):
        sample_images = permuted[sample_idx]  # (T, C, H, W)
        row_grid = vutils.make_grid(
            sample_images, nrow=num_timesteps, normalize=True, value_range=(-1, 1)
        )
        rows.append(row_grid)

    # Stack rows vertically
    final_grid = torch.cat(rows, dim=1)

    # Log to wandb
    wandb.log(
        {
            "sampling_intermediate_steps": wandb.Image(
                final_grid.permute(1, 2, 0).cpu().numpy()
            )
        }
    )
