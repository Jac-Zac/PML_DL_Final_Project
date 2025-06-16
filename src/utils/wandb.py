import os

import torch
import torchvision.utils as vutils
from dotenv import load_dotenv

import wandb


def initialize_wandb(project="diffusion-project", run_name=None, config=None):
    if wandb.run is not None:
        print(f"WandB already initialized with project: {wandb.run.project}")
        return wandb.run  # return the existing run if already initialized

    # Load environment variables from .env if present
    load_dotenv()

    # Get the W&B API key from environment variable
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key is None:
        raise ValueError("WANDB_API_KEY environment variable not set")

    # Login using the API key
    wandb.login(key=wandb_api_key)

    # Initialize W&B run and return it
    run = wandb.init(project=project, name=run_name, config=config)
    return run


def log_training_step(loss: float):
    wandb.log({"train/loss_step": loss})


def log_epoch_metrics(epoch, train_loss, val_loss):
    wandb.log(
        {
            "train/loss_epoch": train_loss,
            "val/loss": val_loss,
            "epoch": epoch,
        }
    )


def save_and_log_model_checkpoint(
    model,
    optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    best_val_loss: float,
    checkpoint_dir="checkpoints",
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pth")

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
        },
        path,
    )

    artifact = wandb.Artifact(
        f"model-epoch-{epoch}",
        type="model",
        metadata={"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss},
    )
    artifact.add_file(path)
    wandb.log_artifact(artifact)


def log_best_model(
    model, optimizer, val_loss: float, epoch: int, path="checkpoints/best_model.pth"
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": val_loss,
        },
        path,
    )
    wandb.summary["best_val_loss"] = val_loss


def log_sample_grid(
    model, diffusion, num_samples=5, num_timesteps=6, max_timesteps=1000
):
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
        y=y,  # pass batch of labels
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
