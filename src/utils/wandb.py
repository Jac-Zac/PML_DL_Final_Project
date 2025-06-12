import os

import torch
import torchvision.utils as vutils
from dotenv import load_dotenv

import wandb


def initialize_wandb(project="diffusion-project", run_name=None, config=None):
    if wandb.run is not None:
        print(f"WandB already initialized with project: {wandb.run.project}")
        return

    # Load environment variables from .env if present
    load_dotenv()

    # Get the W&B API key from environment variable
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key is None:
        raise ValueError("WANDB_API_KEY environment variable not set")

    # Login using the API key
    wandb.login(key=wandb_api_key)

    wandb.init(project=project, name=run_name, config=config)


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


def log_intermediate_steps(samples_steps: list[torch.Tensor]):
    """
    samples_steps: list of tensors [timesteps * C * H * W] per sample
    Each tensor shape: (num_timesteps * C * H * W), stacked along dim=0 for timesteps.

    Goal: Create a grid where each row is a sample, each column is a timestep.
    """

    # Stack all samples along new batch dim: shape (num_samples, num_timesteps, C, H, W)
    stacked = torch.stack(samples_steps)  # shape: [num_samples, num_timesteps*C*H*W]

    # Reshape each sample from flattened timesteps to (num_timesteps, C, H, W)
    num_samples = len(samples_steps)
    num_timesteps = samples_steps[0].shape[0]  # number of timesteps
    C, H, W = samples_steps[0].shape[1:]  # channels, height, width

    # Reshape each sample: (num_timesteps, C, H, W)
    stacked = stacked.view(num_samples, num_timesteps, C, H, W)

    # For each timestep, gather the images for all samples -> (num_timesteps, num_samples, C, H, W)
    stacked = stacked.permute(1, 0, 2, 3, 4)

    # Create grid per timestep (samples in row)
    grids = []
    for t in range(num_timesteps):
        # Make grid of samples for timestep t (batch = num_samples)
        grid_t = vutils.make_grid(
            stacked[t], nrow=num_samples, normalize=True, value_range=(-1, 1)
        )
        grids.append(grid_t)

    # Stack grids horizontally (side by side columns = timesteps)
    final_grid = torch.cat(grids, dim=2)  # concat on width

    wandb.log(
        {
            "sampling_intermediate_steps": wandb.Image(
                final_grid.permute(1, 2, 0).cpu().numpy()
            )
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


def finish_wandb():
    wandb.finish()
