import os

import torch
import torchvision.utils as vutils

import wandb


def initialize_wandb(project="diffusion-project", run_name=None, config=None):
    if wandb.run is not None:
        print(f"WandB already initialized with project: {wandb.run.project}")
        return
    wandb.login()  # will silently succeed if already logged in
    wandb.init(project=project, name=run_name, config=config)


def log_training_step(loss: float):
    wandb.log({"train/loss_step": loss})


def log_epoch_metrics(epoch, train_loss, val_loss, sample_images: torch.Tensor):
    grid = vutils.make_grid(sample_images, nrow=2, normalize=True, value_range=(-1, 1))
    wandb.log(
        {
            "train/loss_epoch": train_loss,
            "val/loss": val_loss,
            "samples": wandb.Image(grid.permute(1, 2, 0).cpu().numpy()),
            "epoch": epoch,
        }
    )


def log_intermediate_steps(sample_steps: list[torch.Tensor]):
    """Logs intermediate sampling images for visualization."""
    for idx, step_img in enumerate(sample_steps):
        img = vutils.make_grid(step_img, normalize=True, value_range=(-1, 1))
        wandb.log(
            {f"sample_step/step_{idx}": wandb.Image(img.permute(1, 2, 0).cpu().numpy())}
        )


def save_and_log_model_checkpoint(
    model, epoch: int, train_loss: float, val_loss: float, checkpoint_dir="checkpoints"
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pth")
    torch.save(model.state_dict(), path)

    artifact = wandb.Artifact(
        f"model-epoch-{epoch}",
        type="model",
        metadata={"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss},
    )
    artifact.add_file(path)
    wandb.log_artifact(artifact)


def log_best_model(model, val_loss: float, path="checkpoints/best_model.pth"):
    torch.save(model.state_dict(), path)
    wandb.summary["best_val_loss"] = val_loss


def finish_wandb():
    wandb.finish()
