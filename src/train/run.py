#!/usr/bin/env python
import argparse
import os
import time

import torch
import wandb

from train.train import train
from utils.data import get_dataloaders
from utils.environment import get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train a simple diffusion model.")
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Training batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--project", type=str, default="ddpm-training", help="WandB project name"
    )
    parser.add_argument(
        "--tags", type=str, nargs="*", default=[], help="Tags for WandB run"
    )
    parser.add_argument("--name", type=str, default=None, help="Name for the WandB run")
    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)
    device = get_device()

    # Create checkpoint directory
    os.makedirs("checkpoints", exist_ok=True)

    # Initialize wandb
    if args.wandb and wandb is not None:
        wandb.init(
            project=args.project,
            name=args.name or f"ddpm-{time.strftime('%Y%m%d-%H%M%S')}",
            config=vars(args),
            tags=args.tags,
        )
        wandb.config.update({"device": str(device)})

    train_loader, val_loader = get_dataloaders(batch_size=args.batch_size)

    model = train(
        num_epochs=args.epochs,
        device=device,
        dataloader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate,
        use_wandb=args.wandb,
    )

    # Save final model
    save_path = os.path.join("checkpoints", "unet_trained.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # Log final model to wandb
    if args.wandb and wandb is not None:
        final_artifact = wandb.Artifact(
            "trained_model_final", type="model", metadata=dict(wandb.config)
        )
        final_artifact.add_file(save_path)
        wandb.log_artifact(final_artifact)
        wandb.finish()


if __name__ == "__main__":
    main()
