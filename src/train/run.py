#!/usr/bin/env python
import argparse
import os

import torch

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
    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)
    device = get_device()

    train_loader, _ = get_dataloaders(batch_size=args.batch_size)

    model = train(
        num_epochs=args.epochs,
        device=device,
        dataloader=train_loader,
        learning_rate=args.learning_rate,
    )

    # Save the model's state_dict
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "unet_trained.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
