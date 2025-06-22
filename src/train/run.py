import argparse
import os

from src.train.train import train
from src.utils.data import get_dataloaders
from src.utils.environment import get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DDPM-style model")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to checkpoint"
    )
    parser.add_argument(
        "--model-name", type=str, default="unet", help="Model name from registry"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="diffusion",
        choices=["diffusion", "flow"],
        help="Training method: diffusion or flow matching",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    os.makedirs("checkpoints", exist_ok=True)

    # Returns DataLoaders that yield (image, timestep, label)
    train_loader, val_loader = get_dataloaders(batch_size=args.batch_size)

    # HACK: Hard-coded number of classes (MNIST = 10)
    model_kwargs = {"num_classes": 10}

    _ = train(
        num_epochs=args.epochs,
        device=device,
        dataloader=train_loader,
        val_loader=val_loader,
        learning_rate=args.lr,
        use_wandb=True,
        checkpoint_path=args.checkpoint,
        model_name=args.model_name,
        model_kwargs=model_kwargs,
        method=args.method,
    )


if __name__ == "__main__":
    main()
