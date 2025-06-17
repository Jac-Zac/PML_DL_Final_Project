import argparse
import os

from src.train.train import train
from src.utils.data import get_dataloaders
from src.utils.environment import get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train DDPM")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to checkpoint"
    )
    parser.add_argument(
        "--model-name", type=str, default="unet", help="Model name to use from registry"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="diffusion",
        choices=["diffusion", "flow"],
        help="Training method: diffusion or flow matching",
    )
    # Optional: you could add JSON string argument for model kwargs if you want to get fancy
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    os.makedirs("checkpoints", exist_ok=True)

    train_loader, val_loader = get_dataloaders(
        # batch_size=args.batch_size, num_elements=100
        batch_size=args.batch_size
    )

    # Get number of classes from dataset and add to model_kwargs ...
    # HACK: We keep it as 10 for now hard coded
    num_classes = 10
    model_kwargs = {"num_classes": num_classes}

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
