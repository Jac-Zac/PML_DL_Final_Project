import argparse
import os

from src.train.train import train
from src.utils.data import get_dataloaders
from src.utils.environment import get_device, set_seed
from src.utils.wandb import finish_wandb, initialize_wandb


def parse_args():
    parser = argparse.ArgumentParser(description="Train DDPM")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to checkpoint"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    os.makedirs("checkpoints", exist_ok=True)
    use_wandb = False

    train_loader, val_loader = get_dataloaders(batch_size=args.batch_size)
    _ = train(
        num_epochs=args.epochs,
        device=device,
        dataloader=train_loader,
        val_loader=val_loader,
        learning_rate=args.lr,
        # use_wandb=True,
        use_wandb=use_wandb,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()
