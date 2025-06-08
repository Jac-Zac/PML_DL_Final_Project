#!/usr/bin/env python

from utils.data import get_dataloaders
from utils.environment import get_device, set_seed

# NOTE: Example class ModelArgs

# Define the dataclass for training arguments
# @dataclass
# class MLPTrainingArgs:
#     batch_size: int = 128
#     epochs: int = 5
#     learning_rate: float = 1e-3
#     hidden_sizes: Sequence[int] = ((64),)
#     optimizer: str = "Adam"  # Default optimizer
#     criterion: str = "CrossEntropyLoss"  # Default loss criterion
#     seed: int = 42


def main():
    set_seed(1337)
    device = get_device()
    # Get only the train dataloader for mnist
    train_dataloader, _ = get_dataloaders()

    train(num_epochs=10, device=device, dataloader=train_dataloader)


if __name__ == "__main__":
    main()
