from typing import Optional, Tuple

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_dataloaders(
    batch_size: int = 120, shuffle: bool = True, num_elements: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Load MNIST dataset and create data loaders.

    Parameters:
    - batch_size (int): Batch size for data loaders.
    - shuffle (bool, optional): Whether to shuffle the training data. Defaults to True.
    - num_elements (int, optional): Number of elements to retrieve from the dataset. Defaults to None.

    Returns:
    - train_loader (DataLoader): Training data loader.
    - test_loader (DataLoader): Testing data loader.
    """

    # Load MNIST dataset
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            # NOTE: Let's keep the resize from the tutorial for now
            transforms.Resize((28, 28)),
        ]
    )
    train = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # If num_elements is specified, create subsets of the datasets
    if num_elements is not None:
        train_subset = Subset(train, range(min(num_elements, len(train))))
        test_subset = Subset(test, range(min(num_elements, len(test))))
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(test_subset, batch_size=batch_size)
    else:
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(test, batch_size=batch_size)

    return train_loader, test_loader
