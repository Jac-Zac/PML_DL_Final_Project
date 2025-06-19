import random
from typing import Optional, Tuple

from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


class DiffusionMNIST(Dataset):
    def __init__(self, base_dataset: Dataset, max_timesteps: int = 1000):
        self.base_dataset = base_dataset
        self.max_timesteps = max_timesteps

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        t = random.randint(0, self.max_timesteps - 1)
        return image, t, label


def get_dataloaders(
    batch_size: int = 120,
    shuffle: bool = True,
    num_elements: Optional[int] = None,
    max_timesteps: int = 1000,
) -> Tuple[DataLoader, DataLoader]:
    """
    Load MNIST dataset and create data loaders that return (image, t, label).

    Parameters:
    - batch_size (int): Batch size for data loaders.
    - shuffle (bool, optional): Whether to shuffle the training data. Defaults to True.
    - num_elements (int, optional): Number of elements to retrieve from the dataset. Defaults to None.
    - max_timesteps (int): Maximum diffusion timestep to sample from. Defaults to 1000.

    Returns:
    - train_loader (DataLoader): Training data loader.
    - test_loader (DataLoader): Testing data loader.
    """

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Resize((28, 28)),
        ]
    )

    base_train = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    base_test = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    if num_elements is not None:
        base_train = Subset(base_train, range(min(num_elements, len(base_train))))
        base_test = Subset(base_test, range(min(num_elements, len(base_test))))

    train_dataset = DiffusionMNIST(base_train, max_timesteps=max_timesteps)
    test_dataset = DiffusionMNIST(base_test, max_timesteps=max_timesteps)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
