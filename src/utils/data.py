from typing import Any, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


def get_transforms(dataset_name: str, train: bool = True):
    if dataset_name == "MNIST":
        mean, std = (0.1307,), (0.3081,)
    elif dataset_name == "FashionMNIST":
        mean, std = (0.2860,), (0.3530,)
    elif dataset_name == "KMNIST":
        mean, std = (0.1918,), (0.3470,)  # Approximate KMNIST mean/std
    else:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Use 'MNIST', 'FashionMNIST', or 'KMNIST'."
        )

    base_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.Resize((28, 28)),
    ]

    if train:
        augmentation = [
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ]
        return transforms.Compose(augmentation + base_transforms)
    else:
        return transforms.Compose(base_transforms)


def get_dataloaders(
    batch_size: int = 120,
    shuffle: bool = True,
    dataset_name: str = "MNIST",
    num_elements: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Load MNIST, FashionMNIST, or KMNIST dataset and return (image, label) DataLoaders.
    """
    if dataset_name == "MNIST":
        DatasetClass = datasets.MNIST
    elif dataset_name == "FashionMNIST":
        DatasetClass = datasets.FashionMNIST
    elif dataset_name == "KMNIST":
        DatasetClass = datasets.KMNIST
    else:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Use 'MNIST', 'FashionMNIST', or 'KMNIST'."
        )

    transform = get_transforms(
        dataset_name, train=False
    )  # No augmentation for these loaders

    train_data = DatasetClass(
        root="./data", train=True, download=True, transform=transform
    )
    test_data = DatasetClass(
        root="./data", train=False, download=True, transform=transform
    )

    if num_elements is not None:
        train_data = Subset(train_data, range(min(num_elements, len(train_data))))
        test_data = Subset(test_data, range(min(num_elements, len(test_data))))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


class DiffusionDataset(Dataset):
    def __init__(self, base_dataset: Dataset, max_timesteps: int = 1000):
        self.base_dataset = base_dataset
        self.max_timesteps = max_timesteps
        self.betas = torch.linspace(1e-4, 0.02, max_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        image, label = self.base_dataset[idx]
        image = image * 2.0 - 1.0  # scale to [-1,1]

        t = torch.randint(low=0, high=self.max_timesteps, size=(1,))
        a = self.alphas_cumprod[t]
        noise = torch.randn_like(image)

        sqrt_a = a.sqrt()
        sqrt_one_minus_a = (1 - a).sqrt()

        if image.dim() > 1:
            sqrt_a = sqrt_a.view(-1, *([1] * (image.dim() - 1)))
            sqrt_one_minus_a = sqrt_one_minus_a.view(-1, *([1] * (image.dim() - 1)))

        x_t = image * sqrt_a + noise * sqrt_one_minus_a

        return x_t, t.squeeze(0), noise, label


class FlowMatchingDataset(Dataset):
    def __init__(self, base_dataset: Dataset):
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        image, label = self.base_dataset[idx]
        x1 = image * 2.0 - 1.0  # scale to [-1, 1]
        x0 = torch.randn_like(x1)
        t = torch.rand(1)
        t_expanded = t.view(-1, 1, 1)

        x_t = (1 - t_expanded) * x0 + t_expanded * x1
        dx = x1 - x0

        return x_t, t.squeeze(0), dx, label


def get_llla_dataloader(
    batch_size: int = 120,
    shuffle: bool = True,
    max_timesteps: int = 1000,
    mode: str = "diffusion",  # "diffusion" or "flow"
    dataset_name: str = "MNIST",
    num_elements: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Load MNIST, FashionMNIST, or KMNIST and return DataLoaders for either:
      - 'diffusion': returns (x_t, t, noise, label)
      - 'flow': returns (x_t, t, dx, label)
    """
    if dataset_name == "MNIST":
        DatasetClass = datasets.MNIST
    elif dataset_name == "FashionMNIST":
        DatasetClass = datasets.FashionMNIST
    elif dataset_name == "KMNIST":
        DatasetClass = datasets.KMNIST
    else:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Use 'MNIST', 'FashionMNIST', or 'KMNIST'."
        )

    train_transform = get_transforms(dataset_name, train=True)
    test_transform = get_transforms(dataset_name, train=False)

    base_train = DatasetClass(
        root="./data", train=True, download=True, transform=train_transform
    )
    base_test = DatasetClass(
        root="./data", train=False, download=True, transform=test_transform
    )

    if num_elements is not None:
        base_train = Subset(base_train, range(min(num_elements, len(base_train))))
        base_test = Subset(base_test, range(min(num_elements, len(base_test))))

    if mode == "diffusion":
        train_dataset = DiffusionDataset(base_train, max_timesteps=max_timesteps)
        test_dataset = DiffusionDataset(base_test, max_timesteps=max_timesteps)
    elif mode == "flow":
        train_dataset = FlowMatchingDataset(base_train)
        test_dataset = FlowMatchingDataset(base_test)
    else:
        raise ValueError(f"Unknown mode '{mode}', must be 'diffusion' or 'flow'.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
