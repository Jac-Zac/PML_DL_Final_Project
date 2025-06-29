from typing import Any, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


def get_dataloaders(
    batch_size: int = 120,
    shuffle: bool = True,
    num_elements: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Load MNIST dataset and return (image, label) DataLoaders.
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Resize((28, 28)),
        ]
    )

    train_data = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_data = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    if num_elements is not None:
        train_data = Subset(train_data, range(min(num_elements, len(train_data))))
        test_data = Subset(test_data, range(min(num_elements, len(test_data))))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# NOTE: This is the dataloader for diffusion LLLA


class DiffusionMNIST(Dataset):
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

        # sample a random timestep
        t = torch.randint(low=0, high=self.max_timesteps, size=(1,))
        a = self.alphas_cumprod[t]
        noise = torch.randn_like(image)

        # Apply proper tensor reshaping for broadcasting
        sqrt_a = a.sqrt()
        sqrt_one_minus_a = (1 - a).sqrt()

        # Ensure proper broadcasting by reshaping if needed
        if image.dim() > 1:  # If image has multiple dimensions
            sqrt_a = sqrt_a.view(-1, *([1] * (image.dim() - 1)))
            sqrt_one_minus_a = sqrt_one_minus_a.view(-1, *([1] * (image.dim() - 1)))

        x_t = image * sqrt_a + noise * sqrt_one_minus_a

        # return (image_noised, timestep, noise, label)
        return x_t, t.squeeze(0), noise, label


class FlowMatchingMNIST(Dataset):
    def __init__(self, base_dataset: Dataset):
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """
        Returns:
            x_t: interpolated image between x0 and x1 at time t
            t: sampled time ∈ [0, 1]
            dx: velocity vector (x1 - x0)
            x0: random noise image
            label: digit label
        """
        image, label = self.base_dataset[idx]
        x1 = image * 2.0 - 1.0  # scale to [-1, 1]
        x0 = torch.randn_like(x1)
        t = torch.rand(1)
        t_expanded = t.view(-1, *([1] * (x1.dim() - 1)))  # broadcast to image shape

        x_t = (1 - t_expanded) * x0 + t_expanded * x1
        dx = x1 - x0

        return x_t, t.squeeze(0), dx, x0, label


def get_llla_dataloader(
    batch_size: int = 120,
    shuffle: bool = True,
    num_elements: Optional[int] = None,
    max_timesteps: int = 1000,
    mode: str = "diffusion",  # "diffusion" or "flow"
) -> Tuple[DataLoader, DataLoader]:
    """
    Load MNIST dataset and return DataLoaders for either:
      - 'diffusion': returns (x_t, t, noise, label)
      - 'flow': returns (x1, label)
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

    if mode == "diffusion":
        train_dataset = DiffusionMNIST(base_train, max_timesteps=max_timesteps)
        test_dataset = DiffusionMNIST(base_test, max_timesteps=max_timesteps)
    elif mode == "flow":
        train_dataset = FlowMatchingMNIST(base_train)
        test_dataset = FlowMatchingMNIST(base_test)
    else:
        raise ValueError(f"Unknown mode '{mode}', must be 'diffusion' or 'flow'.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
