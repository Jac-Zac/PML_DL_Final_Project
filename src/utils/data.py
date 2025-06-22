import random
from typing import Optional, Tuple

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


def get_beta_schedule(beta_start: float, beta_end: float, timesteps: int):
    """Linear beta schedule from beta_start to beta_end."""
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


class DiffusionMNIST(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        betas: torch.Tensor,
        uniform_dequantization: bool = False,
    ):
        """
        base_dataset: any (image, label) dataset whose images are in [0,1]
        betas: 1D tensor of length T with noise schedule
        """
        super().__init__()
        self.base = base_dataset
        self.betas = betas
        self.alphas = 1.0 - betas
        # cumulative product ᾱ_t = ∏_{i=1}^t α_i
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        # 1) get clean image x and label
        x, label = self.base[idx]
        # ensure x in [0,1]
        # torchvision MNIST gives [0,1] after ToTensor, then normalized,
        # so if you want [0,1] you might skip Normalize or invert it here.
        # Here we assume x is already in [0,1]
        x = x.clamp(0.0, 1.0)

        # 2) sample a random timestep t ∈ {0,...,T-1}
        T = self.betas.shape[0]
        t = random.randint(0, T - 1)

        # 3) draw noise ε ∼ N(0, I)
        e = torch.randn_like(x)

        # 4) compute noisy image
        a_bar = self.alpha_bar[t]
        xt = torch.sqrt(a_bar) * x + torch.sqrt(1 - a_bar) * e

        # 5) return everything
        #   xt: noisy image
        #   t: scalar timestep
        #   e_flat: flattened noise (for loss)
        #   x: clean image (optional)
        #   label: class label
        return xt, t, e.view(-1), x, label


def get_llla_dataloader(
    batch_size: int = 120,
    shuffle: bool = True,
    max_timesteps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns train and test loaders that yield tuples:
      (xt, t, e, x, label)
    """
    # 1) prepare transforms so x∈[0,1]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # 2) load MNIST
    base_train = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    base_test = datasets.MNIST(
        "./data", train=False, download=True, transform=transform
    )

    # 4) create beta schedule once
    betas = get_beta_schedule(beta_start, beta_end, max_timesteps)

    # 5) wrap in diffusion dataset
    train_dataset = DiffusionMNIST(base_train, betas)
    test_dataset = DiffusionMNIST(base_test, betas)

    # 6) loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
