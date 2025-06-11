import random

import numpy as np
import torch

from src.models.unet import DiffusionUNet


def set_seed(seed):
    """
    Set random seed for reproducibility.

    Parameters:
    - seed (int): Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.manual_seed(seed)


def get_device() -> torch.device:
    """
    Determine the best available device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def load_pretrained_model(
    model_name: str, model_path: str, device: torch.device, **kwargs
):
    """
    Loads a pretrained model by name and path.
    """
    if model_name.lower() == "unet":
        model = DiffusionUNet(**kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    return model.to(device)
