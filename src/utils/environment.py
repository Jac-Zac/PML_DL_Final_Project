import os
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


def load_pretrained_model(model_name, ckpt_path, device, **kwargs):
    if model_name == "unet":
        model = DiffusionUNet(**kwargs).to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    if ckpt_path is not None and os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location=device)

        # ✅ Check if full checkpoint or raw state_dict
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        if missing:
            print(f"⚠️ Missing keys in state_dict: {missing}")
        if unexpected:
            print(f"⚠️ Unexpected keys in state_dict: {unexpected}")
        print("✅ Loaded model weights from:", ckpt_path)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    return model
