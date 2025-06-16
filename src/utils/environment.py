import logging
import os
import random

import numpy as np
import torch

from src.models.unet import DiffusionUNet

logger = logging.getLogger(__name__)


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


def load_checkpoint(model, optimizer, checkpoint_path, device):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint path not found: {checkpoint_path}")
        return 1, float("inf")

    logger.info(f"🔄 Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0) + 1
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    logger.info(f"✅ Checkpoint loaded. Resuming from epoch {epoch}")
    return epoch, best_val_loss


def load_pretrained_model(model_name, ckpt_path, device, **kwargs):
    model_registry = {
        "unet": DiffusionUNet,
        # Extend here with more models
    }

    model_cls = model_registry.get(model_name)
    if model_cls is None:
        raise ValueError(f"❌ Unknown model name: {model_name}")

    model = model_cls(**kwargs).to(device)

    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"❌ Checkpoint not found: {ckpt_path}")

    logger.info(f"🔄 Loading model weights from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        logger.warning(f"⚠️ Missing keys in state_dict: {missing_keys}")
    if unexpected_keys:
        logger.warning(f"⚠️ Unexpected keys in state_dict: {unexpected_keys}")

    logger.info("✅ Model weights loaded successfully")
    return model
