import logging
import os
import random
from typing import Optional

import numpy as np
import torch
import wandb

from src.models.unet import DiffusionUNet

# Import other models here as needed
logger = logging.getLogger(__name__)

# Centralized model registry dictionary
MODEL_REGISTRY = {
    "unet": DiffusionUNet,
    # "dvit": DiffusionDVit,
    # Add more models here
}


def set_seed(seed):
    """
    Set random seed for reproducibility.
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
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_model(model_name: str, device: torch.device, **kwargs):
    """
    Retrieve and instantiate a model from the registry.

    Args:
        model_name (str): Key for model in MODEL_REGISTRY.
        device (torch.device): Target device for the model.
        kwargs: Additional model-specific kwargs.

    Returns:
        torch.nn.Module: Instantiated model on the device.
    """
    model_cls = MODEL_REGISTRY.get(model_name)
    if model_cls is None:
        raise ValueError(f"❌ Unknown model name: {model_name}")

    model = model_cls(**kwargs).to(device)
    logger.info(f"✅ Instantiated model '{model_name}' on {device}")
    return model


def load_checkpoint(
    model_name: str,
    checkpoint_path: Optional[str],
    device: torch.device,
    optimizer_class=torch.optim.Adam,
    optimizer_kwargs=None,
    model_kwargs=None,
    scheduler=None,
):
    """
    Load checkpoint and instantiate model, optimizer, and optionally scheduler.

    Returns:
        model (nn.Module), optimizer, scheduler, start_epoch, best_val_loss
    """
    optimizer_kwargs = optimizer_kwargs or {}
    model_kwargs = model_kwargs or {}

    model = get_model(model_name, device, **model_kwargs)
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

    epoch = 1
    best_val_loss = float("inf")

    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"🔄 Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        epoch = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        logger.info(f"✅ Checkpoint loaded. Resuming from epoch {epoch}")
    else:
        if checkpoint_path:
            logger.warning(f"⚠️ Checkpoint path not found: {checkpoint_path}")

    return model, optimizer, scheduler, epoch, best_val_loss


def load_pretrained_model(
    model_name: str,
    ckpt_path: str,
    device: torch.device,
    model_kwargs=None,
    use_wandb=False,
):
    """
    Load a pretrained model from a local path or a W&B artifact.

    Args:
        model_name (str): Name of the model architecture.
        ckpt_path (str): Path to checkpoint file or W&B artifact reference (e.g. 'user/project/model:latest').
        device (torch.device): Device to load the model on.
        model_kwargs (dict): Additional kwargs passed to the model constructor.
        use_wandb (bool): Whether to use W&B API for artifact access.

    Returns:
        nn.Module: The loaded model.
    """
    model_kwargs = model_kwargs or {}
    model = get_model(model_name, device, **model_kwargs)

    if use_wandb:
        from dotenv import load_dotenv

        load_dotenv()
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key is None:
            wandb_api_key = input(
                "WANDB_API_KEY environment variable not set. Please enter your WandB API key: "
            ).strip()
            if not wandb_api_key:
                raise ValueError("WandB API key is required to proceed.")
        wandb.login(key=wandb_api_key)

    # Check if it's a W&B artifact path
    if use_wandb and ":" in ckpt_path and "/" in ckpt_path:
        logger.info(f"📦 Loading model checkpoint from W&B artifact: {ckpt_path}")
        artifact = wandb.Api().artifact(ckpt_path, type="model")
        artifact_dir = artifact.download()
        ckpt_path = os.path.join(artifact_dir, os.listdir(artifact_dir)[0])

    if not os.path.exists(ckpt_path):
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
