#!/usr/bin/env python

from src.models.diffusion import Diffusion
from src.models.unet import UNet
from src.utils.data import get_dataloaders
from src.utils.environment import get_device, set_seed


def main():
    set_seed(1337)
    device = get_device()

    # Load your trained model here, or instantiate a new one for inference
    model = UNet().to(device)
    model.eval()

    diffusion = Diffusion(img_size=28, device=device)

    # Here you can implement your LLaMA testing / inference code,
    # e.g. loading test images, performing sampling, etc.

    # Example: Just load test dataloader (if needed)
    _, test_loader = get_dataloaders(batch_size=1)

    # TODO: Implement your inference or LLaMA-related logic here


if __name__ == "__main__":
    main()
