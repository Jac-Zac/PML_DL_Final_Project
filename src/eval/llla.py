# FIX: This is just random testing.
# especially before this we need to make the model have a final linear layer

import argparse
import os

import numpy as np
import torch
from laplace import Laplace

from src.models.diffusion import Diffusion
from src.utils.data import (
    get_dataloaders,
)  # Assuming this is where your dataloaders come from
from src.utils.environment import get_device, load_pretrained_model
from src.utils.plots import plot_image_grid


class ModelWrapper(torch.nn.Module):
    def __init__(self, model, default_t=0):
        super().__init__()
        self.model = model
        self.default_t = default_t

    def forward(self, x, t=None):
        if t is None:
            # Create a tensor of default timesteps with batch size = x.shape[0]
            t = torch.full(
                (x.size(0),), self.default_t, device=x.device, dtype=torch.long
            )
        return self.model(x, t)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate DDPM samples with Laplace uncertainty"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=4,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="samples_with_uncertainty",
        help="Directory to save output images",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Max diffusion timesteps",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for Laplace fitting",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = get_device()

    # Load pretrained model
    model = load_pretrained_model(
        "unet",
        "checkpoints/best_model.pth",
        device,
        time_emb_dim=128,
        num_classes=10,  # fixed number of classes for simplicity
    )

    # Wrap model so it can handle missing 't' argument by providing default timestep
    model = ModelWrapper(model, default_t=0)

    diffusion = Diffusion(img_size=28, device=device)

    # Get train_loader for Laplace fitting
    train_loader, _ = get_dataloaders(batch_size=args.batch_size)

    # Setup Laplace approximation on last layer weights
    la = Laplace(
        model,
        likelihood="regression",
        subset_of_weights="last_layer",  # approximate last layer weights only
        hessian_structure="full",
        prior_precision=1e-3,
    )

    print("Fitting Laplace approximation on training data...")
    la.fit(train_loader)

    print("Optimizing prior precision...")
    la.optimize_prior_precision(method="marglik")

    print("Laplace approximation fitted.")

    # Sample diffusion images
    num_intermediate = 5
    intermediate_steps = np.linspace(
        args.max_steps, 0, num_intermediate + 1, dtype=int
    ).tolist()

    y = torch.arange(args.n) % 10
    y = y.to(device)

    all_samples_grouped = diffusion.sample(
        model.model,  # pass the underlying original model here for sampling
        t_sample_times=intermediate_steps,
        log_intermediate=True,
        y=y,
    )
    print(f"Generated {args.n} samples with labels {y.tolist()}")

    stacked = torch.stack(all_samples_grouped)  # (T, B, C, H, W)
    permuted = stacked.permute(1, 0, 2, 3, 4)  # (B, T, C, H, W)

    samples = permuted[:, -1, :, :, :]  # final timestep (B, C, H, W)

    model.eval()
    with torch.no_grad():
        f_mean, f_var = la(samples, full_cov=False)

    uncertainty_maps = f_var.reshape(samples.shape)

    os.makedirs(args.save_dir, exist_ok=True)

    plot_image_grid(
        samples.cpu(),
        os.path.join(args.save_dir, "samples.png"),
        num_samples=args.n,
        timesteps=[intermediate_steps[-1]],
    )
    plot_image_grid(
        uncertainty_maps.sqrt().cpu(),
        os.path.join(args.save_dir, "uncertainty.png"),
        num_samples=args.n,
        timesteps=[intermediate_steps[-1]],
    )

    print(f"Samples and uncertainty saved to {args.save_dir}")


if __name__ == "__main__":
    main()
