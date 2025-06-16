import argparse
import os

import numpy as np
import torch

from src.models.diffusion import Diffusion
from src.utils.environment import get_device, load_pretrained_model
from src.utils.plots import plot_image_grid


def parse_args():
    parser = argparse.ArgumentParser(description="Generate DDPM samples")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/best_model.pth",
        help="Model checkpoint",
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
        default="samples",
        help="Directory to save output image",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum number of diffusion timesteps (default: 50)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="unet",
        help="Model name to use from registry",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = get_device()
    # HACK: Hard coded number of classes for now
    num_classes = 10

    model_kwargs = {
        "num_classes": num_classes,
        "time_emb_dim": 128,  # you can make this a CLI arg if needed
    }

    model = load_pretrained_model(
        model_name=args.model_name,
        ckpt_path=args.ckpt,
        device=device,
        model_kwargs=model_kwargs,
    )

    diffusion = Diffusion(img_size=28, device=device)

    num_intermediate = 5
    intermediate_steps = np.linspace(
        args.max_steps, 0, num_intermediate + 1, dtype=int
    ).tolist()

    # Generate labels: 0, 1, ..., n-1 modulo num_classes
    y = torch.arange(args.n) % num_classes
    y = y.to(device)

    all_samples_grouped = diffusion.sample(
        model,
        t_sample_times=intermediate_steps,
        log_intermediate=True,
        y=y,  # conditioning labels batch
    )
    print(f"Generated {args.n} samples with labels {y.tolist()}")
    stacked = torch.stack(all_samples_grouped)  # (T, B, C, H, W)

    permuted = stacked.permute(1, 0, 2, 3, 4)  # (B, T, C, H, W)
    flat_samples = permuted.reshape(-1, *permuted.shape[2:])  # (B*T, C, H, W)

    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, "all_samples_grid.png")

    plot_image_grid(
        flat_samples, out_path, num_samples=args.n, timesteps=intermediate_steps
    )


if __name__ == "__main__":
    main()
