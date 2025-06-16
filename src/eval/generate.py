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
        default=50,
        help="Maximum number of diffusion timesteps (default: 50)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="Number of classes used for conditioning (default: 10)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = get_device()
    model = load_pretrained_model(
        "unet",
        args.ckpt,
        device,
        time_emb_dim=128,
        num_classes=args.num_classes,
    )
    diffusion = Diffusion(img_size=28, device=device)

    num_intermediate = 5
    intermediate_steps = np.linspace(
        args.max_steps, 0, num_intermediate + 1, dtype=int
    ).tolist()

    # Generate one class label per sample — e.g., labels 0, 1, ..., n-1 or sampled randomly
    y = torch.arange(args.n) % args.num_classes
    y = y.to(device)

    all_samples_grouped = []

    for i in range(args.n):
        class_y = y[i].unsqueeze(0)  # shape (1,)
        sample_steps = diffusion.sample_ddim(
            model,
            t_sample_times=intermediate_steps,
            log_intermediate=True,
            y=class_y,
        )
        all_samples_grouped.append(sample_steps)
        print(f"Generated sample {i + 1} with label {class_y.item()}")

    flat_samples = [img for sample in all_samples_grouped for img in sample]
    steps = intermediate_steps

    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, "all_samples_grid.png")

    plot_image_grid(flat_samples, out_path, num_samples=args.n, timesteps=steps)


if __name__ == "__main__":
    main()
