import argparse
import os

import numpy as np

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
    return parser.parse_args()


def main():
    args = parse_args()

    device = get_device()
    model = load_pretrained_model("uvit", args.ckpt, device, time_emb_dim=128)
    diffusion = Diffusion(img_size=28, device=device)

    num_intermediate = 5

    # Include timestep zero by extending linspace endpoint and count
    intermediate_steps = np.linspace(
        args.max_steps, 0, num_intermediate + 1, dtype=int
    ).tolist()

    all_samples_grouped = []

    for i in range(args.n):
        # This returns len(intermediate_steps)+1 images (intermediates + final)
        sample_steps = diffusion.sample_ddim(
            model, t_sample_times=intermediate_steps, log_intermediate=True
        )
        all_samples_grouped.append(sample_steps)
        print(f"Generated sample {i + 1}")

    # Flatten the list of lists: one list of all images
    flat_samples = [img for sample in all_samples_grouped for img in sample]
    steps = intermediate_steps

    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, "all_samples_grid.png")

    plot_image_grid(flat_samples, out_path, num_samples=args.n, timesteps=steps)


if __name__ == "__main__":
    main()
