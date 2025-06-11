#!/usr/bin/env python
import argparse
import os

import matplotlib.pyplot as plt
import torch

from models.diffusion import Diffusion
from utils.environment import get_device, load_pretrained_model
from utils.wandb_utils import finish_wandb, initialize_wandb, log_intermediate_steps


def save_or_show_images(images, out_dir=None, prefix="sample"):
    for idx, img in enumerate(images):
        img = img.squeeze().cpu().numpy()
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            plt.savefig(
                os.path.join(out_dir, f"{prefix}_{idx}.png"), bbox_inches="tight"
            )
        else:
            plt.show()
        plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images using trained DDPM.")
    parser.add_argument("--n", type=int, default=1, help="Number of images to generate")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/unet_trained.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--save-dir", type=str, default=None, help="Directory to save generated images"
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="*",
        default=[1000, 800, 600, 400, 200, 0],
        help="Timesteps to log/save",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="Enable logging to Weights and Biases"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()

    if args.wandb:
        initialize_wandb(project="diffusion-project", run_name="eval-run")

    model = load_pretrained_model("unet", args.ckpt, device=device, time_emb_dim=32)
    model.eval()
    diffusion = Diffusion(img_size=28, device=device)

    all_samples = []
    for i in range(args.n):
        sample_steps = diffusion.sample(
            model, t_sample_times=args.steps, log_intermediate=True
        )
        all_samples.extend(sample_steps)
        print(f"Generated sample {i+1}/{args.n}")

    if args.wandb:
        log_intermediate_steps(all_samples)
        finish_wandb()

    save_or_show_images(all_samples, out_dir=args.save_dir, prefix="step")


if __name__ == "__main__":
    main()
