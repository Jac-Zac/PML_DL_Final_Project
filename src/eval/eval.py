#!/usr/bin/env python
import argparse
import os

import matplotlib.pyplot as plt

from models.diffusion import Diffusion
from utils.environment import get_device, load_pretrained_model


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
        default=None,
        help="Optional list of timesteps to visualize",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()

    # Load model via utility function
    model = load_pretrained_model(
        model_name="unet",
        model_path=args.ckpt,
        device=device,
        in_channels=1,
        out_channels=1,
        time_emb_dim=256,
        base_channels=64,
    )
    model.eval()

    # Set up diffusion
    diffusion = Diffusion(img_size=28, device=device)

    for i in range(args.n):
        print(f"Sampling image {i+1}/{args.n}")
        images = diffusion.sample(model, t_sample_times=args.steps)

        # Save all intermediate images
        save_or_show_images(images, out_dir=args.save_dir, prefix=f"sample_{i}_step")


if __name__ == "__main__":
    main()
