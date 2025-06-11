# run_generate.py
import argparse
import os

import matplotlib.pyplot as plt

from src.models.diffusion import Diffusion
from src.utils.environment import get_device, load_pretrained_model
from src.utils.wandb import finish_wandb, initialize_wandb, log_intermediate_steps


def save_images(images, out_dir="samples", prefix="step"):
    os.makedirs(out_dir, exist_ok=True)
    for idx, img in enumerate(images):
        img = img.squeeze().cpu().numpy()
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.savefig(os.path.join(out_dir, f"{prefix}_{idx}.png"), bbox_inches="tight")
        plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Generate DDPM samples")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/best_model.pth",
        help="Model checkpoint",
    )
    parser.add_argument(
        "--n", type=int, default=4, help="Number of samples to generate"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="samples",
        help="Directory to save output images",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    initialize_wandb(project="ddpm-training", run_name="inference")

    device = get_device()
    model = load_pretrained_model("unet", args.ckpt, device, time_emb_dim=32)
    diffusion = Diffusion(img_size=28, device=device)

    all_samples = []
    steps = [1000, 800, 600, 400, 200, 0]
    for _ in range(args.n):
        sample_steps = diffusion.sample(
            model, t_sample_times=steps, log_intermediate=True
        )
        all_samples.extend(sample_steps)

    log_intermediate_steps(all_samples)
    save_images(all_samples, out_dir=args.save_dir)
    finish_wandb()


if __name__ == "__main__":
    main()
