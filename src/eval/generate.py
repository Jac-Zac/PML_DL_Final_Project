import argparse
import os

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
        "--n", type=int, default=4, help="Number of samples to generate"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="samples",
        help="Directory to save output image",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = get_device()
    model = load_pretrained_model("unet", args.ckpt, device, time_emb_dim=128)
    diffusion = Diffusion(img_size=28, device=device)

    steps = [800, 600, 400, 200, 0]

    all_samples_grouped = []

    for i in range(args.n):
        sample_steps = diffusion.sample(
            model, t_sample_times=steps, log_intermediate=True
        )
        print(f"Sample {i+1} returned {len(sample_steps)} images")
        assert len(sample_steps) == len(steps), (
            f"Sample {i+1} returned {len(sample_steps)} images, "
            f"expected {len(steps)}"
        )
        all_samples_grouped.append(sample_steps)

    # Flatten the grouped list after verifying lengths
    flat_samples = [img for sample in all_samples_grouped for img in sample]

    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, "all_samples_grid.png")

    plot_image_grid(flat_samples, out_path, num_samples=args.n, timesteps=steps)


if __name__ == "__main__":
    main()
