import argparse

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
        default=5,
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

    plot_image_grid(model, diffusion, args, device, num_classes)


if __name__ == "__main__":
    main()
