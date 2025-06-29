import argparse

from src.utils.environment import get_device, load_pretrained_model
from src.utils.plots import plot_image_grid


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate samples with either diffusion or flow matching"
    )
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
        help="Maximum number of timesteps for diffusion",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="unet",
        help="Model name to use from registry",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["diffusion", "flow"],
        default="diffusion",
        help="Which generative method to use",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    num_classes = 10

    model_kwargs = {
        "num_classes": num_classes,
        "time_emb_dim": 128,
        # NOTE: Change time embedding to learned for flow which is more sensible
        "time_embedding_type": "mlp" if args.method == "flow" else "sinusoidal",
    }

    # Load pretrained model
    model = load_pretrained_model(
        model_name=args.model_name,
        ckpt_path=args.ckpt,
        device=device,
        model_kwargs=model_kwargs,
    )

    # Choose generation method
    if args.method == "diffusion":
        from src.models.diffusion import Diffusion

        method_instance = Diffusion(img_size=28, device=device)
    elif args.method == "flow":
        from src.models.flow import FlowMatching

        method_instance = FlowMatching(img_size=28, device=device)
        args.max_steps = 10

    plot_image_grid(
        model,
        method_instance,
        num_intermediate=5,
        n=args.n,
        max_steps=args.max_steps,
        save_dir=args.save_dir,
        device=device,
        num_classes=num_classes,
    )


if __name__ == "__main__":
    main()
