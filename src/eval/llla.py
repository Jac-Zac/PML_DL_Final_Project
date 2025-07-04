import argparse
import os
from types import SimpleNamespace

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.models.llla_model import LaplaceApproxModel
from src.utils.data import get_llla_dataloader
from src.utils.environment import get_device, load_pretrained_model
from src.utils.plots import plot_interleaved_image_uncertainty, plot_uncertainty_metric


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Laplace Approximation Sampling with Uncertainty Visualization"
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/best_model.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--model-name", type=str, default="unet", help="Model name to use from registry"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["diffusion", "flow"],
        default="flow",
        help="Method type",
    )
    parser.add_argument(
        "--dataset-name", type=str, default="MNIST", help="Dataset name"
    )
    parser.add_argument(
        "--save-dir", type=str, default="samples", help="Directory to save output plots"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Sample batch size")

    parser.add_argument(
        "--slice-start", type=int, default=0, help="Start index for image slice"
    )
    parser.add_argument(
        "--slice-end", type=int, default=2, help="End index for image slice"
    )
    parser.add_argument(
        "--cov-samples",
        type=int,
        default=100,
        help="Number of samples for covariance estimation",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of generation steps (diffusion/flow)",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    num_classes = 10
    slice_images = slice(args.slice_start, args.slice_end)

    os.makedirs(args.save_dir, exist_ok=True)

    model_kwargs = {
        "num_classes": num_classes,
        "time_embedding_type": "mlp" if args.method == "flow" else "sinusoidal",
    }

    # Load pretrained model
    model = load_pretrained_model(
        model_name=args.model_name,
        ckpt_path=args.ckpt,
        device=device,
        model_kwargs=model_kwargs,
        use_wandb=True,
    )

    # Data loader
    train_loader, _ = get_llla_dataloader(
        batch_size=args.batch_size, mode=args.method, dataset_name=args.dataset_name
    )

    # Config for MNIST or custom dataset
    config = SimpleNamespace()
    config.data = SimpleNamespace(image_size=28)

    # Fit Laplace approximation model
    laplace_model = LaplaceApproxModel(model, train_loader, args=None, config=config)

    # Choose generative method
    if args.method == "diffusion":
        from src.models.diffusion import UQDiffusion

        method_instance = UQDiffusion(img_size=config.data.image_size, device=device)
    else:
        from src.models.flow import UQFlowMatching

        method_instance = UQFlowMatching(img_size=config.data.image_size, device=device)

    mpl.rcParams["figure.dpi"] = 300

    all_samples_grouped, uncertainties = method_instance.sample_with_uncertainty(
        model=laplace_model,
        y=torch.arange(num_classes, device=device) % num_classes,
        cov_num_sample=args.cov_samples,
        num_steps=args.steps,
        log_intermediate=True,
    )

    # Interleaved image + uncertainty plots
    plot_interleaved_image_uncertainty(
        images=all_samples_grouped[:, slice_images, ...],
        uncertainties=uncertainties[:, slice_images, ...],
        save_path=os.path.join(args.save_dir, "image_uncertainty_by_step"),
        timesteps=np.linspace(1, args.steps, 10, dtype=int).tolist(),
        uq_cmp="viridis",
    )

    # Interleaved image + uncertainty plots
    plot_interleaved_image_uncertainty(
        images=all_samples_grouped[:, slice_images, ...],
        uncertainties=uncertainties[:, slice_images, ...],
        save_path=os.path.join(args.save_dir, "image_uncertainty_single"),
        timesteps=[args.steps],
        uq_cmp="viridis",
    )

    # Global plot style
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "legend.title_fontsize": 7,
            "lines.markersize": 3,
            "lines.linewidth": 1,
        }
    )

    # Define class label maps
    FASHION_MNIST_CLASSES = {
        0: "T-shirt/top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot",
    }

    MNIST_CLASSES = {i: str(i) for i in range(10)}

    # Choose correct label map
    label_map = (
        FASHION_MNIST_CLASSES if args.dataset_name == "FashionMNIST" else MNIST_CLASSES
    )

    # Plot with label names
    plot_uncertainty_metric(
        uncertainties,
        metrics=["sum"],
        label_map=label_map,
        save_path=os.path.join(args.save_dir, "sum_image_uncertainty"),
        colormap="tab10",
    )


if __name__ == "__main__":
    main()
