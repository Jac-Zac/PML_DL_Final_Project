# ddpm_bayesdiff.py
import os

import numpy as np
import torch
from laplace import Laplace

from src.models.diffusion import Diffusion
from src.utils.data import get_dataloaders  # adjust import accordingly
from src.utils.environment import get_device, load_pretrained_model
from src.utils.plots import plot_image_grid


def main():
    device = get_device()
    num_classes = 10  # adjust if needed

    # 1️⃣ Load pretrained MAP model using best checkpoint
    model = load_pretrained_model(
        model_name="unet",
        ckpt_path="checkpoints/best_model.pth",
        device=device,
        model_kwargs={"num_classes": num_classes, "time_emb_dim": 128},
    )

    # 2️⃣ Prepare data loaders for the Laplace fit
    train_loader, val_loader = get_dataloaders(batch_size=128)

    # 3️⃣ Wrap model with diagonal last-layer Laplace
    la = Laplace(
        model,
        likelihood="classification",
        subset_of_weights="last_layer",
        hessian_structure="diag",
    )
    la.fit(train_loader)  # fit posterior to last-layer
    la.optimize_prior_precision(method="CV", val_loader=val_loader)

    # 4️⃣ Use the Laplace-wrapped model for sampling
    diffusion = Diffusion(img_size=28, device=device)
    n_samples = 5
    intermediate = np.linspace(
        diffusion.default_max_steps, 0, num=6, dtype=int
    ).tolist()
    y = torch.arange(n_samples).to(device) % num_classes

    all_groups = diffusion.sample(
        model=la,
        t_sample_times=intermediate,
        log_intermediate=True,
        y=y,
    )
    print(f"Generated {n_samples} samples with labels {y.tolist()}")

    # 5️⃣ Assemble and save image grid
    stacked = torch.stack(all_groups)
    permuted = stacked.permute(1, 0, 2, 3, 4)  # shape (B, T, C, H, W)
    flat = permuted.reshape(-1, *permuted.shape[2:])
    os.makedirs("samples_bayesdiff", exist_ok=True)
    plot_image_grid(
        flat,
        "samples_bayesdiff/all_samples_bayesdiff.png",
        num_samples=n_samples,
        timesteps=intermediate,
    )


if __name__ == "__main__":
    main()
