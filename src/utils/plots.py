import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_image_grid(model, diffusion, n, max_steps, save_dir, device, num_classes):
    """
    Generates samples at intermediate steps and plots a grid where
    rows = samples and columns = timesteps.

    Args:
        model: The diffusion model.
        diffusion: The diffusion sampling object.
        n (int): Number of samples to generate.
        max_steps (int): Maximum diffusion steps.
        save_dir (str): Directory to save the output image.
        device: Torch device to run the model on.
        num_classes: Number of classes for label conditioning.
    """

    num_intermediate = 5
    intermediate_steps = np.linspace(
        max_steps, 0, num_intermediate + 1, dtype=int
    ).tolist()

    # Generate labels: 0, 1, ..., n-1 modulo num_classes
    y = torch.arange(n) % num_classes
    y = y.to(device)

    # Sample images
    all_samples_grouped = diffusion.sample(
        model,
        t_sample_times=intermediate_steps,
        log_intermediate=True,
        y=y,
    )
    print(f"Generated {n} samples with labels {y.tolist()}")

    # Reshape and permute
    stacked = torch.stack(all_samples_grouped)  # (T, B, C, H, W)
    permuted = stacked.permute(1, 0, 2, 3, 4)  # (B, T, C, H, W)
    flat_samples = permuted.reshape(-1, *permuted.shape[2:])  # (B*T, C, H, W)

    # Plot
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "all_samples_grid.png")

    num_samples = n
    num_timesteps = len(intermediate_steps)
    assert len(flat_samples) == num_samples * num_timesteps, "Mismatch in image count."

    fig, axes = plt.subplots(
        num_samples, num_timesteps, figsize=(1.5 * num_timesteps, 1.5 * num_samples)
    )

    if num_samples == 1:
        axes = np.expand_dims(axes, 0)
    if num_timesteps == 1:
        axes = np.expand_dims(axes, 1)

    for row in range(num_samples):
        for col in range(num_timesteps):
            idx = row * num_timesteps + col
            img = flat_samples[idx]
            if isinstance(img, torch.Tensor):
                img = img.squeeze().cpu().numpy()

            ax = axes[row, col]
            ax.imshow(img, cmap="gray")
            ax.axis("off")

            if row == 0:
                ax.set_title(f"t={intermediate_steps[col]}", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"Sample {row+1}", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
