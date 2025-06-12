import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_image_grid(images, save_path, num_samples, timesteps):
    """
    Plots a grid where rows = samples and columns = timesteps.

    Args:
        images (list): Flat list of images (torch.Tensor or np.array),
                       length should be num_samples * len(timesteps)
        save_path (str): Path to save the output image.
        num_samples (int): Number of different samples (rows).
        timesteps (list): List of timesteps per sample (columns).
    """
    num_timesteps = len(timesteps)
    assert (
        len(images) == num_samples * num_timesteps
    ), "Image count does not match grid dimensions."

    fig, axes = plt.subplots(
        num_samples, num_timesteps, figsize=(1.5 * num_timesteps, 1.5 * num_samples)
    )

    # Make sure axes is 2D
    if num_samples == 1:
        axes = np.expand_dims(axes, 0)
    if num_timesteps == 1:
        axes = np.expand_dims(axes, 1)

    for row in range(num_samples):
        for col in range(num_timesteps):
            idx = row * num_timesteps + col
            img = images[idx]
            if isinstance(img, torch.Tensor):
                img = img.squeeze().cpu().numpy()

            ax = axes[row, col]
            ax.imshow(img, cmap="gray")
            ax.axis("off")

            # Top row: add timestep label
            if row == 0:
                ax.set_title(f"t={timesteps[col]}", fontsize=10)

            # Left column: add sample label
            if col == 0:
                ax.set_ylabel(f"Sample {row+1}", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
