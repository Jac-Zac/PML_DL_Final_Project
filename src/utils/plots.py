import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_image_grid(
    model,
    method_instance,
    n: int,
    num_intermediate: int,
    num_steps: int,
    save_dir: str,
    device: torch.device,
    num_classes: int,
):
    """
    Generate and plot a grid of intermediate samples for either diffusion or flow.

    Args:
        model: The trained model.
        method_instance: The sampling method instance (Diffusion or FlowMatching).
        n (int): Number of samples to generate.
        num_intermediate (int): Number of intermediate steps to visualize.
        num_steps (int): Maximum number of steps or timesteps.
        save_dir (str): Directory to save the output image.
        device: Torch device.
        num_classes (int): Number of classes for label conditioning.
    """
    os.makedirs(save_dir, exist_ok=True)
    y = torch.arange(n, device=device) % num_classes

    # Choose timesteps depending on method type
    if method_instance.__class__.__name__ == "FlowMatching":
        all_samples = method_instance.sample(
            model,
            steps=num_steps,
            log_intermediate=True,
            y=y,
        )
    else:  # Diffusion
        all_samples = method_instance.sample(
            model,
            log_intermediate=True,
            y=y,
        )

    # (T, B, C, H, W) -> (B, T, C, H, W)

    T = all_samples.shape[0]  # Total time steps
    indices = torch.linspace(
        0, T - 1, steps=num_intermediate
    ).long()  # Generate indices
    selected_samples = all_samples[indices]  # Direct indexing
    permuted = selected_samples.permute(1, 0, 2, 3, 4)

    num_samples, num_timesteps = permuted.shape[:2]

    out_path = os.path.join(save_dir, "all_samples_grid.png")

    fig, axes = plt.subplots(
        num_samples, num_timesteps, figsize=(1.5 * num_timesteps, 1.5 * num_samples)
    )

    if num_samples == 1:
        axes = np.expand_dims(axes, 0)
    if num_timesteps == 1:
        axes = np.expand_dims(axes, 1)

    for row in range(num_samples):
        for col in range(num_timesteps):
            img = permuted[row, col].squeeze().cpu().numpy()
            ax = axes[row, col]
            ax.imshow(img, cmap="gray")
            ax.axis("off")
            if row == 0:
                ax.set_title(f"step={col}", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"Sample {row+1}", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_interleaved_image_uncertainty(
    images: torch.Tensor,
    uncertainties: torch.Tensor,
    save_path: str,
    timesteps: list,
    uq_cmp: str = "viridis",
    img_cmap: str = "gray",
    mult: float = 700.0,
):
    """
    Plot interleaved image and uncertainty maps.

    Args:
        images (Tensor): Shape (T, B, C, H, W)
        uncertainties (Tensor): Shape (T, B, C, H, W)
        save_path (str): Path to save figure.
        timesteps (list[int]): Timestep indices to plot.
        uq_cmp (str): Colormap for uncertainty.
        img_cmap (str): Colormap for images.
        mult (float): Multiplier for uncertainty visualization.
    """
    if isinstance(images, list):
        images = torch.stack(images)
    if isinstance(uncertainties, list):
        uncertainties = torch.stack(uncertainties)

    T, B, C, H, W = images.shape
    assert (
        uncertainties.shape[0] == T and uncertainties.shape[1] == B
    ), "Images and uncertainties must have matching shapes in first two dims."

    # If timesteps length doesn't match T, slice internally
    if timesteps is None or len(timesteps) != T:
        if timesteps is None:
            timesteps = list(range(T))
        else:
            timesteps = list(timesteps)
        images = images[timesteps]
        uncertainties = uncertainties[timesteps]
        T = len(timesteps)

    images = images.permute(1, 0, 2, 3, 4)  # (B, T, C, H, W)
    uncertainties = uncertainties.permute(1, 0, 2, 3, 4) * mult  # (B, T, C, H, W)

    B = images.shape[0]

    fig, axes = plt.subplots(
        2 * B,
        T,
        figsize=(1.5 * T, 1.5 * 2 * B),
    )

    # Make sure axes is 2D array even if B=1 or T=1
    if T == 1:
        axes = np.expand_dims(axes, 1)
    if B == 1:
        axes = np.expand_dims(axes, 0)

    all_unc_values = []
    unc_images = []

    for row in range(B):
        for col in range(T):
            img = images[row, col].squeeze().cpu().numpy()
            unc = uncertainties[row, col].squeeze().cpu().numpy()
            all_unc_values.append(unc)

            ax_img = axes[2 * row, col]
            ax_unc = axes[2 * row + 1, col]

            ax_img.imshow(img, cmap=img_cmap)
            ax_img.axis("off")
            if row == 0:
                ax_img.set_title(f"step={timesteps[col]}", fontsize=10)
            if col == 0:
                ax_img.set_ylabel(f"Sample {row + 1}", fontsize=10)

            im = ax_unc.imshow(unc, cmap=uq_cmp)
            unc_images.append(im)
            ax_unc.axis("off")

    vmin = min(u.min() for u in all_unc_values)
    vmax = max(u.max() for u in all_unc_values)
    for im in unc_images:
        im.set_clim(vmin, vmax)

    plt.tight_layout()
    plt.subplots_adjust(right=0.87)
    cbar_ax = fig.add_axes([0.89, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(unc_images[0], cax=cbar_ax)
    cbar.set_label("Uncertainty", rotation=270, labelpad=15)

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_uncertainty_sums(
    uncertainties,
    samples=None,
    figsize=(10, 6),
    colormap="viridis",
    title="Sum of uncertainties over spatial dimensions",
    xlabel="Index in first dimension",
    ylabel="Sum over spatial dimensions",
    save_path=None,
):
    """
    Plot sums of uncertainties over spatial dims for each label/sample.

    Args:
        uncertainties (Tensor or np.ndarray): shape [T, num_labels, C, H, W]
        samples (list or None): list of sample indices to plot. If None, plot all.
        figsize (tuple): matplotlib figure size.
        colormap (str): name of matplotlib colormap.
        title (str): plot title.
        xlabel (str): x-axis label.
        ylabel (str): y-axis label.
        save_path (str or None): if set, save plot to this path instead of showing.
    """
    # Convert to numpy if tensor
    if hasattr(uncertainties, "cpu"):
        uncertainties = uncertainties.cpu().numpy()

    # Remove singleton channel dim if present
    if uncertainties.shape[2] == 1:
        uncertainties = uncertainties.squeeze(2)  # shape now (T, num_labels, H, W)

    T, num_labels, H, W = uncertainties.shape

    # If samples is None, plot all labels
    if samples is None:
        samples = list(range(num_labels))
    else:
        # Validate samples indices
        samples = [s for s in samples if 0 <= s < num_labels]

    cmap = plt.get_cmap(colormap, len(samples))

    plt.figure(figsize=figsize)

    for idx, label_idx in enumerate(samples):
        sums = uncertainties[:, label_idx].sum(axis=(-1, -2))  # shape (T,)
        plt.plot(
            range(T),
            sums,
            marker="o",
            linestyle="-",
            color=cmap(idx),
            label=f"Label {label_idx}",
        )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend(title="Labels")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
