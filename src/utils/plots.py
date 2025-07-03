import math
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
    timesteps: list,
    save_path: str,
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

    # Ensure axes is a 2D array
    if isinstance(axes, plt.Axes):
        axes = np.array([[axes]])
    elif isinstance(axes, np.ndarray):
        if axes.ndim == 1:
            if T == 1:
                axes = axes[:, np.newaxis]
            elif 2 * B == 1:
                axes = axes[np.newaxis, :]

    all_unc_values = []
    unc_images = []

    for row in range(B):
        for col in range(T):
            img = images[row, col].cpu().numpy()
            if img.shape[0] == 1:
                img = img[0]  # Remove channel dim if single-channel
            else:
                img = np.transpose(img, (1, 2, 0))  # CxHxW → HxWxC

            unc = uncertainties[row, col].squeeze().cpu().numpy()
            all_unc_values.append(unc)

            ax_img = axes[2 * row, col]
            ax_unc = axes[2 * row + 1, col]

            ax_img.imshow(img, cmap=img_cmap)
            ax_img.axis("off")
            if row == 0:
                ax_img.set_title(f"interval={timesteps[col]}", fontsize=10)
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


def plot_uncertainty_metric(
    uncertainties,
    samples=None,
    metrics=["sum"],  # can be str or list of str
    figsize=None,
    colormap="viridis",
    title=None,
    xlabel="Steps",
    ylabel=None,
    save_path=None,
    label_map=None,
):
    """
    Plot one or multiple metrics of uncertainty over spatial dims for each label/sample.
    If multiple metrics are passed, create a grid of subplots.

    Args:
        uncertainties: np.ndarray or tensor, shape [T, num_labels, C?, H, W]
        samples: list of sample indices to plot. If None, plot all.
        metrics: str or list of str, metrics to plot. One of ['sum', 'mean', 'var', 'delta', 'mean_std']
        figsize: tuple or None. If None, auto set based on number of subplots.
        colormap: matplotlib colormap name.
        title: str or None. If multiple metrics, can be None or list of titles.
        xlabel, ylabel: axis labels. If None, auto-set.
        save_path: str or None to save figure instead of showing.
    """

    # Convert tensor to numpy if needed
    if hasattr(uncertainties, "cpu"):
        uncertainties = uncertainties.cpu().numpy()

    # Remove singleton channel dim if present
    if uncertainties.ndim == 5 and uncertainties.shape[2] == 1:
        uncertainties = uncertainties.squeeze(2)

    T, num_labels, H, W = uncertainties.shape

    if samples is None:
        samples = list(range(num_labels))
    else:
        samples = [s for s in samples if 0 <= s < num_labels]

    # Normalize metrics to list
    if isinstance(metrics, str):
        metrics = [metrics]

    n_metrics = len(metrics)

    # Determine grid size for subplots if multiple metrics
    if n_metrics == 1:
        nrows, ncols = 1, 1
    else:
        ncols = math.ceil(math.sqrt(n_metrics))
        nrows = math.ceil(n_metrics / ncols)

    # Set figsize if not provided
    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    if n_metrics == 1:
        axs = np.array([axs])  # make iterable
    else:
        axs = axs.flatten()

    cmap = plt.get_cmap(colormap, len(samples))

    for i, metric in enumerate(metrics):
        ax = axs[i]

        for idx, label_idx in enumerate(samples):
            data = uncertainties[:, label_idx]

            if metric == "sum":
                values = data.sum(axis=(-1, -2))
                steps = range(T)
            elif metric == "mean":
                values = data.mean(axis=(-1, -2))
                steps = range(T)
            elif metric == "var":
                values = data.var(axis=(-1, -2))
                steps = range(T)
            elif metric == "delta":
                values = np.diff(data.sum(axis=(-1, -2)))
                steps = range(1, T)
            elif metric == "mean_std":
                means = data.mean(axis=(-1, -2))
                stds = data.std(axis=(-1, -2))
                label_name = label_map[label_idx] if label_map else f"Label {label_idx}"
                ax.errorbar(
                    range(T),
                    means,
                    yerr=stds,
                    label=label_name,
                    color=cmap(idx),
                    capsize=3,
                    fmt="o-",
                )
                continue
            else:
                raise ValueError(f"Unsupported metric: {metric}")

        label_name = label_map[label_idx] if label_map else f"Label {label_idx}"
        ax.plot(
            steps,
            values,
            marker="o",
            linestyle="-",
            color=cmap(idx),
            label=label_name,
        )
        # Titles and labels
        if title is None:
            plot_title = (
                f"{metric.replace('_', ' ').title()} of uncertainties over time"
            )
        elif isinstance(title, list) and len(title) == n_metrics:
            plot_title = title[i]
        else:
            plot_title = title

        ax.set_title(plot_title)

        ax.set_xlabel(xlabel)

        if ylabel is None:
            ylabel_map = {
                "sum": "Sum of pixel uncertainty",
                "mean": "Mean pixel uncertainty",
                "var": "Variance of pixel uncertainty",
                "delta": "Change in uncertainty sum",
                "mean_std": "Mean ± Std of pixel uncertainty",
            }
            ax.set_ylabel(ylabel_map.get(metric, "Uncertainty"))
        else:
            ax.set_ylabel(ylabel)

        ax.grid(True)
        ax.legend(title="Labels")

    # Hide any unused subplots if metrics < nrows*ncols
    for j in range(n_metrics, nrows * ncols):
        fig.delaxes(axs[j])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
