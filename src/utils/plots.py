import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_image_grid(
    model,
    method_instance,
    n: int,
    num_intermediate: int,
    max_steps: int,
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
        max_steps (int): Maximum number of steps or timesteps.
        save_dir (str): Directory to save the output image.
        device: Torch device.
        num_classes (int): Number of classes for label conditioning.
    """
    # Prepare conditioning labels
    y = torch.arange(n, device=device) % num_classes

    # Decide which type of timesteps to generate
    if method_instance.__class__.__name__ == "FlowMatching":
        # Flow matching: choose indices between 0 and (steps-1)
        step_indices = torch.linspace(
            0, max_steps - 1, steps=num_intermediate, dtype=torch.int32
        ).tolist()

        all_samples_grouped = method_instance.sample(
            model,
            steps=max_steps,
            log_intermediate=True,
            t_sample_times=step_indices,
            y=y,
        )
        timesteps = step_indices
    else:
        # Diffusion: choose timesteps between max_steps and 0
        t_sample_times = torch.linspace(
            max_steps,
            0,
            steps=num_intermediate,
            dtype=torch.int32,
        ).tolist()

        all_samples_grouped = method_instance.sample(
            model,
            t_sample_times=t_sample_times,
            log_intermediate=True,
            y=y,
        )
        timesteps = t_sample_times

    # Stack all generated images into a (B, T, C, H, W) tensor
    stacked = torch.stack(all_samples_grouped)  # (T, B, C, H, W)
    permuted = stacked.permute(1, 0, 2, 3, 4)  # (B, T, C, H, W)
    num_samples, num_timesteps = permuted.shape[:2]

    # Save as a grid
    os.makedirs(save_dir, exist_ok=True)
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
                ax.set_title(f"step={timesteps[col]}", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"Sample {row+1}", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_image_uncertainty_grid(
    model,
    method_instance,
    n: int,
    num_intermediate: int,
    total_steps: int,
    save_dir: str,
    device: torch.device,
    num_classes: int,
    cov_num_sample: int = 50,
    uq_cmp: str = "grey",
):
    """
    Generate and plot a grid of intermediate samples for either diffusion or flow.

    Args:
        model: The trained model.
        method_instance: The sampling method instance (Diffusion or FlowMatching).
        n (int): Number of classes from which to generate ([0,1,..,n-1]).
        num_intermediate (int): Number of intermediate steps to visualize.
        total_steps (int): Maximum number of steps or timesteps.
        save_dir (str): Directory to save the output image.
        device: Torch device.
        num_classes (int): Number of classes for label conditioning.
    """
    # Prepare conditioning labels
    y = torch.arange(n, device=device) % num_classes

    # Decide which type of timesteps to generate
    if method_instance.__class__.__name__ == "UQFlowMatching":
        # Flow matching: choose indices between 0 and (steps-1)
        step_indices = torch.linspace(
            0, total_steps - 1, steps=num_intermediate, dtype=torch.int32
        ).tolist()

        all_samples_grouped, uncertainties = method_instance.sample_with_uncertainty(
            model,
            log_intermediate=True,
            t_sample_times=step_indices,
            y=y,
            cov_num_sample=cov_num_sample,
            num_steps=total_steps,
        )

    elif method_instance.__class__.__name__ == "UQDiffusion":
        # Diffusion: choose timesteps between total_steps and 0
        step_indices = torch.linspace(  # indices (=time value) of the saved steps
            total_steps - 1,
            0,
            steps=num_intermediate,
            dtype=torch.int32,
        ).tolist()

        all_samples_grouped, uncertainties = method_instance.sample_with_uncertainty(
            model,
            t_sample_times=step_indices,
            log_intermediate=True,
            y=y,
            cov_num_sample=cov_num_sample,
        )
    else:
        raise ValueError(f"Unknown method type: {method_instance.__class__.__name__}")

    timesteps = step_indices

    ### ------------------ Plot images grid ------------------ ###

    # Stack all generated images into a (B, T, C, H, W) tensor
    stacked = torch.stack(all_samples_grouped)  # (T, B, C, H, W)
    permuted = stacked.permute(1, 0, 2, 3, 4)  # (B, T, C, H, W)
    num_classes, num_timesteps = permuted.shape[:2]  # extract B and T

    # Save as a grid
    os.makedirs(save_dir, exist_ok=True)
    out_path_img = os.path.join(save_dir, "all_samples_grid.png")

    fig, axes = plt.subplots(
        num_classes,
        num_intermediate,
        figsize=(1.5 * num_intermediate, 1.5 * num_classes),
    )

    if num_classes == 1:
        axes = np.expand_dims(axes, 0)
    if num_intermediate == 1:
        axes = np.expand_dims(axes, 1)

    for row in range(num_classes):
        for col in range(num_timesteps):
            img = permuted[row, col].squeeze().cpu().numpy()
            ax = axes[row, col]
            ax.imshow(img, cmap="gray")
            ax.axis("off")
            if row == 0:
                ax.set_title(f"step={timesteps[col]}", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"Sample {row+1}", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path_img, bbox_inches="tight")
    plt.close()

    ### ------------------ Plot uncertainties grid ------------------ ###

    # Convert uncertainties to tensor if needed
    if isinstance(uncertainties, list):
        uncertainties = torch.stack(uncertainties)  # (T, B, C, H, W)

    # Multiplier
    # mult = 700
    mult = 1

    # Ensure uncertainties has same ordering: (B, T, C, H, W)
    uncertainties_permuted = uncertainties.permute(1, 0, 2, 3, 4) * mult

    out_path_unc = os.path.join(save_dir, "all_uncertainties_grid.png")

    fig, axes = plt.subplots(
        num_classes,
        num_intermediate,
        figsize=(1.5 * num_intermediate, 1.5 * num_classes),
    )

    if num_classes == 1:
        axes = np.expand_dims(axes, 0)
    if num_intermediate == 1:
        axes = np.expand_dims(axes, 1)

    for row in range(num_classes):
        for col in range(num_timesteps):
            unc = uncertainties_permuted[row, col].squeeze().cpu().numpy()
            ax = axes[row, col]
            im = ax.imshow(unc, cmap=uq_cmp)  # Heatmap for uncertainty
            ax.axis("off")
            if row == 0:
                ax.set_title(f"step={timesteps[col]}", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"Sample {row+1}", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path_unc, bbox_inches="tight")
    plt.close()

    return all_samples_grouped, uncertainties
