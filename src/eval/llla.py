from types import SimpleNamespace

from src.models.flow import UQFlowMatching
from src.models.llla_model import LaplaceApproxModel
from src.utils.data import get_llla_dataloader
from src.utils.environment import get_device, load_pretrained_model
from src.utils.plots import plot_interleaved_image_uncertainty


def main():
    device = get_device()
    num_classes = 10  # adjust if needed
    sample_batch_size = 16
    num_classes = 10
    method = "flow"
    save_dir = "samples"

    # NOTE: If you have local wanbd directory and different runs online it might interfere
    # Perhaps fetch the data somehow or remove it so that it looks it up online

    model_kwargs = {
        "num_classes": num_classes,
        "time_emb_dim": 128,
        # NOTE: We are currently using different time embedding because of a small bug but it is fine
        "time_embedding_type": "mlp" if method == "flow" else "sinusoidal",
    }

    # Load pretrained MAP model using best checkpoint
    model = load_pretrained_model(
        model_name="unet",
        ckpt_path="jac-zac/diffusion-project/best-model:v91",
        device=device,
        model_kwargs=model_kwargs,
        use_wandb=True,
    )

    # Prepare data loaders for the Laplace fit
    train_loader, _ = get_llla_dataloader(batch_size=sample_batch_size, mode=method)

    mnist_config = SimpleNamespace()
    mnist_config.data = SimpleNamespace()
    mnist_config.data.image_size = 28  # MNIST image size

    # Wrap diffusion model with your Custom Model for Laplace last layer approx
    # NOTE: Automatically call fit
    laplace_model = LaplaceApproxModel(
        model, train_loader, args=None, config=mnist_config
    )

    # NOTE:
    # You can use custom_model.forward or custom_model.accurate_forward for predictions

    total_steps = 20
    num_intermediate = 15

    # Initialize uncertainty-aware diffusion (same interface as base class)
    flow = UQFlowMatching(img_size=mnist_config.data.image_size, device=device)

    # FIX: Needs to be fixed taking inspiration from the notebook
    # TODO: Implement the remaining code
    # Flow Numbers 1/0 plots

    # TODO:
    # And pants plot for both Flow and Diffusion
    # + Plot of pants jsut last step for both
    # Create Only sum plot


if __name__ == "__main__":
    main()
