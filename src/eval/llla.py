from src.models.diffusion import QUDiffusion
from src.models.llla_model import LaplaceApproxModel
from src.utils.data import get_llla_dataloader
from src.utils.environment import get_device, load_pretrained_model


def main():
    device = get_device()
    num_classes = 10  # adjust if needed

    # NOTE: If you have local wanbd directory and different runs online it might interfere
    # Perhaps fetch the data somehow or remove it so that it looks it up online

    # Load pretrained MAP model using best checkpoint
    diff_model = load_pretrained_model(
        model_name="unet",
        ckpt_path="jac-zac/diffusion-project/best-model:v22",
        device=device,
        model_kwargs={"num_classes": num_classes, "time_emb_dim": 128},
        use_wandb=True,
    )

    # 2️⃣ Prepare data loaders for the Laplace fit
    train_loader, _ = get_llla_dataloader(batch_size=128, num_elements=10)

    # WARNING: This is currently wrong I have to use the Diffusion class perhaps
    # to return a dataloader with images with noise or somehow use directly the functions inside diffusion

    # Wrap diffusion model with your Custom Model for Laplace last layer approx
    # NOTE: Automatically call fit
    laplace_model = LaplaceApproxModel(diff_model, train_loader, args=None, config=None)

    print("Laplace fitting completed on last layer of the diffusion model.")

    # NOTE:
    # You can use custom_model.forward or custom_model.accurate_forward for predictions

    # Initialize uncertainty-aware diffusion (same interface as base class)
    diffusion = QUDiffusion(img_size=28, device=device)

    # Sample to detailed uncertainty information
    intermediates, uncertainties, covariances = diffusion.sample_with_uncertainty(
        model=laplace_model,
        channels=1,
    )

    # print(intermediates)
    # print(uncertainties)
    # print(covariances)

    return diffusion

    # Sample using the Laplace-approximated model
    samples = diffusion.sample(model=custom_model, channels=1)
    final_images = samples[-1]  # normalized to [0,1]

    # Do something with final_images — save them or visualize
    print(f"Generated batch of {final_images.shape[0]} images!")


if __name__ == "__main__":
    main()
