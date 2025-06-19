# from src.models.diffusion import Diffusion

from src.models.llla_model import LaplaceApproxModel
from src.utils.data import get_dataloaders
from src.utils.environment import get_device, load_pretrained_model


def main():
    device = get_device()
    num_classes = 10  # adjust if needed

    # Load pretrained MAP model using best checkpoint
    diff_model = load_pretrained_model(
        model_name="unet",
        ckpt_path="checkpoints/best_model.pth",
        device=device,
        model_kwargs={"num_classes": num_classes, "time_emb_dim": 128},
    )

    # 2️⃣ Prepare data loaders for the Laplace fit
    train_loader, _ = get_dataloaders(batch_size=128)

    # 3️⃣ Wrap diffusion model with your CustomModel for Laplace last layer approx
    custom_model = LaplaceApproxModel(diff_model, train_loader, args=None, config=None)

    custom_model.to(device)

    # 4️⃣ Now the custom model is fit during initialization, or you can call fit explicitly:
    custom_model.fit(train_loader)

    # 5️⃣ You can use custom_model.forward or custom_model.accurate_forward for predictions

    # diffusion = Diffusion(img_size=28, device=device)

    # Example usage (replace with actual usage):
    # x, t, y = next(iter(train_loader))
    # logits, var = custom_model(x.to(device), t.to(device), y.to(device))

    print("Laplace fitting completed on last layer of the diffusion model.")


if __name__ == "__main__":
    main()
