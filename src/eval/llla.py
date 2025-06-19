from laplace import Laplace
from torch import nn

from src.models.diffusion import Diffusion
from src.utils.data import get_dataloaders
from src.utils.environment import get_device, load_pretrained_model


class WrappedModel(nn.Module):
    """
    Wraps your UNet-based diffusion model so Laplace knows how to forward inputs
    including the time step t.
    """

    def __init__(self, diffusion_unet):
        super().__init__()
        self.model = diffusion_unet

    def forward(self, xt):
        # xt is a tuple (x, t, y) during Laplace .fit(), or you can pack t manually
        x, t, y = xt
        return self.model(x, t=t, y=y)  # adjust keyword args as needed


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

    wrapped = WrappedModel(model)

    # 2️⃣ Prepare data loaders for the Laplace fit
    train_loader, _ = get_dataloaders(batch_size=128)

    # 3️⃣ Wrap model with diagonal last-layer Laplace
    la = Laplace(
        wrapped,
        likelihood="classification",
        subset_of_weights="last_layer",
        hessian_structure="diag",
    )

    la.fit(train_loader)  # fit posterior to last-layer

    # TODO: I don't think this is necessary as a first step
    # la.optimize_prior_precision(
    #     method="CV",  # BayesDiff uses cross-validation
    #     pred_type="glm",  # required for classification
    #     link_approx="probit",  # common choice for classification
    #     val_loader=val_loader,
    # )
    print(la)

    # 4️⃣ Use the Laplace-wrapped model for sampling
    diffusion = Diffusion(img_size=28, device=device)

    # mean, var = la(x, pred_type="glm", link_approx="probit")

    # plot_image_grid(
    #     model,
    #     diffusion,
    #     n=10,
    #     max_steps=1000,
    #     save_dir="samples",
    #     device=device,
    #     num_classes=num_classes,
    # )


if __name__ == "__main__":
    main()
