import torch
import torch.optim as optim
from tqdm import tqdm

from src.models.diffusion import Diffusion
from src.models.unet import DiffusionUNet
from src.utils.wandb import (
    finish_wandb,
    initialize_wandb,
    log_best_model,
    log_epoch_metrics,
    log_intermediate_steps,
    log_training_step,
    save_and_log_model_checkpoint,
)


def train(
    num_epochs: int,
    device: torch.device,
    dataloader,
    val_loader,
    learning_rate: float = 1e-3,
    use_wandb: bool = False,
):
    model = DiffusionUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    diffusion = Diffusion(img_size=28, device=device)
    best_val_loss = float("inf")

    if use_wandb:
        initialize_wandb(
            project="diffusion-project",
            config={
                "epochs": num_epochs,
                "lr": learning_rate,
                "model": "DiffusionUNet",
            },
        )

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0

        for images, _ in tqdm(
            dataloader, desc=f"Epoch {epoch}/{num_epochs} [Train]", leave=False
        ):
            images = images.to(device) * 2 - 1  # Normalize to [-1, 1]

            optimizer.zero_grad()
            loss = diffusion.perform_training_step(model, images)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            train_loss += loss_val
            if use_wandb:
                log_training_step(loss_val)

        avg_train_loss = train_loss / len(dataloader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, _ in tqdm(
                val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]", leave=False
            ):
                images = images.to(device) * 2 - 1
                loss = diffusion.perform_training_step(model, images)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # Generate 4 samples with intermediate steps
        samples_steps = []
        with torch.no_grad():
            for _ in range(4):
                steps = diffusion.sample(
                    model,
                    t_sample_times=[1000, 800, 600, 400, 200, 0],
                    log_intermediate=True,
                )
                # steps is list of tensors per timestep
                samples_steps.append(torch.cat(steps, dim=0))

        # samples_steps: list of [timesteps x C x H x W] tensors, one per sample
        # Transpose to list of timesteps with batch of samples for logging
        # We'll pass this to wandb util to create correct grid

        if use_wandb:
            log_epoch_metrics(epoch, avg_train_loss, avg_val_loss)
            log_intermediate_steps(samples_steps)
            save_and_log_model_checkpoint(model, epoch, avg_train_loss, avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            log_best_model(model, best_val_loss)

        print(
            f"Epoch [{epoch}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )

    if use_wandb:
        finish_wandb()

    print("\nTraining complete.")
    return model
