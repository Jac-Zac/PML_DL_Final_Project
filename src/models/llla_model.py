import copy

import torch
import torch.nn as nn
from laplace.baselaplace import DiagLaplace

# BackPack curvature estimator backend
from laplace.curvature.backpack import BackPackEF
from torch.nn.utils import parameters_to_vector
from tqdm import tqdm


class LaplaceApproxModel(nn.Module):
    def __init__(self, diff_model, dataloader, args, config):
        super().__init__()
        self.args, self.config = args, config

        # Extract the final output convolutional layer of the original diffusion model
        self.conv_out = diff_model.output_conv

        # Make a deep copy of the final output layer for deterministic evaluation later
        self.copied_conv_out = copy.deepcopy(self.conv_out)

        # Remove the final output layer from the model to use it as a feature extractor
        self.feature_extractor = diff_model
        self.feature_extractor.output_conv = nn.Identity()

        # Wrap the final output layer with a DiagLaplace approximation
        # This gives a Bayesian linear model with a Gaussian posterior over weights
        self.conv_out_la = DiagLaplace(
            nn.Sequential(
                self.conv_out, nn.Flatten(1, -1)
            ),  # Flatten output to 2D tensor [B, D]
            likelihood="regression",  # Gaussian likelihood for prediction
            sigma_noise=1.0,  # Observation std
            prior_precision=1.0,  # Prior precision (lambda)
            prior_mean=0.0,  # Prior mean
            temperature=1.0,  # Temperature scaling
            backend=BackPackEF,  # Curvature estimator backend
        )

        # Fit the Laplace approximation using the training data
        self.fit(dataloader)

    def fit(self, train_loader, override=True):
        """
        Fit the Laplace approximation by estimating the curvature (Hessian) of the
        loss around the MAP estimate using the training data.
        """
        if override:
            self.conv_out_la._init_H()
            self.conv_out_la.loss, self.conv_out_la.n_data = 0, 0

        self.conv_out_la.model.eval()

        # Save the MAP (mean) estimate of the parameters
        self.conv_out_la.mean = parameters_to_vector(
            self.conv_out_la.model.parameters()
        ).detach()

        # Peek at one batch to set output size
        x_t, t, _, y = next(iter(train_loader))
        device = self.conv_out_la._device

        with torch.no_grad():
            feats = self.feature_extractor(
                x_t.to(device),
                t.to(device),
                y=y.to(device),
            )
            out = self.conv_out_la.model(feats)

        # Total number of outputs per sample (e.g. 28x28 = 784 for MNIST)
        self.conv_out_la.n_outputs = out[0].numel()
        setattr(self.conv_out_la.model, "output_size", self.conv_out_la.n_outputs)

        # Size of the data loader
        N = len(train_loader.dataset)

        for x_t, t, pred, y in tqdm(train_loader, desc="Fitting Laplace", leave=False):
            self.conv_out_la.model.zero_grad()

            # Move batch to the appropriate device
            x_t, t, pred, y = [tensor.to(device) for tensor in (x_t, t, pred, y)]

            with torch.no_grad():
                feats = self.feature_extractor(x_t, t, y=y)

            # Use the true noise added during forward diffusion as regression target
            targets = pred.view(pred.size(0), -1)  # flatten: [B, D]

            # Compute per-batch curvature (Hessian approx) and loss
            loss_b, H_b = self.conv_out_la._curv_closure(feats, targets, N)

            self.conv_out_la.loss += loss_b
            self.conv_out_la.H += H_b

        self.conv_out_la.n_data += N

    def forward(self, x, t, y=None):
        """
        Forward pass using the Laplace-approximated model.
        Returns both predictive mean and variance for uncertainty estimation.
        """
        self.feature_extractor.eval()
        with torch.no_grad():
            feats = self.feature_extractor(x, t, y=y)

        # Predictive mean and variance via Monte Carlo approximation
        mean, var = self.conv_out_la(
            feats, pred_type="nn", link_approx="mc", n_samples=100
        )

        # Reshape [B, 784] -> [B, 1, 28, 28] for image-shaped output
        B = x.size(0)
        img_size = self.config.data.image_size
        mean = mean.view(B, 1, img_size, img_size)
        var = var.view(B, 1, img_size, img_size)

        return mean, var

    def accurate_forward(self, x, t, y=None):
        """
        Forward pass without uncertainty.
        This uses the copied output layer directly for deterministic predictions.
        """
        self.feature_extractor.eval()
        with torch.no_grad():
            feats = self.feature_extractor(x, t, y=y)
            logits = self.copied_conv_out(feats)

        # Reshape logits to image format
        B = x.size(0)
        img_size = self.config.data.image_size
        return logits.view(B, 1, img_size, img_size)
