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

        # Make a deep copy of the final output layer for later use in accurate forward pass
        self.copied_cov_out = copy.deepcopy(self.conv_out)

        # Remove the final output layer from the model to use it as a feature extractor
        # This allows extracting intermediate features before the final classification layer
        self.feature_extractor = diff_model
        self.feature_extractor.output_conv = nn.Identity()

        # Wrap the final output layer with a DiagLaplace approximation
        # This provides Bayesian uncertainty estimation on the classification output
        self.conv_out_la = DiagLaplace(
            nn.Sequential(
                self.conv_out, nn.Flatten(1)
            ),  # Flatten output for classification likelihood
            likelihood="regression",  # Specify classification likelihood model (regression in the case of image ddpm)
            sigma_noise=1.0,  # Noise parameter for the likelihood
            prior_precision=1,  # Precision of the Gaussian prior
            prior_mean=0.0,  # Mean of the prior
            temperature=1.0,  # Temperature scaling for predictions
            backend=BackPackEF,  # Backend for curvature/Hessian estimation
        )

        # Fit the Laplace approximation using the provided data loader
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

        # Save current parameters as MAP estimate
        self.conv_out_la.mean = parameters_to_vector(
            self.conv_out_la.model.parameters()
        ).detach()

        # Peek at one batch to set output size
        x_t, t, labels = next(iter(train_loader))
        with torch.no_grad():
            feats = self.feature_extractor(
                x_t.to(self.conv_out_la._device),
                t.to(self.conv_out_la._device),
                y=labels.to(self.conv_out_la._device),
            )
            out = self.conv_out_la.model(feats)

        # Expect unflattened image output: [B, C, H, W]
        self.conv_out_la.n_outputs = out[0].numel()  # total image size
        setattr(self.conv_out_la.model, "output_size", self.conv_out_la.n_outputs)

        N = len(train_loader.dataset)

        for x_t, t, labels in tqdm(train_loader, desc="Fitting", leave=False):
            self.conv_out_la.model.zero_grad()

            x_t, t, labels = [
                tensor.to(self.conv_out_la._device) for tensor in (x_t, t, labels)
            ]

            with torch.no_grad():
                feats = self.feature_extractor(x_t, t, y=labels)

            # Flatten both model output and target before computing loss
            target = x_t.view(x_t.size(0), -1)  # [B, C*H*W]

            loss_b, H_b = self.conv_out_la._curv_closure(feats, target, N)

            self.conv_out_la.loss += loss_b
            self.conv_out_la.H += H_b

        self.conv_out_la.n_data += N

    # NOTE: This can be changed so that you pass the n_samples to estimate the uncertainty
    def forward(self, x, t, y=None):
        """
        Forward pass using the Laplace-approximated model.
        Returns both predictive logits and uncertainty estimates (variance).
        """
        self.feature_extractor.eval()
        with torch.no_grad():
            feats = self.feature_extractor(x, t, y=y)

        # Predictive mean and variance using MC sampling
        mean, var = self.conv_out_la(
            feats, pred_type="nn", link_approx="mc", n_samples=50
        )

        # NOTE: Reshape the output for convenience later
        # Reshape [B, 784] -> [B, 1, 28, 28]
        B = x.size(0)
        mean = mean.view(B, 1, 28, 28)
        var = var.view(B, 1, 28, 28)

        return mean, var

    def accurate_forward(self, x, t, y=None):
        """
        Forward pass without uncertainty estimation.
        Directly uses the copied final output layer to get deterministic logits.
        """
        self.feature_extractor.eval()
        with torch.no_grad():
            feats = self.feature_extractor(x, t, y=y)
            logits = self.copied_cov_out(feats)
        return logits
