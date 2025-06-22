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
            likelihood="classification",  # Specify classification likelihood model
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
        # NOTE: Add here the modification that you need to do to pass correct samples to it not in the
        # dataloader and you can use the Diffusion class

        if override:
            # Initialize the Hessian approximation matrix and reset loss counters
            self.conv_out_la._init_H()
            self.conv_out_la.loss, self.conv_out_la.n_data = 0, 0

        # Set model to evaluation mode to disable dropout/batchnorm updates
        self.conv_out_la.model.eval()

        # Vectorize and save the current parameters of the Laplace-approximated model
        self.conv_out_la.mean = parameters_to_vector(
            self.conv_out_la.model.parameters()
        ).detach()

        # Look at one batch from the data loader to determine output dimensions
        X, t, labels = next(iter(train_loader))
        with torch.no_grad():
            # Extract features using the feature extractor (without the final layer)
            feats = self.feature_extractor(
                X.to(self.conv_out_la._device),
                t.to(self.conv_out_la._device),
                y=labels.to(self.conv_out_la._device),
            )
            # Get output logits from the Laplace-approximated final layer
            out = self.conv_out_la.model(feats)

        # Store the number of output classes/units
        self.conv_out_la.n_outputs = out.shape[-1]
        setattr(self.conv_out_la.model, "output_size", self.conv_out_la.n_outputs)

        # Total number of training samples
        N = len(train_loader.dataset)

        # Loop through all batches to accumulate curvature (Hessian) and loss statistics
        for X, t, labels in tqdm(train_loader, desc="Fitting", leave=False):
            # Reset gradients
            self.conv_out_la.model.zero_grad()

            # Move inputs and labels to the correct device (CPU/GPU)
            X, t, labels = [x.to(self.conv_out_la._device) for x in (X, t, labels)]

            with torch.no_grad():
                # Extract features from inputs
                feats = self.feature_extractor(X, t, y=labels)

            # Compute loss and curvature approximation for this batch
            loss_b, H_b = self.conv_out_la._curv_closure(feats, labels, N)

            # Accumulate loss and Hessian
            self.conv_out_la.loss += loss_b
            self.conv_out_la.H += H_b

        # Update the number of data points seen for Laplace approximation
        self.conv_out_la.n_data += N

    # NOTE: This can be changed so that you pass the n_samples to estimate the uncertainty
    def forward(self, x, t, y=None):
        """
        Forward pass using the Laplace-approximated model.
        Returns both predictive logits and uncertainty estimates (variance).
        """
        self.feature_extractor.eval()
        with torch.no_grad():
            # Extract features from input
            feats = self.feature_extractor(x, t, y=y)

        # Pass features through Laplace layer to get predictions and uncertainty
        logits, var = self.conv_out_la(
            feats, pred_type="nn", link_approx="mc", n_samples=50
        )
        return logits, var

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
