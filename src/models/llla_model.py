import copy

import torch
import torch.nn as nn
from laplace.baselaplace import DiagLaplace
from laplace.curvature.backpack import BackPackEF
from torch.nn.utils import parameters_to_vector


class CustomModel(nn.Module):
    def __init__(self, diff_model, dataloader, args, config):
        super().__init__()
        self.args, self.config = args, config

        # Extract the final output layer
        self.conv_out = diff_model.output_conv
        self.copied_cov_out = copy.deepcopy(self.conv_out)

        # Remove final layer for feature extraction
        self.feature_extractor = diff_model
        self.feature_extractor.output_conv = nn.Identity()

        # Use classification likelihood for conditioning task
        self.conv_out_la = DiagLaplace(
            nn.Sequential(self.conv_out, nn.Flatten(1)),
            likelihood="classification",
            sigma_noise=1.0,
            prior_precision=1,
            prior_mean=0.0,
            temperature=1.0,
            backend=BackPackEF,
        )

        self.fit(dataloader)

    def fit(self, train_loader, override=True):
        if override:
            self.conv_out_la._init_H()
            self.conv_out_la.loss, self.conv_out_la.n_data = 0, 0

        self.conv_out_la.model.eval()
        self.conv_out_la.mean = parameters_to_vector(
            self.conv_out_la.model.parameters()
        ).detach()

        # Inspect output dimensions
        (X, t), labels = next(iter(train_loader))
        with torch.no_grad():
            feats = self.feature_extractor(
                X.to(self.conv_out_la._device),
                t.to(self.conv_out_la._device),
                y=labels.to(self.conv_out_la._device),
            )
            out = self.conv_out_la.model(feats)
        self.conv_out_la.n_outputs = out.shape[-1]
        setattr(self.conv_out_la.model, "output_size", self.conv_out_la.n_outputs)

        N = len(train_loader.dataset)
        for i, ((X, t), labels) in enumerate(train_loader):
            print(f"Batch {i}")
            self.conv_out_la.model.zero_grad()
            X, t, labels = [x.to(self.conv_out_la._device) for x in (X, t, labels)]

            with torch.no_grad():
                feats = self.feature_extractor(X, t, y=labels)

            loss_b, H_b = self.conv_out_la._curv_closure(feats, labels, N)

            self.conv_out_la.loss += loss_b
            self.conv_out_la.H += H_b

        self.conv_out_la.n_data += N

    def forward(self, x, t, y=None):
        self.feature_extractor.eval()
        with torch.no_grad():
            feats = self.feature_extractor(x, t, y=y)

        logits, var = self.conv_out_la(
            feats, pred_type="nn", link_approx="mc", n_samples=50
        )
        return logits, var

    def accurate_forward(self, x, t, y=None):
        self.feature_extractor.eval()
        with torch.no_grad():
            feats = self.feature_extractor(x, t, y=y)
            logits = self.copied_cov_out(feats)
        return logits
