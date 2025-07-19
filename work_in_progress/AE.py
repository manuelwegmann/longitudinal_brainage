import torch
import torch.nn as nn


class AE_MLP(nn.Module):
    """
    MLP model for flattened autoencoder features + optional metadata.

    Args:
        input_feature_size: int, number of features after flattening the autoencoder output.
        additional_feature: int, number of optional metadata features.
    """
    def __init__(self, input_feature_size, additional_feature=0):
        super(AE_MLP, self).__init__()
        
        self.input_feature_size = input_feature_size
        self.additional_feature = additional_feature
        total_in = input_feature_size + additional_feature
        
        self.mlp = nn.Sequential(
            nn.Linear(total_in, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 96),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(96, 1, bias=False)
        )

    def forward(self, x, meta=None):
        """
        Args:
            x: torch.Tensor, shape [B, *], flattened feature map from autoencoder
            meta: torch.Tensor, shape [B, n_meta] if provided
        """
        if meta is not None:
            x = torch.cat((x, meta), dim=1)
        return self.mlp(x)