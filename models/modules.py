import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import *


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, feature_dim=2):
        assert feature_dim in [2, 3], "only supports 2D and 3D data"
        super(DownBlock, self).__init__()
        self.feature_dim = feature_dim
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.net_before_pool = nn.Sequential(
            get_conv(feature_dim, in_channels=in_channels, out_channels=out_channels,
                     kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            get_bn(feature_dim, num_features=out_channels),
            get_conv(feature_dim, in_channels=out_channels, out_channels=out_channels,
                     kernel_size=3, stride=1, padding=1)
        )
        self.pool = get_pool(feature_dim, kernel_size=2, stride=2)

    def forward(self, x):
        x_before_pool = self.net_before_pool(x)

        return self.pool(x), x_before_pool
