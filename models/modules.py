import numpy as np
import torch
import torch.nn.functional as F

from .utils import *

# U-Net
###################################################################


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

        return self.pool(x_before_pool), x_before_pool


class GridAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim, feature_dim=2, inter_dim=None, subsample_factor=2):
        # enc_dim, dec_dim are number of channels resp.
        assert feature_dim in [2, 3], "only supports 2D and 3D features"
        super(GridAttention, self).__init__()
        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        self.inter_dim = inter_dim if inter_dim is not None else enc_dim // 2
        self.feature_dim = feature_dim
        self.interp_mode = "bilinear" if feature_dim == 2 else "trilinear"
        self.att = None

        # att: W_theta * x + W_phi * g + bias
        self.theta = get_conv(feature_dim, in_channels=enc_dim, out_channels=self.inter_dim,
                              kernel_size=subsample_factor, stride=subsample_factor)
        self.phi = get_conv(feature_dim, in_channels=dec_dim, out_channels=self.inter_dim, kernel_size=1)
        self.psi = get_conv(feature_dim, in_channels=self.inter_dim, out_channels=1, kernel_size=1)
        self.out_conv = nn.Sequential(
            get_conv(feature_dim, in_channels=enc_dim, out_channels=enc_dim, kernel_size=1),
            get_bn(feature_dim, num_features=enc_dim)
        )

    def forward(self, enc, dec):
        enc_theta = self.theta(enc)
        dec_phi = self.phi(dec)
        dec_phi = F.interpolate(dec_phi, size=enc_theta.shape[2:], mode=self.interp_mode, align_corners=False)
        merged = enc_theta + dec_phi
        att = self.psi(F.relu(merged))
        att = torch.sigmoid(att)
        att = F.interpolate(att, size=enc.shape[2:], mode=self.interp_mode, align_corners=False)  # (B, 1, ...)
        out = self.out_conv(att * enc)
        self.att = att

        return out, att


class IdentityAttention(nn.Module):
    def __init__(self):
        super(IdentityAttention, self).__init__()

    def forward(self, enc, dec):
        return enc, None


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, enc_dim, feature_dim=2, attention=True):
        assert feature_dim in [2, 3], "only supports 2D and 3D features"
        super(UpBlock, self).__init__()
        self.feature_dim = feature_dim
        self.attention = GridAttention(enc_dim, out_channels, feature_dim) if attention else IdentityAttention()
        self.up = nn.Sequential(
            get_conv_transpose(feature_dim, in_channels=in_channels, out_channels=in_channels, kernel_size=4,
                               stride=2, padding=1),
            nn.ReLU(),
            get_bn(feature_dim, num_features=in_channels),
            get_conv(feature_dim, in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            get_bn(feature_dim, num_features=out_channels)
        )
        self.out_convs = nn.Sequential(
            get_conv(feature_dim, in_channels=enc_dim + out_channels, out_channels=out_channels,
                     kernel_size=3, padding=1),
            nn.ReLU(),
            get_bn(feature_dim, num_features=out_channels),
            get_conv(feature_dim, in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            get_bn(feature_dim, num_features=out_channels)
        )

    def forward(self, enc, dec):
        # "dec" comes from the layer below, so up-sample it first
        dec_up = self.up(dec)
        enc_crop, dec_up = auto_crop(enc, dec_up)
        enc_att, _ = self.attention(enc_crop, dec_up)
        merged = concat_enc_dec(enc_att, dec_up)
        out = self.out_convs(merged)

        return out

# MADE
###################################################################


class MaskedLinear(nn.Linear):
    def __init__(self, n_din, n_out):
        super(MaskedLinear, self).__init__(n_din, n_out)
        self.register_buffer("mask", torch.ones_like(self.weight))

    def set_mask(self, mask: np.ndarray):
        assert isinstance(mask, np.ndarray), "please input a np.ndarray"
        assert mask.shape == self.weight.shape, f"wrong shape: target shape is {self.weight.shape}"
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8)))

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


