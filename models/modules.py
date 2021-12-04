import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from .utils import *
from torchvision.utils import make_grid

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


# RealNVP
###################################################################
class WeightNormConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super(WeightNormConv2d, self).__init__()
        self.conv = nn.utils.weight_norm(nn.Conv2d(*args, **kwargs))  # (C_in, C_out, K, K), re-parameterization: dim=0

    def forward(self, x):
        # x: (B, C, H, W)
        return self.conv(x)


class ActNorm(nn.Module):
    # Normalize each channel of an instance
    def __init__(self, num_channels):
        super(ActNorm, self).__init__()
        self.num_channels = num_channels
        self.log_scale = nn.Parameter(torch.zeros((1, num_channels, 1, 1), dtype=torch.float32))
        self.shift = nn.Parameter(torch.zeros((1, num_channels, 1, 1), dtype=torch.float32))
        self.if_init = False

    def forward(self, x, reverse=False):
        # x: (B, C, H, W)
        if not self.if_init:
            self.if_init = True
            x_std = torch.std(x,dim=[0, 2, 3], keepdim=True)  # (1, C, 1, 1)
            self.log_scale.data = -torch.log(x_std)
            self.shift.data = -torch.mean(x, dim=[0, 2, 3], keepdim=True) / x_std

        if reverse:
            x_out = (x - self.shift) * (-self.log_scale).exp()
            return x_out, None

        x_out = x * self.log_scale.exp() + self.shift  # (B, C, H, W)
        log_det = self.log_scale  # (1, C, 1, 1), broadcast later

        return x_out, log_det


class ResnetBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResnetBlock, self).__init__()
        self.num_channels = num_channels
        self.net = nn.Sequential(
            WeightNormConv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=1),
            nn.ReLU(),
            WeightNormConv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            WeightNormConv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=1)
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = x + self.net(x)  # (B, C, H, W)

        return x


class ResnetLittle(nn.Module):
    def __init__(self, in_channels, out_channels, num_filters=128, num_blocks=8):
        super(ResnetLittle, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_filters = num_filters
        self.num_blocks = num_blocks
        net = [WeightNormConv2d(in_channels=in_channels, out_channels=num_filters, kernel_size=3, padding=1), nn.ReLU()]
        for _ in range(num_blocks):
            net.append(ResnetBlock(num_filters))
        net.extend([nn.ReLU(), WeightNormConv2d(in_channels=num_filters, out_channels=out_channels, kernel_size=3,
                                                padding=1)])
        self.net = nn.Sequential(*net)

    def forward(self, x):
        # x: (B, C, H, W)
        # x_out: (B, 2 * C, H, W)
        return self.net(x)


class AffineCheckerboard(nn.Module):
    def __init__(self, mask_type, in_channels, num_filters=128, num_blocks=8, device=None):
        assert mask_type == 0 or mask_type == 1, "mask_type should be 0 or 1"
        super(AffineCheckerboard, self).__init__()
        self.mask_type = mask_type
        self.resnet = ResnetLittle(in_channels=in_channels, out_channels=2 * in_channels, num_filters=num_filters,
                                   num_blocks=num_blocks)
        self.scale_for_log_scale = nn.Parameter(torch.zeros(1))
        self.shift_for_log_scale = nn.Parameter(torch.zeros(1))  # these two are broadcast later
        self.device = device if device is not None else torch.device("cuda")

    def forward(self, x, reverse=False):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        mask = (torch.arange(0, H).unsqueeze(1) + torch.arange(0, W) + self.mask_type) % 2
        self.register_buffer("mask", mask.to(self.device).view(1, 1, H, W))
        # print(f"mask: {self.mask[0, 0]}")
        x_ = self.mask * x
        log_s, t = self.resnet(x_).chunk(2, dim=1)  # (B, 2 * C, H, W) -> (B, C, H, W) each
        log_s = log_s * self.scale_for_log_scale + self.shift_for_log_scale
        log_s, t = log_s * (1 - self.mask), t * (1 - self.mask)  # now operate on unconditioned part, (B, C, H, W) each
        if reverse:
            x_out = (x - t) * (-log_s).exp()
            return x_out, None

        x_out = x * log_s.exp() + t
        # log_det = log_s: (B, C, H, W)
        return x_out, log_s


class AffineChannel(nn.Module):
    def __init__(self, modify_top, in_channels, num_filters=128, num_blocks=8):
        super(AffineChannel, self).__init__()
        self.modify_top = modify_top
        self.resnet = ResnetLittle(in_channels=in_channels, out_channels=2 * in_channels, num_filters=num_filters,
                                   num_blocks=num_blocks)
        self.scale_for_log_scale = nn.Parameter(torch.zeros(1))
        self.shift_for_log_scale = nn.Parameter(torch.zeros(1))  # these two are broadcast later

    def forward(self, x, reverse=False):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        assert C % 2 == 0, "number of channels should be even"
        if self.modify_top:
            x_off = x[:, C // 2:, ...]  # (B, C // 2, H, W) each
            x_on = x[:, :C // 2, ...]
        else:
            x_on = x[:, C // 2:, ...]
            x_off = x[:, :C // 2, ...]
        log_s, t = self.resnet(x_off).chunk(2, dim=1)  # (B, C, H, W) -> (B, C // 2, H, W) each
        log_s = log_s * self.scale_for_log_scale + self.shift_for_log_scale
        if reverse:
            x_on_out = (x_on - t) * (-log_s).exp()
            if self.modify_top:
                return torch.cat([x_on_out, x_off], dim=1), None
            else:
                return torch.cat([x_off, x_on_out], dim=1), None

        x_on_out = x_on * log_s.exp() + t
        if self.modify_top:
            # log_det: (B, C, H, W)
            return torch.cat([x_on_out, x_off], dim=1), torch.cat([log_s, torch.zeros_like(log_s)], dim=1)
        else:
            return torch.cat([x_off, x_on_out], dim=1), torch.cat([torch.zeros_like(log_s), log_s], dim=1)

# MNISTVAE
###################################################################


class MNISTEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MNISTEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
        )
        self.fc = nn.Linear(3136, out_channels)

    def forward(self, X):
        # X: (B, 3, 28, 28)
        batch_size = X.shape[0]
        X = self.convs(X)  # (B, 64, 7, 7)
        X = X.view(batch_size, -1)  # (B, 3136)
        X = self.fc(X)  # (B, 2 * lat_dim) or (B, lat_dim)

        return X


class MNISTDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MNISTDecoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 1568),
            nn.ReLU()
        )
        self.convs = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, padding=1, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, X):
        # X: (B, lat_dim)
        batch_size = X.shape[0]
        X = self.fc(X)  # (B, 1568)
        X = X.view(batch_size, 32, 7, 7)
        X = self.convs(X)  # (B, 3, 28, 28)

        return X


class IntWarper(nn.Module):
    def __init__(self, lat_dim, if_cross_entropy=False):
        super(IntWarper, self).__init__()
        self.lat_dim = lat_dim
        self.if_cross_entropy = if_cross_entropy
        in_channels = 3 * 256 + 3 if if_cross_entropy else 6
        self.encoder = MNISTEncoder(in_channels, lat_dim)
        out_channels = 256 if if_cross_entropy else 2
        self.decoder = MNISTDecoder(lat_dim, out_channels)

    def forward(self, X1, X2):
        # X1, X2: (B, C, H, W) each
        X = torch.cat([X1, X2], dim=1)  # (B, 2 * C, H, W) or (B, K * C + C, H, W)
        Z = self.encoder(X)  # (B, lat_dim)
        warp_field = self.decoder(Z)  # (B, 2, H, W) or (B, K, H, W)

        return warp_field


class ShapeWarper(nn.Module):
    def __init__(self, lat_dim, if_cross_entropy=False):
        super(ShapeWarper, self).__init__()
        self.lat_dim = lat_dim
        self.if_cross_entropy = if_cross_entropy
        in_channels = 3 * 256 + 3 if if_cross_entropy else 6
        self.encoder = MNISTEncoder(in_channels, lat_dim)
        out_channels = 2
        self.decoder = MNISTDecoder(lat_dim, out_channels)

    def forward(self, X1, X2):
        # X1, X2: (B, C, H, W) each
        X = torch.cat([X1, X2], dim=1)  # (B, 2 * C, H, W)
        Z = self.encoder(X)  # (B, lat_dim)
        warp_field = self.decoder(Z)  # (B, 2, H, W)

        return warp_field


class MNISTVAE(nn.Module):
    def __init__(self, lat_dim, lat_split_ind, if_cross_entropy=False, device=None):
        assert 0 <= lat_split_ind < lat_dim, "invalid lat_split_ind"
        super(MNISTVAE, self).__init__()
        self.lat_dim = lat_dim
        self.lat_split_ind = lat_split_ind
        self.device = device if device is not None else torch.device("cuda")
        self.encoder = MNISTEncoder(3, 2 * lat_dim)
        self.if_cross_entropy = if_cross_entropy
        out_channels = 3 * 256 if if_cross_entropy else 3
        self.decoder = MNISTDecoder(lat_dim, out_channels)

    def forward(self, X1, X2):
        # X1, X2: (B, 3, 28, 28) each
        # reconstruction
        # X1 = 2 * X1 - 1
        # X2 = 2 * X2 - 1
        Z1_mu, Z1_log_sigma = torch.chunk(self.encoder(X1), 2, dim=1)  # (B, lat_dim) each
        Z2_mu, Z2_log_sigma = torch.chunk(self.encoder(X2), 2, dim=1)
        Z1 = Z1_mu + torch.randn_like(Z1_mu) * Z1_log_sigma.exp()
        Z2 = Z2_mu + torch.randn_like(Z2_mu) * Z2_log_sigma.exp()
        X1 = self.decoder(Z1)
        X2 = self.decoder(Z2)

        # disentanglement
        t = torch.rand((1,)).to(self.device)
        Z1_int, Z1_shape = Z1_mu[:, :self.lat_split_ind], Z1_mu[:, self.lat_split_ind:] # (B, lat_dim') for each
        Z2_int, Z2_shape = Z2_mu[:, :self.lat_split_ind], Z2_mu[:, self.lat_split_ind:]
        Z_int_interp = Z1_int + t * (Z2_int - Z1_int)
        Z_shape_interp = Z1_shape + t * (Z2_shape - Z1_shape)
        Z_int_interp_shape_1, Z_int_interp_shape_2 = torch.cat([Z_int_interp, Z1_shape], dim=1), \
                                                     torch.cat([Z_int_interp, Z2_shape], dim=1)
        Z_int_1_shape_interp, Z_int_2_shape_interp = torch.cat([Z1_int, Z_shape_interp], dim=1), \
                                                     torch.cat([Z2_int, Z_shape_interp], dim=1)
        X_int_interp_shape_1, X_int_interp_shape_2 = self.decoder(Z_int_interp_shape_1), \
                                                     self.decoder(Z_int_interp_shape_2)
        X_int_1_shape_interp, X_int_2_shape_interp = self.decoder(Z_int_1_shape_interp), \
                                                     self.decoder(Z_int_2_shape_interp)

        return Z1_mu, Z1_log_sigma, Z2_mu, Z2_log_sigma,\
               X1, X2, X_int_interp_shape_1, X_int_interp_shape_2, X_int_1_shape_interp, X_int_2_shape_interp

    @torch.no_grad()
    def interpolate(self, X1, X2, time_steps=5):
        # X1, X2: (1, 3, 28, 28)
        time_intervals = np.linspace(0, 1, time_steps)
        samples = torch.zeros(time_steps, time_steps, *X1.shape[1:], dtype=X1.dtype)
        # X1 = 2 * X1 - 1
        # X2 = 2 * X2 - 1
        Z1, _ = self.encoder(X1).chunk(2, dim=1)  # (1, lat_dim)
        Z2, _ = self.encoder(X2).chunk(2, dim=1)
        Z1_int, Z1_shape = Z1[:, :self.lat_split_ind], Z1[:, self.lat_split_ind:]
        Z2_int, Z2_shape = Z2[:, :self.lat_split_ind], Z2[:, self.lat_split_ind:]
        for t_row in range(time_steps):
            for t_col in range(time_steps):
                Z_int_interp = Z1_int + time_intervals[t_row] * (Z2_int - Z1_int)  # (1, lat_dim')
                Z_shape_interp = Z1_shape + time_intervals[t_col] * (Z2_shape - Z1_shape)
                if not self.if_cross_entropy:
                    X_interp = self.decoder(torch.cat([Z_int_interp, Z_shape_interp], dim=1))  # (1, 3, 28, 28)
                else:
                    X_interp = self.decoder(torch.cat([Z_int_interp, Z_shape_interp], dim=1))  # (1, 3 * 256, 28, 28)
                    X_interp = X_interp.view(1, 3, 256, 28, 28)
                    X_interp = torch.argmax(X_interp, dim=2).float() / 255
                samples[t_row, t_col] = X_interp[0]
        samples = samples.view(-1, *X1.shape[1:])  # (time_step^2, 3, 28, 28)
        samples_grid = make_grid(samples, nrow=time_steps)  # (C, H', W')
        plt.imshow(samples_grid.permute(1, 2, 0).detach().cpu().numpy())
        # plt.colorbar()
        plt.show()


# Normalizer
###################################################################
class Normalizer(nn.Module):
    def __init__(self, num_layers=3, kernel_size=1, in_channels=1, intermediate_channels=16):
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        super(Normalizer, self).__init__()
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.intermediate_channels = intermediate_channels
        padding = (kernel_size - 1) // 2
        layers = [nn.Conv2d(in_channels, intermediate_channels, kernel_size, padding=padding)]
        for _ in range(num_layers - 1):
            layers += [nn.ReLU(),
                       nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size, padding=padding)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, C, H, W)

        # x_out: (B, C, H, W)
        return self.layers(x)
