from .modules import *


class Encoder(nn.Module):
    def __init__(self, num_down_blocks, in_channels=None, init_out_channels=None, feature_dim=2):
        # num_down_blocks include the first non-down-sampling-block
        assert feature_dim in [2, 3], "only supports 2D and 3D data"
        super(Encoder, self).__init__()
        self.num_down_blocks = num_down_blocks
        self.in_channels = in_channels if in_channels is not None else 3
        self.init_out_channels = init_out_channels if init_out_channels is not None else 32
        self.feature_dim = feature_dim

        self.conv_in = nn.Sequential(
            get_conv(feature_dim, in_channels=self.in_channels, out_channels=self.init_out_channels,
                     kernel_size=3, padding=1),
            nn.ReLU(),
            get_bn(feature_dim, num_features=self.init_out_channels),
            get_conv(feature_dim, in_channels=self.init_out_channels, out_channels=self.init_out_channels,
                     kernel_size=3, padding=1),
            nn.ReLU(),
            get_bn(feature_dim, num_features=self.init_out_channels)
        )
        self.blocks = nn.ModuleList()
        in_channels = self.init_out_channels
        out_channels = in_channels * 2
        for _ in range(num_down_blocks - 1):
            # print(f"inside: in_channels: {in_channels}, out_channels: {out_channels}")
            self.blocks.append(DownBlock(in_channels, out_channels, feature_dim))
            in_channels = out_channels
            out_channels *= 2

    def forward(self, x):
        before_pools = []
        # print(x.shape)
        x = self.conv_in(x)
        before_pools.append(x)
        for block in self.blocks:
            x, before_pool = block(x)
            before_pools.append(before_pool)

        return x, before_pools


class UNet(nn.Module):
    def __init__(self, num_down_blocks, target_channels, in_channels=None, feature_dim=2, encoder=None, **kwargs):
        assert feature_dim in [2, 3], "only supports 2D and 3D data"
        super(UNet, self).__init__()
        self.num_down_blocks = num_down_blocks
        self.feature_dim = feature_dim
        if encoder is None:
            self.encoder = Encoder(num_down_blocks, feature_dim=feature_dim,
                                   in_channels=in_channels, init_out_channels=kwargs.get("init_out_channels", None))
        self.bottom_channels = self.encoder.init_out_channels * (2 ** (num_down_blocks - 1))
        self.bottom_conv = get_conv(feature_dim, in_channels=self.bottom_channels, out_channels=self.bottom_channels,
                                    kernel_size=3, padding=1)
        self.up_blocks = nn.ModuleList()
        in_channels = self.bottom_channels
        out_channels = in_channels // 2
        for _ in range(num_down_blocks - 1):
            self.up_blocks.append(UpBlock(in_channels, out_channels, out_channels, feature_dim))
            in_channels = out_channels
            out_channels //= 2
        self.out_conv = get_conv(feature_dim, in_channels=in_channels, out_channels=target_channels,
                                 kernel_size=1)

    def forward(self, x):
        x_enc, before_pools = self.encoder(x)
        x_dec = self.bottom_conv(x_enc)
        for enc, up_block in zip(reversed(before_pools[:-1]), self.up_blocks):
            # print(f"inside: enc: {enc.shape}, x_dec: {x_dec.shape}")
            x_dec = up_block(enc, x_dec)

        return self.out_conv(x_dec)
