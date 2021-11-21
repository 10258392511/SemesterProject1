import torch
import torch.nn as nn

from .modules import ActNorm, AffineCheckerboard, AffineChannel

class RealNVP(nn.Module):
    def __init__(self, in_channels, device=None):
        super(RealNVP, self).__init__()
        self.in_channels = in_channels
        self.device = device if device is not None else torch.device("cuda")

        checkerboard1 = [AffineCheckerboard(mask_type=1, in_channels=in_channels, device=self.device)]
        for mask_type in [0, 1, 0]:
            checkerboard1.append(ActNorm(num_channels=in_channels))
            AffineCheckerboard(mask_type=mask_type, in_channels=in_channels, device=self.device)
        self.checkerboard1 = nn.ModuleList(checkerboard1)

        # after self._squeeze(.): (B, C, H, W) -> (B, 4C, H // 2, W // 2)
        # only half the channels are used for conditioning
        channel = [AffineChannel(modify_top=True, in_channels=in_channels * 2)]
        for modify_top in [False, True]:
            channel.append(ActNorm(num_channels=in_channels * 2))
            channel.append(AffineChannel(modify_top=modify_top, in_channels=in_channels * 2))
        self.channel = nn.ModuleList(channel)

        # after self._unsqueeze(.): (B, 4C, H // 2, W // 2) -> (B, C, H, W)
        checkerboard2 = [AffineCheckerboard(mask_type=1, in_channels=in_channels, device=self.device)]
        for mask_type in [0, 1]:
            checkerboard1.append(ActNorm(num_channels=in_channels))
            AffineCheckerboard(mask_type=mask_type, in_channels=in_channels, device=self.device)
        self.checkerboard2 = nn.ModuleList(checkerboard2)

    def _squeeze(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, "H and W must be even"
        x = x.reshape(B, C, H // 2, 2, W // 2, 2).contiguous()  # (B, C, H // 2, 2, W // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4)  # (B, C, 2, 2, H // 2, W // 2)
        x = x.reshape(B, -1, H // 2, W // 2).contiguous()  # (B, 4C, H // 2, W // 2)

        return x

    def _unsqueeze(self, x):
        # x: (B, 4C, H // 2, W // 2)
        B, C, H, W = x.shape
        assert C % 4 == 0, "C must be a multiple of 4"
        x = x.reshape(B, C // 4, 2, 2, H, W).contiguous()  # (B, C, 2, 2, H // 2, W // 2)
        x = x.permute(0, 1, 4, 2, 5, 3)  # (B, C, H // 2, 2, W // 2, 2)
        x = x.reshape(B, C // 4, H * 2, W * 2).contiguous()  # (B, C, H, W)

        return x

    def forward(self, x, reverse=False):
        # x: (B, C, H, W)
        if reverse:
            for layer in reversed(self.checkerboard2):
                x, _ = layer(x, reverse=True)
            x = self._squeeze(x)
            for layer in reversed(self.channel):
                x, _ = layer(x, reverse=True)
            x = self._unsqueeze(x)
            for layer in reversed(self.checkerboard1):
                x, _ = layer(x, reverse=True)

            return x, None

        log_det_out = torch.zeros_like(x)
        for layer in self.checkerboard1:
            x, log_det = layer(x)
            log_det_out += log_det

        x, log_det_out = self._squeeze(x), self._squeeze(log_det_out)  # both (B, 4C, H // 2, W // 2)
        # print(f"{x.shape}, {log_det_out.shape}")
        for layer in self.channel:
            x, log_det = layer(x)
            log_det_out += log_det

        x, log_det_out = self._unsqueeze(x), self._unsqueeze(log_det_out)  # both (B, C, H, W)
        for layer in self.checkerboard2:
            x, log_det = layer(x)
            log_det_out += log_det

        # both (B, C, H, W)
        return x, log_det_out
