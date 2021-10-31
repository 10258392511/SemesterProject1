import torch.nn as nn
from .modules import MaskedLinear


class MADEBase(nn.Module):
    def __init__(self):
        super(MADEBase, self).__init__()
        pass

    def _sample_ordering(self):
        pass

    def _create_masks(self):
        pass


class MADE(MADEBase):
    def __init__(self):
        super(MADE, self).__init__()
        pass

    def forward(self):
        pass

    def loss(self):
        pass

    def sample(self):
        pass


class MADEVAE(MADEBase):
    def __init__(self):
        super(MADEVAE, self).__init__()

    def forward(self):
        pass
