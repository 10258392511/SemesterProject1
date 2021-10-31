import numpy as np
import torch.nn as nn
from pprint import pprint
from .modules import MaskedLinear


class MADEBase(nn.Module):
    def __init__(self, in_shape, out_num_comp, hidden_sizes=None, ordering=None):
        super(MADEBase, self).__init__()
        self.in_shape = np.prod(in_shape).astype(np.int)  # (1, H, W) -> HW
        self.out_num_comp = out_num_comp  # e.g. 2 for MNIST
        self.out_shape = self.in_shape * self.out_num_comp  # HW * 2
        self.hidden_sizes = [512, 512] if hidden_sizes is None else hidden_sizes
        self.ordering = np.arange(self.in_shape) if ordering is None else ordering
        self.net = nn.ModuleList()
        features = [self.in_shape] + self.hidden_sizes + [self.out_shape]
        for in_features, out_features in zip(features, features[1:]):
            self.net.extend([MaskedLinear(in_features, out_features), nn.ReLU()])
        self.net = self.net[:-1]
        self._create_masks()

    def _sample_ordering(self):
        num_activations = len(self.hidden_sizes) + 2
        feature_sizes = [self.in_shape] + self.hidden_sizes + [self.out_shape]
        activation_orders = [self.ordering]
        for i in range(1, num_activations - 1):
            order = np.random.randint(0, self.in_shape, (feature_sizes[i],))
            while 0 not in order:
                order = np.random.randint(0, self.in_shape, (feature_sizes[i],))
            activation_orders.append(order)
        activation_orders.append(self.ordering)
        # pprint(f"inside MADEBase: activation_orders: {activation_orders}")
        self.masks = []
        for in_feature_order, out_feature_order in zip(activation_orders, activation_orders[1:-1]):  # except the last
            # e.g. (2, 0, 1) -> (0, 1): [[2, 0, 1],     [[0, 0, 0],
            #                            [2, 0, 1]] <=   [1, 1, 1]]
            self.masks.append(in_feature_order <= out_feature_order[:, np.newaxis])
        out_mask = (activation_orders[-2] < activation_orders[-1][:, np.newaxis])
        # repeat: e.g (0, 1) -> (2, 0, 1): [[1, 1], [0, 0], [1, 0]] -> [[1, 1], [1, 1], [0, 0], [0, 0], [1, 0], [1, 0]]
        # equivalent as kron(mask, [0, 1]^T)
        self.masks.append(np.repeat(out_mask, self.out_num_comp, axis=0))
        # pprint(f"inside MADEBase: masks: {self.masks}")

    def _create_masks(self):
        self._sample_ordering()
        counter = 0
        for net in self.net:
            if isinstance(net, MaskedLinear):
                net.set_mask(self.masks[counter])
                counter += 1
        # # debug only
        # for i, net in enumerate(self.net):
        #     if isinstance(net, MaskedLinear):
        #         print(f"inside MADEBase: {i}: {net.mask}")


class MADE(MADEBase):
    def __init__(self, in_shape, out_num_comp, hidden_sizes=None, ordering=None):
        super(MADE, self).__init__(in_shape, out_num_comp, hidden_sizes=None, ordering=None)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)  # (B, N)
        for net in self.net:
            x = net(x)

        # (B, N * out_num_comp) -> (B, out_num_comp, N), in accordance with nn.CrossEntropyLoss(.)
        return x.reshape(batch_size, self.out_num_comp, -1).contiguous()

    def loss(self, x, label):
        # x: (B, C, H, W), torch.float32, label: (B, C, H, W), torch.int64
        batch_size = label.shape[0]
        label = label.view(batch_size, -1)  # (B, N)
        criterion = nn.CrossEntropyLoss()
        dist_out = self(x)
        return criterion(dist_out, label)

    def sample(self):
        pass


class MADEVAE(MADEBase):
    def __init__(self, in_shape, out_num_comp, hidden_sizes=None, ordering=None):
        super(MADEVAE, self).__init__(in_shape, out_num_comp, hidden_sizes=None, ordering=None)

    def forward(self):
        pass
