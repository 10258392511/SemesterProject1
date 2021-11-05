import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pprint import pprint
from .modules import MaskedLinear


class MADEBase(nn.Module):
    def __init__(self, in_shape, out_num_comp, hidden_sizes=None, ordering=None):
        super(MADEBase, self).__init__()
        self.orginal_shape = in_shape
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
        super(MADE, self).__init__(in_shape, out_num_comp, hidden_sizes=hidden_sizes, ordering=ordering)
        self.inv_order = None

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
        dist_out = self(x)  # (B, out_num_comp, N)
        return criterion(dist_out, label)

    @torch.no_grad()
    def sample(self, num_samples, device=None):
        if self.inv_order is None:
            self.inv_order = {x: i for i, x in enumerate(self.ordering)}
        if device is None:
            device = torch.device("cuda")
        samples_out = torch.zeros((num_samples, int(self.in_shape)), dtype=torch.float32, device=device)  # (B, Nin)
        for generate_order in range(self.in_shape):
            dist_out = self.forward(samples_out)  # (B, out_num_comp, Nin)
            real_index = self.inv_order[generate_order]
            cur_dist_out = F.softmax(dist_out[..., real_index], dim=1)  # (B, out_num_comp)
            # sample instead of argmax
            samples_out[:, real_index] = torch.multinomial(cur_dist_out, 1).squeeze()  # (B, 1) -> (B,)
            # # debug only
            # print(f"generating order: {generate_order}, real_index: {real_index}")
            # print(samples_out)
            # print("-" * 40)

        return samples_out.detach().cpu().reshape(num_samples, *self.orginal_shape)  # (B, C, H, W)


class MADEVAE(MADEBase):
    def __init__(self, in_shape, out_num_comp=2, hidden_sizes=None, ordering=None):
        assert len(in_shape) == 1, "Only for 1D feature"
        super(MADEVAE, self).__init__(in_shape, out_num_comp=out_num_comp, hidden_sizes=hidden_sizes, ordering=ordering)
        # out_num_comp: mu and sigma

    def forward(self, z):
        # X --q(z|x)--> Z --MADE--> eps ~ N(0; I)
        # eps = z * sigma(z) + mu(z)
        batch_size = z.shape[0]
        z = z.view(batch_size, -1)  # (B, N)
        for net in self.net:
            z = net(z)

        # (B, N * out_num_comp) -> (B, out_num_comp, N), in accordance with nn.CrossEntropyLoss(.)
        mu, sigma = torch.chunk(z, 2, dim=1)
        return mu, sigma
