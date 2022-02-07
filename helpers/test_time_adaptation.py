import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader
from .losses import dice_loss_3d, dice_loss, cross_entropy_loss, symmetric_loss
from .utils import random_contrast_transform, augmentation_by_normalizer


def data_aug(X, normalizer_list, device):
    # X: (B, 1, H, W), [0, 1]
    X = random_contrast_transform(X)
    # X = augmentation_by_normalizer(X, normalizer_list, device=device)

    return X.float()


def average_model(normalizer, normalizer_cps):
    state_dict = normalizer.state_dict()
    for key in state_dict:
        avg = 0
        for normalizer_cp in normalizer_cps:
            state_dict_cp = normalizer_cp.state_dict()
            assert key in state_dict_cp
            avg += state_dict_cp[key]
        avg /= len(normalizer_cps)
        state_dict[key] = avg
    normalizer.load_state_dict(state_dict)


def update_normalizer(data_loader, normalizer, u_net, normalizer_cps, norm_opt_cps,
                      normalizer_list, loss_fn, device):
    """
    One iteration through the whole data_loader, denoted by a "step"; all models have been set to correct mode (train
    or eval) outside the scope
    """
    for X in data_loader:
        X = X[0]
        X = X.to(device)
        for normalizer_cp, norm_opt_cp in zip(normalizer_cps, norm_opt_cps):
            normalizer_cp.load_state_dict(normalizer.state_dict())

            X_aug1 = data_aug(X, normalizer_list, device)
            X_aug2 = data_aug(X, normalizer_list, device)
            X_aug1 = X_aug1 * 2 - 1
            X_aug2 = X_aug2 * 2 - 1
            X_pred_1 = u_net(normalizer_cp(X_aug1))
            X_pred_2 = u_net(normalizer_cp(X_aug2))
            loss = symmetric_loss(X_pred_1, X_pred_2, loss_fn)
            norm_opt_cp.zero_grad()
            loss.backward()
            norm_opt_cp.step()

        average_model(normalizer, normalizer_cps)


@torch.no_grad()
def _compute_3D_loss(X, mask, u_net, normalizer):
    # X, mask: (D, 1, H, W), already sent to DEVICE
    normalizer.eval()
    X_pred = u_net(normalizer(2 * X - 1))  # (D, K, H, W)
    loss = dice_loss_3d(X_pred, mask.squeeze(1))
    normalizer.train()

    return loss.item()


@torch.no_grad()
def _compute_tta_loss(X, mask, u_net, normalizer, loss_fn, normalizer_list, device):
    # X, mask: (D, 1, H, W), already sent to DEVICE
    normalizer.eval()
    X_aug1 = data_aug(X, normalizer_list, device)
    X_aug2 = data_aug(X, normalizer_list, device)
    X_aug1 = X_aug1 * 2 - 1
    X_aug2 = X_aug2 * 2 - 1
    X_pred_1 = u_net(normalizer(X_aug1))
    X_pred_2 = u_net(normalizer(X_aug2))
    loss = symmetric_loss(X_pred_1, X_pred_2, loss_fn)
    normalizer.train()

    return loss.item()


@torch.no_grad()
def make_prediction_plot(X, mask, normalizer, normalizer_cp, u_net, device, if_notebook=False, **kwargs):
    normalizer.eval()
    normalizer_cp.eval()
    u_net.eval()
    figsize = kwargs.get("figsize", (10.8, 2.4))
    fraction = kwargs.get("fraction", 0.05)

    X_pred_original = u_net(normalizer_cp(2 * X.to(device) - 1)).argmax(1)  # (D, K, H, W) -> (D, H, W)
    X_pred_adapted = u_net(normalizer(2 * X.to(device) - 1)).argmax(1)

    fig, axes = plt.subplots(X.shape[0], 4, figsize=(figsize[0], figsize[1] * X.shape[0]))
    titles = ["X_orig", "gt", "X_pred_no_adapt", "X_pred_adapt"]
    for i in range(X.shape[0]):
        axes_iter = axes[i, :]
        # X, mask: (D, 1, H, W)
        imgs = [X[i, 0], mask[i, 0], X_pred_original[i], X_pred_adapted[i]]
        for axis, img_iter, title in zip(axes_iter, imgs, titles):
            handle = axis.imshow(img_iter.detach().cpu().numpy(), cmap="gray")
            axis.set_title(title)
            plt.colorbar(handle, ax=axis, fraction=fraction)
    fig.tight_layout()
    if if_notebook:
        plt.show()
    plt.close()

    return fig


def test_time_adaptation_avg(X, mask, normalizer, u_net, normalizer_cps, norm_opt_cps, normalizer_list,
                             batch_size, loss_fn=None, device=None, diff_rel=1e-4, max_iters=50, if_notebook=False):
    # X, mask: (D, 1, H, W); no need make a copy of normalizer
    assert len(normalizer_cps) == len(norm_opt_cps)
    if loss_fn is None:
        loss_fn = lambda X, mask: 0.5 * cross_entropy_loss(X, mask) + 0.5 * dice_loss(X, mask)
    device = torch.device("cuda") if device is None else device
    u_net.eval()
    normalizer.train()
    for normalizer_cp in normalizer_cps:
        normalizer_cp.train()
    local_dataset = TensorDataset(X)
    local_dataloader = DataLoader(local_dataset, batch_size=batch_size, shuffle=True)

    X = X.to(device)
    mask = mask.to(device)
    if if_notebook:
        from tqdm.notebook import trange
    else:
        from tqdm import trange
    pbar = trange(max_iters, desc="tta")
    tta_losses, dice_losses = [_compute_tta_loss(X, mask, u_net, normalizer, loss_fn, normalizer_list, device)], \
                              [_compute_3D_loss(X, mask, u_net, normalizer)]

    for step in pbar:
        update_normalizer(local_dataloader, normalizer, u_net, normalizer_cps, norm_opt_cps, normalizer_list, loss_fn,
                          device)
        tta_losses.append(_compute_tta_loss(X, mask, u_net, normalizer, loss_fn, normalizer_list, device))
        dice_losses.append(_compute_3D_loss(X, mask, u_net, normalizer))

        if abs((tta_losses[-1] - tta_losses[-2]) / tta_losses[-2]) < diff_rel:
            break

        pbar.set_description(f"step {step + 1}/{max_iters}: tta loss: {tta_losses[-1]:.4f}, "
                             f"dice loss: {dice_losses[-1]:.4f}")

    fig, axis = plt.subplots()
    axis.plot(tta_losses, label="tta")
    axis.plot(dice_losses, label="dice loss")
    axis.set_title(f"DICE loss: original: {dice_losses[0]:.4f}, adapted: {dice_losses[-1]:.4f}")
    axis.grid(True)
    axis.legend()
    if if_notebook:
        plt.show()
    plt.close()

    return dice_losses[0], dice_losses[-1], fig


def evaluate_3D_adapt_avg(dataset_dict, normalizer, u_net, normalizer_cp, normalizer_cps, norm_opt_cps, normalizer_list,
                          batch_size, loss_fn=None, device=None, diff_rel=1e-4, max_iters=50, if_notebook=False):
    # dataset_dict: {"csf": ...}
    losses_out = {}
    figs_out = {}  # {"csf": [fig_vol0, fig_vol1, ...]}
    figs_pred_out = {}
    if if_notebook:
        from tqdm.notebook import trange
    else:
        from tqdm import trange

    for key in dataset_dict:
        dataset_iter = dataset_dict[key]
        fig_list = []
        fig_pred_list = []
        loss_avg = 0
        loss_start_avg = 0
        pbar = trange(len(dataset_iter), desc=f"{key}")
        for i in pbar:
            # ###
            # # debug only
            # if i >= 2:
            #     break
            # ###
            X, mask = dataset_iter[i]  # (1, D, H, W), (1, D, H, W)
            X = X.permute(1, 0, 2, 3)  # (D, 1, H, W)
            mask = mask.permute(1, 0, 2, 3)
            normalizer.load_state_dict(normalizer_cp.state_dict())
            loss_start, loss, fig = test_time_adaptation_avg(X, mask, normalizer, u_net, normalizer_cps, norm_opt_cps,
                                                 normalizer_list, batch_size, loss_fn=loss_fn, device=device,
                                                 diff_rel=diff_rel, max_iters=max_iters, if_notebook=if_notebook)
            fig_list.append(fig)
            loss_start_avg += loss_start
            loss_avg += loss

            fig_pred = make_prediction_plot(X, mask, normalizer, normalizer_cp, u_net, device, if_notebook=if_notebook)
            fig_pred_list.append(fig_pred)

        loss_start_avg /= len(dataset_iter)
        loss_avg /= len(dataset_iter)
        losses_out[key] = (loss_start_avg, loss_avg)
        figs_out[key] = fig_list
        figs_pred_out[key] = fig_pred_list

    return losses_out, figs_out, figs_pred_out
