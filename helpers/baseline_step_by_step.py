import torch
import os
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from .losses import dice_loss_3d, dice_loss, cross_entropy_loss, symmetric_loss
from .utils import random_gamma_transform, sample_from_loader, random_contrast_transform


@torch.no_grad()
def evaluate_3d_no_adapt(X, mask, normalizer, u_net, if_normalizer=True, device=None):
    # X, mask: (1, D, H, W), (1, D, H, W), X: [0, 1]
    device = torch.device("cuda") if device is None else device
    normalizer.eval()
    u_net.eval()
    X = (2 * X[0] - 1).unsqueeze(1).to(device)  # (D, H, W) -> (D, 1, H, W)
    mask = mask[0].to(device)  # (D, H, W)
    if if_normalizer:
        X = normalizer(X)
    mask_pred = u_net(X)  # (D, C, H, W)
    loss = dice_loss_3d(mask_pred, mask, num_classes=4)

    return loss.item()


def compute_norm_loss_and_update(data_loader, normalizer, u_net, norm_opt,  loss_fn, device):
    # X: cpu; X: [0, 1]
    u_net.eval()
    normalizer.train()
    loss_acc = 0
    num_samples = 0
    # original implementation
    # for X in data_loader:
    #     X = (2 * X - 1).float().to(device)
    #     X_pred = u_net(X)
    #     X_norm_pred = u_net(normalizer(X))
    #     loss = symmetric_loss(X_pred, X_norm_pred, loss_fn)
    #
    #     norm_opt.zero_grad()
    #     loss.backward()
    #     norm_opt.step()
    #     loss_acc += loss.item() * X.shape[0]
    #     num_samples += X.shape[0]

    for X1, X2 in data_loader:
        X1 = (2 * X1 - 1).float().to(device)
        X2 = (2 * X2 - 1).float().to(device)
        X_norm_pred_1 = u_net(normalizer(X1))
        X_norm_pred_2 = u_net(normalizer(X2))
        loss = symmetric_loss(X_norm_pred_1, X_norm_pred_2, loss_fn)

        norm_opt.zero_grad()
        loss.backward()
        norm_opt.step()
        loss_acc += loss.item() * X1.shape[0]
        num_samples += X1.shape[0]

    return loss_acc / num_samples


def test_time_adaptation(X, mask, normalizer, u_net, norm_opt, batch_size,
                         loss_fn=None, device=None, diff_rel=1e-4, max_iters=10):
    # X, mask: (B, 1, H, W) each, cpu, [0, 1]; normalizer: bottleneck; "mask" can be None
    if loss_fn is None:
        loss_fn = lambda X, mask: 0.5 * dice_loss(X, mask) + 0.5 * cross_entropy_loss(X, mask)

    device = torch.device("cuda") if device is None else device
    # X = (2 * X - 1).to(device)

    if mask is not None:
        mask = mask.to(device)

    u_net.eval()
    normalizer.train()

    # original implementation
    # local_loader = DataLoader(X, batch_size=batch_size)

    X_aug_1 = random_contrast_transform(X)
    X_aug_2 = random_contrast_transform(X)
    local_dataset = TensorDataset(X_aug_1, X_aug_2)
    local_loader = DataLoader(local_dataset, batch_size=batch_size)

    X = (2 * X - 1).float().to(device)

    if mask is not None:
        fig_orig = make_summary_plot_simplified(X / 2 + 0.5, mask, normalizer, u_net, device=device)

    losses = []
    if mask is not None:
        dice_losses = []

    cur_loss, next_loss = None, None
    # X_pred = u_net(X)  # (B, 1, H, W)
    # X_norm = normalizer(X)  # (B, 1, H, W)
    # X_norm_pred = u_net(X_norm)
    # next_loss = symmetric_loss(X_pred, X_norm_pred, loss_fn)
    next_loss = compute_norm_loss_and_update(local_loader, normalizer, u_net, norm_opt, loss_fn, device)

    num_steps = 0
    # while cur_loss is None or abs((next_loss.item() - cur_loss) / cur_loss) > diff_rel:
    while cur_loss is None or abs((next_loss - cur_loss) / cur_loss) > diff_rel:
        cur_loss = next_loss
        next_loss = compute_norm_loss_and_update(local_loader, normalizer, u_net, norm_opt, loss_fn, device)
        losses.append(cur_loss)

        # cur_loss = next_loss.item()
        # losses.append(cur_loss)
        # norm_opt.zero_grad()
        # next_loss.backward()
        # norm_opt.step()
        # # torch.cuda.empty_cache()

        # X_pred = u_net(X)  # (1, 1, H, W)
        # X_norm = normalizer(X)  # (1, 1, H, W)
        # X_norm_pred = u_net(X_norm)
        # next_loss = symmetric_loss(X_pred, X_norm_pred, loss_fn)

        if mask is not None:
            with torch.no_grad():
                dice_losses.append(dice_loss(u_net(normalizer(X)), mask).item())

        num_steps += 1
        # print(f"current: {num_steps}/{max_iters}")
        if num_steps > max_iters:
            break

    normalizer.eval()
    if mask is not None:
        fig_adapted = make_summary_plot_simplified(X / 2 + 0.5, mask, normalizer, u_net, device=device)

    fig, axis = plt.subplots()
    axis.plot(losses, label="tta")
    if mask is not None:
        axis.plot(dice_losses, label="dice loss")
    axis.legend()
    axis.grid(True)

    with torch.no_grad():
        X_pred = u_net(normalizer(X))  # (1, K, H, W)
    plt.close()

    if mask is not None:
        return X_pred.detach().cpu(), fig_orig, fig_adapted, fig

    return X_pred.detach().cpu(), fig


def evaluate_3d_adapt(X, mask, normalizer, u_net, norm_opt_config: dict, device=None, normalizer_cp=None,
                      out_channels=4, max_iters=10, batch_size=1):
    # X, mask: (1, D, H, W), (1, D, H, W), X: [0, 1]; normalizer should be a copy
    u_net.eval()
    normalizer_cp.eval()

    local_dataset = TensorDataset(X[0], mask[0])  # (D, H, W)
    local_dataloader = DataLoader(local_dataset, batch_size=batch_size)

    loss_orig = evaluate_3d_no_adapt(X, mask, normalizer_cp, u_net, device=device)
    print(f"loss_orig: {loss_orig}")
    D, H, W = X.shape[1:]
    X_preds = torch.empty(D, out_channels, H, W)

    for d, (X_cur, mask_cur) in enumerate(local_dataloader):
        print(f"current d: {d + 1}/{D}")
        # print(f"{X_cur.shape}, {mask_cur.shape}")
        normalizer.load_state_dict(normalizer_cp.state_dict())
        normalizer.train()
        norm_opt = torch.optim.Adam(normalizer.parameters(), **norm_opt_config)
        # X_cur, mask_cur = X[:, d:d + 1, ...], mask[:, d:d + 1, ...]
        X_cur, mask_cur = X_cur.unsqueeze(0), mask_cur.unsqueeze(0)  # (1, 1, H, W)
        X_pred, _, _, _ = test_time_adaptation(X_cur, mask_cur, normalizer, u_net, norm_opt, batch_size,
                                               device=device, max_iters=max_iters)  # (1, K, H, W)
        X_preds[d, ...] = X_pred

    normalizer.eval()
    loss = dice_loss_3d(X_preds.to(device), mask[0].to(device)).item()
    del X_preds

    return loss


def evaluate_3d_adapt_batch(X, mask, normalizer, u_net, norm_opt_config: dict, device=None,
                            max_iters=10, batch_size=4):
    # X, mask: (1, D, H, W), (1, D, H, W), X: [0, 1]; normalizer should be a copy
    u_net.eval()
    normalizer.train()
    X = X[0].unsqueeze(1)  # (D, 1, H, W)
    mask = mask[0].unsqueeze(1)
    norm_opt = torch.optim.Adam(normalizer.parameters(), **norm_opt_config)
    # (B, K, H, W)
    X_pred, _, = test_time_adaptation(X, None, normalizer, u_net, norm_opt, batch_size,
                                           device=device, max_iters=max_iters)

    normalizer.eval()
    loss = dice_loss_3d(X_pred, mask[:, 0, ...]).item()
    del X_pred

    return loss


def evaluate_3d_wrapper(evaluate_fn, dataset, normalizer, u_net, device=None, if_notebook=False, **kwargs):
    """
    Since (D, H, W): D is different for each patient, so a DataLoader is impossible. We use Dataset directly here with
    iteration.
    """
    if if_notebook:
        from tqdm.notebook import trange
    else:
        from tqdm import trange
    pbar = trange(len(dataset), desc="Evaluation")

    loss_avg = 0
    for i in pbar:
        X, mask = dataset[i]  # (1, D, H, W), (1, D, H, W)
        # loss = evaluate_fn(X, mask, normalizer, u_net, device=device)
        loss = evaluate_fn(X, mask, normalizer, u_net, device=device, **kwargs)
        loss_avg += loss
        pbar.set_description(desc=f"batch {i + 1}/{len(dataset)}: loss: {loss:.4f}")

    return loss_avg / len(dataset)


@torch.no_grad()
def make_summary_plot_simplified(X, mask, normalizer, u_net, if_show=False,
                                 if_save=False, save_path=None, device=None, **kwargs):
    """
    X, mask: (1, 1, H, W); X: [0, 1]
    device: already torch.device(...)
    """
    if if_save:
        assert save_path is not None, "please specify a filename to save the image"
    normalizer.eval()
    u_net.eval()
    device = torch.device("cuda") if device is None else device
    X_orig = X.clone().to(device)
    X = (2 * X - 1).to(device)
    mask = mask.to(device)
    X_norm = normalizer(X)  # (1, 1, H, W)
    X_norm_plot = (X_norm + 1) / 2  # (1, 1, H, W)
    X_norm_diff = X_norm_plot - X_orig  # (1, 1, H, W)
    mask_direct_pred_full = u_net(X)  # (1, C, H, W)
    mask_direct_pred = mask_direct_pred_full.argmax(dim=1)  # (1, C, H, W) -> (1, H, W)
    mask_norm_pred_full = u_net(X_norm)  # (1, C, H, W)
    mask_norm_pred = mask_norm_pred_full.argmax(dim=1)  # (1, C, H, W) -> (1, H, W)

    figsize = kwargs.get("figsize", (10.8, 7.2))
    fraction = kwargs.get("fraction", 0.5)
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    img_grid = [[X_orig[0, 0], mask[0, 0], mask_direct_pred[0]],
                [X_norm_plot[0, 0], X_norm_diff[0, 0], mask_norm_pred[0]]]
    title_grid = [["orig", "gt", "direct pred"],
                  ["normed", "normed diff", "normed pred"]]

    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            axis = axes[i, j]
            img_iter = img_grid[i][j].detach().cpu().numpy()
            title_iter = title_grid[i][j]
            handle = axis.imshow(img_iter, cmap="gray")
            plt.colorbar(handle, ax=axis, fraction=fraction)
            axis.set_title(title_iter)

    loss_direct = dice_loss(mask_direct_pred_full, mask)
    loss_normed = dice_loss(mask_norm_pred_full, mask)

    suptitle = f"loss_direct: {loss_direct:.4f}, loss_normed: {loss_normed:.4f}"
    fig.suptitle(suptitle)
    fig.tight_layout()

    if if_save:
        plt.savefig(save_path)

    if if_show:
        plt.show()

    plt.close()

    return fig

@torch.no_grad()
def make_summary_plot_2_by_2(X, mask, normalizer, u_net, if_show=False,
                             if_save=False, save_path=None, device=None, **kwargs):
    if if_save:
        assert save_path is not None, "please specify a filename to save the image"
    normalizer.eval()
    u_net.eval()
    device = torch.device("cuda") if device is None else device
    X_orig = X.clone().to(device)
    B, C_in, H, W = X.shape
    X = (2 * X - 1).to(device)
    mask = mask.to(device)
    X_norm = normalizer(X)  # (1, 16, H, W)
    # X_norm_plot = (X_norm + 1) / 2  # (1, 1, H, W)
    # X_norm_diff = X_norm_plot - X_orig  # (1, 1, H, W)
    mask_direct_pred_full = u_net(X.expand(B, X_norm.shape[1], H, W))  # (1, 16, H, W) -> (1, C, H, W)
    mask_direct_pred = mask_direct_pred_full.argmax(dim=1)  # (1, C, H, W) -> (1, H, W)
    mask_norm_pred_full = u_net(X_norm)  # (1, C, H, W)
    mask_norm_pred = mask_norm_pred_full.argmax(dim=1)  # (1, C, H, W) -> (1, H, W)

    figsize = kwargs.get("figsize", (10.8, 7.2))
    fraction = kwargs.get("fraction", 0.5)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    img_grid = [[X_orig[0, 0], mask[0, 0]],
                [mask_direct_pred[0], mask_norm_pred[0]]]
    title_grid = [["orig", "gt"],
                  ["direct pred", "normed pred"]]

    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            axis = axes[i, j]
            img_iter = img_grid[i][j].detach().cpu().numpy()
            title_iter = title_grid[i][j]
            handle = axis.imshow(img_iter, cmap="gray")
            plt.colorbar(handle, ax=axis, fraction=fraction)
            axis.set_title(title_iter)

    loss_direct = dice_loss(mask_direct_pred_full, mask)
    loss_normed = dice_loss(mask_norm_pred_full, mask)

    suptitle = f"loss_direct: {loss_direct:.4f}, loss_normed: {loss_normed:.4f}"
    fig.suptitle(suptitle)
    fig.tight_layout()

    if if_save:
        plt.savefig(save_path)

    if if_show:
        plt.show()

    plt.close()

    return fig


@torch.no_grad()
def make_summary_plot_1_by_3(X, mask, normalizer, u_net, if_show=False,
                             if_save=False, save_path=None, device=None, **kwargs):
    if if_save:
        assert save_path is not None, "please specify a filename to save the image"
    normalizer.eval()
    u_net.eval()
    device = torch.device("cuda") if device is None else device
    X_orig = X.clone().to(device)
    B, C_in, H, W = X.shape
    X = (2 * X - 1).to(device)
    mask = mask.to(device)
    # X_norm = normalizer(X)  # (1, 16, H, W)
    # X_norm_plot = (X_norm + 1) / 2  # (1, 1, H, W)
    # X_norm_diff = X_norm_plot - X_orig  # (1, 1, H, W)
    mask_direct_pred_full = u_net(X)  # (1, 16, H, W) -> (1, C, H, W)
    mask_direct_pred = mask_direct_pred_full.argmax(dim=1)  # (1, C, H, W) -> (1, H, W)
    # mask_norm_pred_full = u_net(X_norm)  # (1, C, H, W)
    # mask_norm_pred = mask_norm_pred_full.argmax(dim=1)  # (1, C, H, W) -> (1, H, W)

    figsize = kwargs.get("figsize", (10.8, 7.2))
    fraction = kwargs.get("fraction", 0.5)
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    axes = axes[None, :]
    img_grid = [[X_orig[0, 0], mask[0, 0], mask_direct_pred[0]]]
    title_grid = [["orig", "gt", "direct_pred"]]

    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            axis = axes[i, j]
            img_iter = img_grid[i][j].detach().cpu().numpy()
            title_iter = title_grid[i][j]
            handle = axis.imshow(img_iter, cmap="gray")
            plt.colorbar(handle, ax=axis, fraction=fraction)
            axis.set_title(title_iter)

    loss_direct = dice_loss(mask_direct_pred_full, mask)
    # loss_normed = dice_loss(mask_norm_pred_full, mask)

    suptitle = f"loss_direct: {loss_direct:.4f}"
    fig.suptitle(suptitle)
    fig.tight_layout()

    if if_save:
        plt.savefig(save_path)

    if if_show:
        plt.show()

    plt.close()

    return fig


class BasicTrainer(object):
    def __init__(self, train_loader, eval_loader, test_loader, normalizer, u_net, norm_opt, u_net_opt,
                 epochs, num_classes, weights: dict, notebook: bool, writer: SummaryWriter,
                 param_save_dir, time_stamp, device=None, scheduler=None):
        for key in ["lam_ce", "lam_dsc", "lam_smooth"]:
            assert key in weights, f"{key} is missing"

        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.test_loader = test_loader
        self.normalizer = normalizer
        self.u_net = u_net
        self.norm_opt = norm_opt
        self.u_net_opt = u_net_opt
        self.scheduler = scheduler
        self.epochs = epochs
        self.num_classes = num_classes
        self.weights = weights  # {lam_ce: ..., lam_dsc: ..., lam_smooth: ...}
        self.notebook = notebook
        self.global_steps = {"train": 0, "eval": 0, "epoch": 0}
        self.writer = writer
        self.param_save_dir = param_save_dir  # param/norm_u_net
        self.time_stamp = time_stamp  # extended time_stamp
        self.device = torch.device("cuda") if device is None else torch.device(device)

    def _train(self, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def _eval(self, **kwargs):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    @torch.no_grad()
    def _end_epoch_plot(self):
        ind = np.random.randint(len(self.test_loader.dataset))
        X, mask = self.test_loader.dataset[ind]  # (1, H, W), (1, H, W)
        # fig = make_summary_plot_simplified(X.unsqueeze(0), mask.unsqueeze(0), self.normalizer, self.u_net,
        #                                    if_show=self.notebook, device=self.device)
        fig = make_summary_plot_2_by_2(X.unsqueeze(0), mask.unsqueeze(0), self.normalizer, self.u_net,
                                       if_show=self.notebook, device=self.device)
        # fig = make_summary_plot_1_by_3(X.unsqueeze(0), mask.unsqueeze(0), self.normalizer, self.u_net,
        #                                if_show=self.notebook, device=self.device)
        return fig

    def _create_save_dir(self):
        if not os.path.isdir(self.param_save_dir):
            os.mkdir(self.param_save_dir)
        os.mkdir(os.path.join(self.param_save_dir, self.time_stamp))

    def _save_params(self, epoch, eval_loss):
        dir_path = os.path.join(self.param_save_dir, self.time_stamp)
        for filename in os.listdir(dir_path):
            filename_abs = os.path.join(dir_path, filename)
            if os.path.isfile(filename_abs):
                os.remove(filename_abs)
        filename_common = f"epoch_{epoch}_eval_loss_{eval_loss:.4f}".replace(".", "_") + ".pt"
        torch.save(self.normalizer.state_dict(), f"{dir_path}/norm_{filename_common}")
        torch.save(self.u_net.state_dict(), f"{dir_path}/u_net_{filename_common}")


class OnePassTrainer(BasicTrainer):
    def __init__(self, test_dataset_dict, **kwargs):
        super(OnePassTrainer, self).__init__(**kwargs)
        self.test_dataset_dict = test_dataset_dict  # {"csf": ..., "hvhd": ..., "uhe": ...}

    def _train(self, **kwargs):
        self.normalizer.train()
        self.u_net.train()

        if self.notebook:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="training", leave=False)
        loss_avg = 0
        num_samples = 0

        for i, (X, mask) in pbar:
            # # debug only #
            # if i > 2:
            #     break
            # ################
            X = X.to(self.device)
            mask = mask.to(self.device)
            loss_sup, loss_unsup = self._compute_normalizer_loss(X, mask)  # only one pass is required
            # loss_sup, loss_unsup = self._compute_u_net_loss(X, mask)
            loss_all = loss_sup + loss_unsup
            self.u_net_opt.zero_grad()
            self.norm_opt.zero_grad()
            loss_all.backward()
            loss_avg += loss_all.item() * X.shape[0]
            num_samples += X.shape[0]
            self.u_net_opt.step()
            self.norm_opt.step()

            pbar.set_description(f"batch {i + 1}/{len(self.train_loader)}: loss_all: {loss_all.item():.4f}, "
                                 f"loss_sup: {loss_sup.item():.4f}, loss_unsup: {loss_unsup.item():.4f}")
            self.writer.add_scalar("train_loss_all", loss_all.item(), self.global_steps["train"])
            self.writer.add_scalar("train_loss_sup", loss_sup.item(), self.global_steps["train"])
            self.writer.add_scalar("train_loss_unsup", loss_unsup.item(), self.global_steps["train"])

            self.global_steps["train"] += 1

        loss_avg /= num_samples
        self.writer.add_scalar("train_loss_epoch", loss_avg, self.global_steps["eval"])
        pbar.close()

    @torch.no_grad()
    def _eval(self, **kwargs):
        self.normalizer.eval()
        self.u_net.eval()

        if self.notebook:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        pbar = tqdm(enumerate(self.eval_loader), total=len(self.eval_loader), desc="eval", leave=False)

        loss_sup_avg, loss_unsup_avg = 0, 0
        num_samples = 0
        for i, (X, mask) in pbar:
            # # debug only #
            # if i > 2:
            #     break
            # ###############
            X = X.to(self.device)
            mask = mask.to(self.device)
            loss_sup, loss_unsup = self._compute_normalizer_loss(X, mask)  # only one pass is required
            # loss_sup, loss_unsup = self._compute_u_net_loss(X, mask)
            loss_all = loss_sup + loss_unsup
            loss_sup_avg += loss_sup.item() * X.shape[0]
            loss_unsup_avg += loss_unsup.item() * X.shape[0]
            num_samples += X.shape[0]

            pbar.set_description(f"batch {i + 1}/{len(self.eval_loader)}: loss_all: {loss_all.item():.4f}, "
                                 f"loss_sup: {loss_sup.item():.4f}, loss_unsup: {loss_unsup.item():.4f}")

        loss_sup_avg /= num_samples
        loss_unsup_avg /= num_samples
        loss_all_avg = loss_sup_avg + loss_unsup_avg

        self.writer.add_scalar("eval_loss_all", loss_all_avg, self.global_steps["eval"])
        self.writer.add_scalar("eval_loss_sup", loss_sup_avg, self.global_steps["eval"])
        self.writer.add_scalar("eval_loss_unsup", loss_unsup_avg, self.global_steps["eval"])

        self.global_steps["eval"] += 1
        pbar.close()

        return loss_all_avg

    def train(self):
        if self.notebook:
            from tqdm.notebook import trange
        else:
            from tqdm import trange
        pbar = trange(self.epochs, desc="epoch")

        self._create_save_dir()
        lowest_loss = float("inf")

        for epoch in pbar:
            self._train()
            loss_all_avg = self._eval()
            if self.scheduler is not None:
                self.scheduler.step()

            fig = self._end_epoch_plot()
            losses_eval_3d = {}
            self.writer.add_figure("epoch_plot", fig, self.global_steps["epoch"])
            for key in self.test_dataset_dict:
                dataset = self.test_dataset_dict[key]
                loss = evaluate_3d_wrapper(evaluate_3d_no_adapt, dataset, self.normalizer, self.u_net,
                                           self.device, self.notebook)
                losses_eval_3d[key] = loss
                self.writer.add_scalar(f"epoch_3d_{key}", loss, self.global_steps["epoch"])

            if loss_all_avg < lowest_loss:
                self._save_params(epoch, loss_all_avg)
                lowest_loss = loss_all_avg
            cur_loss = sum(list(losses_eval_3d.values())) / 3
            self.writer.add_scalar(f"epoch_3d_avg", cur_loss, self.global_steps["epoch"])

            eval_3d_loss_desc = ""
            for key in losses_eval_3d:
                eval_3d_loss_desc += f", loss_3d_{key}: {losses_eval_3d[key]:.4f}"
            pbar.set_description(f"epoch {epoch + 1}/{self.epochs}, loss_all: {loss_all_avg:4f}" + eval_3d_loss_desc)
            self.global_steps["epoch"] += 1

    def _compute_normalizer_loss(self, X, mask):
        # X, mask: (B, 1, H, W), (B, 1, H, W); already sent to self.device; X: [0, 1]
        # X_aug = random_gamma_transform(X)
        # X = 2 * X - 1
        # X_aug = 2 * X_aug - 1
        # mask_pred_direct = self.u_net(X_aug)  # (B, C, H, W)
        # X_norm = self.normalizer(X)  # (B, 1, H, W)
        # mask_pred_norm = self.u_net(X_norm)  # (B, C, H, W)
        # loss_fn = lambda X, mask: self.weights["lam_ce"] * cross_entropy_loss(X, mask) + \
        #                           self.weights["lam_dsc"] * dice_loss(X, mask)
        # # TODO: change back
        # loss_sup = loss_fn(mask_pred_norm, mask)
        # # loss_sup = loss_fn(mask_pred_direct, mask)
        # loss_unsup = self.weights["lam_smooth"] * symmetric_loss(mask_pred_direct, mask_pred_norm, loss_fn)

        # full normalizer
        # X, mask: (B, 1, H, W), (B, 1, H, W); already sent to self.device; X: [0, 1]
        B, C_in, H, W = X.shape
        X_aug = random_gamma_transform(X)
        X = 2 * X - 1
        X_aug = 2 * X_aug - 1
        X_norm = self.normalizer(X)  # (B, 1, H, W)
        mask_pred_direct = self.u_net(X_aug.expand(B, X_norm.shape[1], H, W))  # (B, C, H, W)
        mask_pred_norm = self.u_net(X_norm)  # (B, C, H, W)
        loss_fn = lambda X, mask: self.weights["lam_ce"] * cross_entropy_loss(X, mask) + \
                                  self.weights["lam_dsc"] * dice_loss(X, mask)
        # TODO: change back
        loss_sup = loss_fn(mask_pred_norm, mask)
        # loss_sup = loss_fn(mask_pred_direct, mask)
        loss_unsup = self.weights["lam_smooth"] * symmetric_loss(mask_pred_direct, mask_pred_norm, loss_fn)

        return loss_sup, loss_unsup

    def _compute_u_net_loss(self, X, mask):
        # # return self._compute_normalizer_loss(X, mask)

        # standard U-Net training
        # X = 2 * X - 1
        # mask_pred = self.u_net(X)  # (B, 1, H, W) -> (B, K, H, W)
        # loss_fn = lambda X, mask: self.weights["lam_ce"] * cross_entropy_loss(X, mask) + \
        #                           self.weights["lam_dsc"] * dice_loss(X, mask)
        # loss_sup = loss_fn(mask_pred, mask)
        #
        # return loss_sup, loss_sup * self.weights["lam_smooth"]

        # U-Net with data consistency
        X_aug = random_gamma_transform(X)
        X_aug = 2 * X_aug - 1
        X = 2 * X - 1
        mask_pred_direct = self.u_net(X)  # (B, 1, H, W) -> (B, K, H, W)
        mask_pred_aug = self.u_net(X_aug)
        loss_fn = lambda X, mask: self.weights["lam_ce"] * cross_entropy_loss(X, mask) + \
                                  self.weights["lam_dsc"] * dice_loss(X, mask)
        loss_sup = loss_fn(mask_pred_direct, mask)
        loss_unsup = self.weights["lam_smooth"] * symmetric_loss(mask_pred_direct, mask_pred_aug, loss_fn)

        return loss_sup, loss_unsup


class AltTrainer(BasicTrainer):
    def __init__(self, test_dataset_dict, **kwargs):
        super(AltTrainer, self).__init__(**kwargs)
        self.test_dataset_dict = test_dataset_dict  # {"csf": ..., "hvhd": ..., "uhe": ...}

    def _train(self, **kwargs):
        if self.notebook:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="training", leave=False)

        loss_norm_all_avg, loss_u_net_all_avg = 0, 0
        num_samples = 0
        for i, (X, mask) in pbar:
            # # debug only #
            # if i > 2:
            #     break
            # ################
            num_samples += X.shape[0]
            X = X.to(self.device)
            mask = mask.to(self.device)

            # update self.u_net
            self.normalizer.eval()
            self.u_net.train()
            loss_sup, loss_unsup = self._compute_u_net_loss(X, mask)
            loss_all = loss_sup + loss_unsup
            loss_u_net_all_avg += loss_all.item() * X.shape[0]
            self.u_net_opt.zero_grad()
            loss_all.backward()
            self.u_net_opt.step()
            loss_sup_u_net, loss_unsup_u_net, loss_all_u_net = loss_sup.item(), loss_unsup.item(), loss_all.item()

            # update self.normalizer
            self.u_net.eval()
            self.normalizer.train()
            loss_sup, loss_unsup = self._compute_normalizer_loss(X, mask)
            loss_all = loss_sup + loss_unsup
            loss_norm_all_avg += loss_all.item() * X.shape[0]
            self.norm_opt.zero_grad()
            loss_all.backward()
            self.norm_opt.step()

            pbar.set_description(f"batch {i + 1}/{len(self.train_loader)}: loss_all_norm: {loss_all.item():.4f}, "
                                 f"loss_sup: {loss_sup_u_net:.4f}, loss_unsup: {loss_unsup_u_net:.4f}, "
                                 f"loss_all: {loss_all_u_net:.4f}")

            tags = ["train_loss_all", "train_loss_sup", "train_loss_unsup",
                    "train_loss_all_norm", "train_loss_sup_norm", "train_loss_unsup_norm"]
            scalars = [loss_all_u_net, loss_sup_u_net, loss_unsup_u_net,
                       loss_all.item(), loss_sup.item(), loss_unsup.item()]
            for tag, scalar in zip(tags, scalars):
                self.writer.add_scalar(tag, scalar, self.global_steps["train"])

            self.global_steps["train"] += 1

        loss_norm_all_avg /= num_samples
        loss_u_net_all_avg /= num_samples
        self.writer.add_scalar("train_epoch_norm", loss_norm_all_avg, self.global_steps["eval"])
        self.writer.add_scalar("train_epoch_u_net", loss_u_net_all_avg, self.global_steps["eval"])
        pbar.close()

    @torch.no_grad()
    def _eval(self, **kwargs):
        self.normalizer.eval()
        self.u_net.eval()

        if self.notebook:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        pbar = tqdm(enumerate(self.eval_loader), total=len(self.eval_loader), desc="eval", leave=False)

        loss_sup_avg, loss_unsup_avg = 0, 0
        num_samples = 0
        for i, (X, mask) in pbar:
            # # debug only #
            # if i > 2:
            #     break
            # ################
            X = X.to(self.device)
            mask = mask.to(self.device)
            loss_sup, loss_unsup = self._compute_normalizer_loss(X, mask)  # only one pass is required
            loss_all = loss_sup + loss_unsup
            loss_sup_avg += loss_sup * X.shape[0]
            loss_unsup_avg += loss_unsup * X.shape[0]
            num_samples += X.shape[0]

            pbar.set_description(f"batch {i + 1}/{len(self.eval_loader)}: loss_all: {loss_all.item():.4f}, "
                                 f"loss_sup: {loss_sup.item():.4f}, loss_unsup: {loss_unsup.item():.4f}")

        loss_sup_avg /= num_samples
        loss_unsup_avg /= num_samples
        loss_all_avg = loss_sup_avg + loss_unsup_avg

        self.writer.add_scalar("eval_loss_all", loss_all_avg, self.global_steps["eval"])
        self.writer.add_scalar("eval_loss_sup", loss_sup_avg, self.global_steps["eval"])
        self.writer.add_scalar("eval_loss_unsup", loss_unsup_avg, self.global_steps["eval"])

        self.global_steps["eval"] += 1
        pbar.close()

        return loss_all_avg

    def train(self):
        if self.notebook:
            from tqdm.notebook import trange
        else:
            from tqdm import trange
        pbar = trange(self.epochs, desc="epoch")

        self._create_save_dir()
        lowest_loss = float("inf")

        for epoch in pbar:
            self._train()
            loss_all_avg = self._eval()
            if self.scheduler is not None:
                for scheduler_iter in self.scheduler:
                    scheduler_iter.step()
                    scheduler_iter.print_lr()

            fig = self._end_epoch_plot()
            losses_eval_3d = {}
            self.writer.add_figure("epoch_plot", fig, self.global_steps["epoch"])
            for key in self.test_dataset_dict:
                dataset = self.test_dataset_dict[key]
                loss = evaluate_3d_wrapper(evaluate_3d_no_adapt, dataset, self.normalizer, self.u_net,
                                           self.device, self.notebook)
                losses_eval_3d[key] = loss
                self.writer.add_scalar(f"epoch_3d_{key}", loss, self.global_steps["epoch"])

            if loss_all_avg < lowest_loss:
                self._save_params(epoch, loss_all_avg)
                lowest_loss = loss_all_avg
            cur_loss = sum(list(losses_eval_3d.values())) / 3
            self.writer.add_scalar(f"epoch_3d_avg", cur_loss, self.global_steps["epoch"])
            # if cur_loss < lowest_loss:
            #     self._save_params(epoch, cur_loss)
            #     lowest_loss = cur_loss

            eval_3d_loss_desc = ""
            for key in losses_eval_3d:
                eval_3d_loss_desc += f", loss_3d_{key}: {losses_eval_3d[key]:.4f}"
            pbar.set_description(f"epoch {epoch + 1}/{self.epochs}, loss_all: {loss_all_avg:4f}" + eval_3d_loss_desc)
            self.global_steps["epoch"] += 1

    def _compute_normalizer_loss(self, X, mask):
        # # X, mask: (B, 1, H, W), (B, 1, H, W); already sent to self.device; X: [0, 1]
        # X = 2 * X - 1
        # mask_pred_direct = self.u_net(X)  # (B, C, H, W)
        # X_norm = self.normalizer(X)  # (B, 1, H, W)
        # mask_pred_norm = self.u_net(X_norm)  # (B, C, H, W)
        # loss_fn = lambda X, mask: self.weights["lam_ce"] * cross_entropy_loss(X, mask) + \
        #                           self.weights["lam_dsc"] * dice_loss(X, mask)
        # # TODO: change back
        # loss_sup = loss_fn(mask_pred_norm, mask)
        # # loss_sup = loss_fn(mask_pred_direct, mask)
        # loss_unsup = self.weights["lam_smooth"] * symmetric_loss(mask_pred_direct, mask_pred_norm, loss_fn)

        # full normalizer
        # X, mask: (B, 1, H, W), (B, 1, H, W); already sent to self.device; X: [0, 1]
        # self.u_net.eval(), self.normalizer.train(): set outside the scope
        B, C_in, H, W = X.shape
        X = 2 * X - 1
        X_norm = self.normalizer(X)  # (B, 1, H, W)
        mask_pred_direct = self.u_net(X.expand(B, X_norm.shape[1], H, W))  # (B, C, H, W)
        mask_pred_norm = self.u_net(X_norm)  # (B, C, H, W)
        loss_fn = lambda X, mask: self.weights["lam_ce"] * cross_entropy_loss(X, mask) + \
                                  self.weights["lam_dsc"] * dice_loss(X, mask)
        loss_sup = loss_fn(mask_pred_norm, mask)
        loss_unsup = self.weights["lam_smooth"] * symmetric_loss(mask_pred_direct, mask_pred_norm, loss_fn)

        return loss_sup, loss_unsup

    def _compute_u_net_loss(self, X, mask):
        # self.u_net.train(), self.normalizer.eval(): set outside the scope
        return self._compute_normalizer_loss(X, mask)


class MetaLearner(BasicTrainer):
    """
    train_loader: with RandomSampler
    """
    def __init__(self, test_dataset_dict, normalizer_cp, norm_opt_config,
                 num_batches_to_sample, num_learner_steps, total_steps=10000, eval_interval=100, pre_train_epochs=20,
                 **kwargs):
        super(MetaLearner, self).__init__(**kwargs)
        self.test_dataset_dict = test_dataset_dict  # {"csf": ..., "hvhd": ..., "uhe": ...}
        self.normalizer_cp = normalizer_cp
        self.num_batches_to_sample = num_batches_to_sample
        self.num_learner_steps = num_learner_steps
        self.norm_opt_config = norm_opt_config
        self.total_steps = total_steps
        self.epochs = total_steps // num_batches_to_sample

        self.global_steps["eval_3d"] = 0
        self.global_steps["pre_train"] = 0
        self.global_steps["pre_train_eval"] = 0
        self.global_steps["pre_train_epochs"] = 0

        self.eval_interval = eval_interval
        self.eval_interval_epoch = int(self.epochs * eval_interval / total_steps)
        self.pretrain_epochs = pre_train_epochs

    # def _train(self, **kwargs):
    #     # print("training...")
    #     batches = []
    #     for i in range(self.num_batches_to_sample):
    #         batches.append(sample_from_loader(self.train_loader))
    #
    #     self._meta_train(batches)
    #     loss_avg_all_train = self._meta_learn(batches)
    #
    #     del batches
    #     return loss_avg_all_train
    #
    # def _meta_train(self, batches):
    #     self.u_net.eval()
    #     self.normalizer.train()
    #     fig, axis = plt.subplots()
    #     learner_losses = []  # (B * num_learner_steps,)
    #     xticks = []
    #
    #     for b, (X, _) in enumerate(batches):
    #         # X: (B, 1, H, W)
    #         X = X.to(self.device)
    #         for i in range(self.num_learner_steps):
    #             loss_unsup = self._compute_normalizer_loss(X)
    #             self.norm_opt.zero_grad()
    #             loss_unsup.backward()
    #             self.norm_opt.step()
    #             learner_losses.append(loss_unsup.item())
    #             # xticks.append(float(f"{b}.{i}"))
    #             xticks.append(b + i / 1000)
    #
    #     axis.plot(xticks, learner_losses)
    #     axis.grid(True)
    #     plt.close()
    #     del learner_losses
    #     del xticks
    #     # loss curve is summarized per epoch
    #     self.writer.add_figure("learner_traj_epoch", fig, self.global_steps["epoch"])
    #
    # def _meta_learn(self, batches):
    #     self.u_net.train()
    #     self.normalizer.train()
    #
    #     loss_avg_all_train = 0
    #     num_samples = 0
    #     for X, mask in batches:
    #         # X: (B, 1, H, W)
    #         X = X.to(self.device)
    #         mask = mask.to(self.device)
    #         loss_sup = self._compute_u_net_loss(X, mask)
    #         self.norm_opt.zero_grad()
    #         self.u_net_opt.zero_grad()
    #         loss_sup.backward()
    #         self.norm_opt.step()
    #         self.u_net_opt.step()
    #
    #         loss_avg_all_train += loss_sup.item() * X.shape[0]
    #         num_samples += X.shape[0]
    #         self.writer.add_scalar("loss_meta_learn", loss_sup.item(), self.global_steps["train"])
    #         self.global_steps["train"] += 1
    #
    #     loss_avg_all_train /= num_samples
    #     self.writer.add_scalar("train_epoch", loss_avg_all_train, self.global_steps["epoch"])
    #
    #     return loss_avg_all_train

    def _train(self, **kwargs):
        # print("training...")
        batches_unsup, batches = [], []
        for i in range(self.num_batches_to_sample):
            X, mask = sample_from_loader(self.train_loader)
            batches_unsup.append((random_contrast_transform(X), random_contrast_transform(X)))
            batches.append((X, mask))

        self._meta_train(batches_unsup)
        loss_avg_all_train = self._meta_learn(batches)

        del batches_unsup
        del batches
        return loss_avg_all_train

    def _meta_train(self, batches):
        self.u_net.eval()
        self.normalizer.train()
        fig, axis = plt.subplots()
        learner_losses = []  # (B * num_learner_steps,)
        xticks = []

        for b, (X1, X2) in enumerate(batches):
            # X: (B, 1, H, W)
            X1 = X1.to(self.device)
            X2 = X2.to(self.device)
            for i in range(self.num_learner_steps):
                loss_unsup = self._compute_normalizer_loss(X1, X2)
                self.norm_opt.zero_grad()
                loss_unsup.backward()
                self.norm_opt.step()
                learner_losses.append(loss_unsup.item())
                # xticks.append(float(f"{b}.{i}"))
                xticks.append(b + i / 1000)

        axis.plot(xticks, learner_losses)
        axis.grid(True)
        plt.close()
        del learner_losses
        del xticks
        # loss curve is summarized per epoch
        self.writer.add_figure("learner_traj_epoch", fig, self.global_steps["epoch"])

    def _meta_learn(self, batches):
        self.u_net.train()
        self.normalizer.train()

        loss_avg_all_train = 0
        num_samples = 0
        for X, mask in batches:
            # X: (B, 1, H, W)
            X = X.to(self.device)
            mask = mask.to(self.device)
            loss_sup = self._compute_u_net_loss(X, mask)
            self.norm_opt.zero_grad()
            self.u_net_opt.zero_grad()
            loss_sup.backward()
            self.norm_opt.step()
            self.u_net_opt.step()

            loss_avg_all_train += loss_sup.item() * X.shape[0]
            num_samples += X.shape[0]
            self.writer.add_scalar("loss_meta_learn", loss_sup.item(), self.global_steps["train"])
            self.global_steps["train"] += 1

        loss_avg_all_train /= num_samples
        self.writer.add_scalar("train_epoch", loss_avg_all_train, self.global_steps["epoch"])

        return loss_avg_all_train

    def _pre_train(self, **kwargs):
        self.u_net.train()
        self.normalizer.train()

        if self.notebook:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="train", leave=False)

        loss_avg_all_train = 0
        num_samples = 0
        for i, (X, mask) in pbar:
            # X: (B, 1, H, W)
            X = X.to(self.device)
            mask = mask.to(self.device)
            loss_sup = self._compute_u_net_loss(X, mask)
            self.norm_opt.zero_grad()
            self.u_net_opt.zero_grad()
            loss_sup.backward()
            self.norm_opt.step()
            self.u_net_opt.step()

            pbar.set_description(f"batch {i + 1}/{len(self.train_loader)}: loss_sup: {loss_sup.item():.4f}")

            loss_avg_all_train += loss_sup.item() * X.shape[0]
            num_samples += X.shape[0]
            self.writer.add_scalar("pre_train", loss_sup.item(), self.global_steps["pre_train"])
            self.global_steps["pre_train"] += 1

        loss_avg_all_train /= num_samples
        self.writer.add_scalar("pre_train_epoch", loss_avg_all_train, self.global_steps["pre_train_epochs"])
        pbar.close()

        return loss_avg_all_train

    @torch.no_grad()
    def _eval(self, **kwargs):
        self.normalizer.eval()
        self.u_net.eval()

        if self.notebook:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        pbar = tqdm(enumerate(self.eval_loader), total=len(self.eval_loader), desc="eval", leave=False)

        loss_sup_avg = 0
        num_samples = 0
        for i, (X, mask) in pbar:
            # # debug only #
            # if i > 2:
            #     break
            # ################
            X = X.to(self.device)
            mask = mask.to(self.device)
            loss_sup = self._compute_u_net_loss(X, mask)
            loss_sup_avg += loss_sup * X.shape[0]
            num_samples += X.shape[0]

            pbar.set_description(f"batch {i + 1}/{len(self.eval_loader)}: loss_sup: {loss_sup.item():.4f}")

        loss_sup_avg /= num_samples

        self.writer.add_scalar("eval_loss", loss_sup_avg, self.global_steps["eval"])
        self.global_steps["eval"] += 1
        pbar.close()

        return loss_sup_avg

    @torch.no_grad()
    def _pre_train_eval(self, **kwargs):
        self.normalizer.eval()
        self.u_net.eval()

        if self.notebook:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        pbar = tqdm(enumerate(self.eval_loader), total=len(self.eval_loader), desc="eval", leave=False)

        loss_sup_avg = 0
        num_samples = 0
        for i, (X, mask) in pbar:
            # # debug only #
            # if i > 2:
            #     break
            # ################
            X = X.to(self.device)
            mask = mask.to(self.device)
            loss_sup = self._compute_u_net_loss(X, mask)
            loss_sup_avg += loss_sup * X.shape[0]
            num_samples += X.shape[0]

            pbar.set_description(f"batch {i + 1}/{len(self.eval_loader)}: loss_sup: {loss_sup.item():.4f}")

        loss_sup_avg /= num_samples

        self.writer.add_scalar("pre_train_eval", loss_sup_avg, self.global_steps["pre_train_eval"])
        self.global_steps["pre_train_eval"] += 1
        pbar.close()

        return loss_sup_avg

    def train(self, **kwargs):
        if self.notebook:
            from tqdm.notebook import trange
        else:
            from tqdm import trange

        self._create_save_dir()
        lowest_loss = float("inf")

        pbar = trange(self.pretrain_epochs, desc="pre_train")
        for epoch in pbar:
            pre_train_loss_avg = self._pre_train()
            pre_train_eval_avg = self._pre_train_eval()
            self.global_steps["pre_train_epochs"] += 1
            pbar.set_description(f"epoch {epoch + 1}/{self.pretrain_epochs}: loss_train: {pre_train_loss_avg:.4f}, "
                                 f"loss_eval: {pre_train_eval_avg:.4f}")

        pbar = trange(self.epochs, desc="epoch")

        for epoch in pbar:
            loss_avg_all_train = self._train()

            if self.scheduler is not None:
                for scheduler_iter in self.scheduler:
                    scheduler_iter.step()
                    # for param_group in scheduler_iter.optimizer.param_groups:
                    #     print(f"current lr: {param_group['lr']}")

            # interval = kwargs.get("eval_3d_interval", 1)
            # if epoch % interval == 0:
            if epoch % self.eval_interval_epoch == 0 or epoch == 0 or epoch == self.epochs - 1:
                # print("evaluation")
                # continue

                loss_avg_all_eval = self._eval()
                # if loss_avg_all_eval < lowest_loss:
                #     self._save_params(epoch, loss_avg_all_eval)
                #     # self._save_params(self.global_steps["train"], loss_avg_all_eval)
                #     lowest_loss = loss_avg_all_eval

                losses_eval_3d = {}
                for key in self.test_dataset_dict:
                    self.normalizer_cp.load_state_dict(self.normalizer.state_dict())
                    self.normalizer_cp.eval()
                    dataset = self.test_dataset_dict[key]
                    loss = evaluate_3d_wrapper(evaluate_3d_adapt_batch, dataset, self.normalizer_cp, self.u_net,
                                               self.device, self.notebook, norm_opt_config=self.norm_opt_config,
                                               max_iters=self.num_learner_steps)
                    losses_eval_3d[key] = loss

                for key in losses_eval_3d:
                    self.writer.add_scalar(f"epoch_3d_{key}", losses_eval_3d[key], self.global_steps["eval_3d"])

                cur_loss = sum(list(losses_eval_3d.values())) / 3
                self.writer.add_scalar(f"epoch_3d_avg", cur_loss, self.global_steps["eval_3d"])
                if cur_loss < lowest_loss:
                    self._save_params(epoch, cur_loss)
                    lowest_loss = cur_loss

                fig_orig, fig_adapt, fig_loss_curve = self._end_epoch_plot()
                self.writer.add_figure("epoch_plot_orig", fig_orig, self.global_steps["eval_3d"])
                self.writer.add_figure("epoch_plot_adapt", fig_adapt, self.global_steps["eval_3d"])
                self.writer.add_figure("epoch_plot_curve", fig_loss_curve, self.global_steps["eval_3d"])
                self.global_steps["eval_3d"] += 1

            # loss_avg_all_eval = 0
            # losses_eval_3d = {}

            eval_3d_loss_desc = ""
            for key in losses_eval_3d:
                eval_3d_loss_desc += f", loss_3d_{key}: {losses_eval_3d[key]:.4f}"
            pbar.set_description(f"epoch {epoch + 1}/{self.epochs}, loss_train: {loss_avg_all_train:.4f}, "
                                 f"loss_eval: {loss_avg_all_eval:.4f}" + eval_3d_loss_desc)

            self.global_steps["epoch"] += 1
            self.writer.flush()

    # def _compute_normalizer_loss_old(self, X):
    #     # X: (B, 1, H, W)
    #     # original implementation
    #     X_aug = random_contrast_transform(X)
    #     X = 2 * X - 1
    #     X_aug = 2 * X_aug - 1
    #     loss_fn = lambda X, mask: self.weights["lam_ce"] * cross_entropy_loss(X, mask) + \
    #                               self.weights["lam_dsc"] * dice_loss(X, mask)
    #     X_pred = self.u_net(X_aug)
    #     X_norm_pred = self.u_net(self.normalizer(X))
    #
    #     return symmetric_loss(X_pred, X_norm_pred, loss_fn)

    def _compute_normalizer_loss(self, X1, X2):
        # two transforms
        X_aug_1 = 2 * X1 - 1
        X_aug_2 = 2 * X2 - 1
        loss_fn = lambda X, mask: self.weights["lam_ce"] * cross_entropy_loss(X, mask) + \
                                  self.weights["lam_dsc"] * dice_loss(X, mask)
        X_norm_pred_1 = self.u_net(self.normalizer(X_aug_1))
        X_norm_pred_2 = self.u_net(self.normalizer(X_aug_2))

        return symmetric_loss(X_norm_pred_1, X_norm_pred_2, loss_fn)

    def _compute_normalizer_loss_entropy(self, X, X_dumb=None):
        X = 2 * X - 1
        X_norm_pred = self.u_net(self.normalizer(X))  # (B, K, H, W)
        S = torch.softmax(X_norm_pred, dim=1)
        S_log = -torch.log(S)
        S = S * S_log  # (B, K, H, W)
        S_mean = S.mean(dim=1)  # (B, H, W)

        return S_mean.mean()

    def _compute_u_net_loss(self, X, mask):
        # X: (B, 1, H, W)
        X_orig = X.clone()
        X = 2 * X - 1
        loss_fn = lambda X, mask: self.weights["lam_ce"] * cross_entropy_loss(X, mask) + \
                                  self.weights["lam_dsc"] * dice_loss(X, mask)
        X_norm_pred = self.u_net(self.normalizer(X))

        # loss_unsup = self._compute_normalizer_loss(X_orig)
        #
        # return loss_fn(X_norm_pred, mask) + self.weights["lam_smooth"] * loss_unsup

        return loss_fn(X_norm_pred, mask)

    def _end_epoch_plot(self):
        ind = np.random.randint(len(self.test_loader.dataset))
        X, mask = self.test_loader.dataset[ind]  # (1, H, W), (1, H, W)
        self.normalizer_cp.load_state_dict(self.normalizer.state_dict())
        self.normalizer_cp.eval()
        norm_opt_cp = torch.optim.Adam(self.normalizer_cp.parameters(), **self.norm_opt_config)
        loss_fn = lambda X, mask: self.weights["lam_ce"] * cross_entropy_loss(X, mask) + \
                                  self.weights["lam_dsc"] * dice_loss(X, mask)
        _, fig_orig, fig_adapt, fig_loss_curve = test_time_adaptation(X.unsqueeze(0), mask.unsqueeze(0),
                                                                      self.normalizer_cp, self.u_net, norm_opt_cp,
                                                                      batch_size=1, loss_fn=loss_fn, device=self.device,
                                                                      max_iters=self.num_learner_steps)

        return fig_orig, fig_adapt, fig_loss_curve

    @torch.no_grad()
    def _make_and_save_fig_from_tensor(self, X, title, filename):
        # X: (B, 1, H, W)
        # print(f"{title}: {X.shape}")
        X_grid = make_grid(X.detach().cpu(), nrow=2)  # (3, H', W')
        # print(f"{title}_grid: {X_grid.shape}")
        fig, axis = plt.subplots()
        axis.imshow(X_grid[0], cmap="gray")
        axis.set_title(title)
        fig.savefig(filename)
        plt.close()

    def meta_train_vis(self, loss_type="data-consistency", norm=None, u_net=None):
        # norm, u_net: GPU
        assert loss_type in ["data-consistency", "avg-entropy"]
        self._create_save_dir()

        if norm is None or u_net is None:
            if self.notebook:
                from tqdm.notebook import trange
            else:
                from tqdm import trange

            # self._create_save_dir()
            # lowest_loss = float("inf")

            pbar = trange(self.pretrain_epochs, desc="pre_train")
            for epoch in pbar:
                pre_train_loss_avg = self._pre_train()
                pre_train_eval_avg = self._pre_train_eval()
                self.global_steps["pre_train_epochs"] += 1
                pbar.set_description(f"epoch {epoch + 1}/{self.pretrain_epochs}: loss_train: {pre_train_loss_avg:.4f}, "
                                     f"loss_eval: {pre_train_eval_avg:.4f}")

            self._save_params(self.pretrain_epochs, pre_train_eval_avg)

        else:
            self.normalizer.load_state_dict(norm.state_dict())
            self.u_net.load_state_dict(u_net.state_dict())

        X, mask = sample_from_loader(self.train_loader)  # (B, 1, H, W)
        X_aug_1, X_aug_2 = random_contrast_transform(X), random_contrast_transform(X)
        X = X.to(self.device)
        mask = mask.to(self.device)
        X_aug_1 = X_aug_1.to(self.device)
        X_aug_2 = X_aug_2.to(self.device)
        with torch.no_grad():
            X_pred = self.u_net(self.normalizer(2 * X - 1)).argmax(dim=1, keepdim=True)  # (B, 1, H, W)
            X_aug_1_pred = self.u_net(self.normalizer(2 * X_aug_1 - 1)).argmax(dim=1, keepdim=True)
            X_aug_2_pred = self.u_net(self.normalizer( 2 * X_aug_2 - 1)).argmax(dim=1, keepdim=True)

        self.normalizer.train()
        self.u_net.eval()

        fig, axis = plt.subplots()
        learner_losses = []  # (B * num_learner_steps,)
        xticks = []

        for i in range(self.num_learner_steps):
            if loss_type == "data-consistency":
                loss_unsup = self._compute_normalizer_loss(X_aug_1, X_aug_2)
            elif loss_type == "avg-entropy":
                loss_unsup = self._compute_normalizer_loss_entropy(X, None)
            self.norm_opt.zero_grad()
            loss_unsup.backward()
            self.norm_opt.step()
            learner_losses.append(loss_unsup.item())
            # xticks.append(float(f"{b}.{i}"))
            xticks.append(i)

        axis.plot(xticks, learner_losses)
        axis.grid(True)
        plt.close()

        print("Done adaptation!")
        self.normalizer.eval()
        X_pred_adapt = self.u_net(self.normalizer(2 * X - 1)).argmax(dim=1, keepdim=True)  # (B, 1, H, W)
        X_aug_1_pred_adapt = self.u_net(self.normalizer(2 * X_aug_1 - 1)).argmax(dim=1, keepdim=True)
        X_aug_2_pred_adapt = self.u_net(self.normalizer( 2 * X_aug_2 - 1)).argmax(dim=1, keepdim=True)

        if loss_type == "data-consistency":
            tensors = [X, mask, X_aug_1, X_aug_2, X_pred, X_aug_1_pred, X_aug_2_pred, X_pred_adapt,
                       X_aug_1_pred_adapt, X_aug_2_pred_adapt]
            titles = ["X", "mask", "aug_1", "aug_2", "X_pred", "aug_1_pred", "aug_2_pred", "X_pred_adapt",
                      "aug_1_pred_adapt", "aug_2_pred_adapt"]

        elif loss_type == "avg-entropy":
            tensors = [X, mask, X_pred, X_pred_adapt]
            titles = ["X", "mask", "X_pred", "X_pred_adapt"]

        dir_path = os.path.join(self.param_save_dir, self.time_stamp)
        filenames = [f"{dir_path}/{title}.png" for title in titles]
        for tensor, title, filename in zip(tensors, titles, filenames):
            self._make_and_save_fig_from_tensor(tensor, title, filename)

        fig.savefig(f"{dir_path}/curve.png")
