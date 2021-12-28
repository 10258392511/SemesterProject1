import torch
import os
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from .losses import dice_loss_3d, dice_loss, cross_entropy_loss, symmetric_loss


@torch.no_grad()
def evaluate_3d_no_adapt(X, mask, normalizer, u_net, if_normalizer=False, device=None):
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


@torch.no_grad()
def evaluate_3d_wrapper(evaluate_fn, dataset, normalizer, u_net, device=None, if_notebook=False):
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
        loss = evaluate_fn(X, mask, normalizer, u_net, device=device)
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

    def _train(self):
        raise NotImplementedError

    @torch.no_grad()
    def _eval(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    @torch.no_grad()
    def _end_epoch_plot(self):
        ind = np.random.randint(len(self.test_loader.dataset))
        X, mask = self.test_loader.dataset[ind]  # (1, H, W), (1, H, W)
        # fig = make_summary_plot_simplified(X.unsqueeze(0), mask.unsqueeze(0), self.normalizer, self.u_net,
        #                                    if_show=self.notebook, device=self.device)
        # fig = make_summary_plot_2_by_2(X.unsqueeze(0), mask.unsqueeze(0), self.normalizer, self.u_net,
        #                                if_show=self.notebook, device=self.device)
        fig = make_summary_plot_1_by_3(X.unsqueeze(0), mask.unsqueeze(0), self.normalizer, self.u_net,
                                       if_show=self.notebook, device=self.device)
        return fig

    def _create_save_dir(self):
        if not os.path.isdir(self.param_save_dir):
            os.mkdir(self.param_save_dir)
        os.mkdir(os.path.join(self.param_save_dir, self.time_stamp))

    def _save_params(self, epoch, eval_loss):
        dir_path = os.path.join(self.param_save_dir, self.time_stamp)
        # for filename in os.listdir(dir_path):
        #     if os.path.isfile(filename):
        #         filename_abs = os.path.join(dir_path, filename)
        #         os.remove(filename_abs)
        filename_common = f"epoch_{epoch}_eval_loss_{eval_loss:.4f}".replace(".", "_") + ".pt"
        torch.save(self.normalizer.state_dict(), f"{dir_path}/norm_{filename_common}")
        torch.save(self.u_net.state_dict(), f"{dir_path}/u_net_{filename_common}")


class OnePassTrainer(BasicTrainer):
    def __init__(self, test_dataset_dict, **kwargs):
        super(OnePassTrainer, self).__init__(**kwargs)
        self.test_dataset_dict = test_dataset_dict  # {"csf": ..., "hvhd": ..., "uhe": ...}

    def _train(self):
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
            # loss_sup, loss_unsup = self._compute_normalizer_loss(X, mask)  # only one pass is required
            loss_sup, loss_unsup = self._compute_u_net_loss(X, mask)
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
    def _eval(self):
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
            # loss_sup, loss_unsup = self._compute_normalizer_loss(X, mask)  # only one pass is required
            loss_sup, loss_unsup = self._compute_u_net_loss(X, mask)
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

        # X, mask: (B, 1, H, W), (B, 1, H, W); already sent to self.device; X: [0, 1]
        B, C_in, H, W = X.shape
        X = 2 * X - 1
        X_norm = self.normalizer(X)  # (B, 1, H, W)
        mask_pred_direct = self.u_net(X.expand(B, X_norm.shape[1], H, W))  # (B, C, H, W)
        mask_pred_norm = self.u_net(X_norm)  # (B, C, H, W)
        loss_fn = lambda X, mask: self.weights["lam_ce"] * cross_entropy_loss(X, mask) + \
                                  self.weights["lam_dsc"] * dice_loss(X, mask)
        # TODO: change back
        loss_sup = loss_fn(mask_pred_norm, mask)
        # loss_sup = loss_fn(mask_pred_direct, mask)
        loss_unsup = self.weights["lam_smooth"] * symmetric_loss(mask_pred_direct, mask_pred_norm, loss_fn)

        return loss_sup, loss_unsup

    def _compute_u_net_loss(self, X, mask):
        # return self._compute_normalizer_loss(X, mask)
        X = 2 * X - 1
        mask_pred = self.u_net(X)  # (B, 1, H, W) -> (B, K, H, W)
        loss_fn = lambda X, mask: self.weights["lam_ce"] * cross_entropy_loss(X, mask) + \
                                  self.weights["lam_dsc"] * dice_loss(X, mask)
        loss_sup = loss_fn(mask_pred, mask)

        return loss_sup, loss_sup * self.weights["lam_smooth"]


class AltTrainer(BasicTrainer):
    def __init__(self, test_dataset_dict, **kwargs):
        super(AltTrainer, self).__init__(**kwargs)
        self.test_dataset_dict = test_dataset_dict  # {"csf": ..., "hvhd": ..., "uhe": ...}

    def _train(self):
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
    def _eval(self):
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
