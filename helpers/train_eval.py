import os
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import time

from torchvision.utils import make_grid
from torch_lr_finder import LRFinder
from .utils import real_nvp_preprocess, compute_derivatives, warp_optical_flow, make_summary_plot
from .losses import mask_to_one_hot, cross_entropy_loss, dice_loss, symmetric_loss
from torch.utils.tensorboard import SummaryWriter


def lr_search(model, criterion, optimizer, train_loader, end_lr=1, device=None):
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    lr_finder = LRFinder(model, optimizer, criterion, device)
    lr_finder.range_test(train_loader, end_lr=end_lr, num_iter=100)
    lr_finder.plot()
    lr_finder.reset()

    return lr_finder.history

######################################################################
## CarTraininer ##


class CarTrainer(object):
    def __init__(self, model, train_loader, eval_loader, optimizer,
                 lr_scheduler=None, device=None, epochs=20, notebook=True):
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.epochs = epochs
        self.notebook = notebook

    def _train(self):
        self.model.train()
        if self.notebook:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm

        pbar = tqdm(enumerate(self.train_loader), desc="training", total=len(self.train_loader), leave=False)
        train_losses, lr_rates = [], []
        for i, (img, mask) in pbar:
            img = img.to(self.device)
            mask = mask.to(self.device)
            mask_out = self.model(img)
            loss = self.criterion(mask_out, mask)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            train_losses.append(loss.item())
            lr_rates.append(self.optimizer.param_groups[0]["lr"])

            pbar.set_description(f"training: batch {i + 1}/{len(self.train_loader)}, loss: {train_losses[-1]:.4f}, "
                                 f"lr: {lr_rates[-1]:.4f}")

        pbar.close()

        return train_losses, lr_rates

    @torch.no_grad()
    def _eval(self):
        self.model.eval()

        if self.notebook:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm

        pbar = tqdm(enumerate(self.eval_loader), desc="eval", total=len(self.eval_loader), leave=False)
        avg, num_samples = 0, 0
        for i, (img, mask) in pbar:
            img = img.to(self.device)
            mask = mask.to(self.device)
            mask_out = self.model(img)
            loss = self.criterion(mask_out, mask)
            avg += loss.item() * img.shape[0]
            num_samples += img.shape[0]

        avg /= num_samples
        # pbar.set_description(f"eval: loss: {avg:.4f}")
        pbar.close()

        return avg

    def train(self, if_plot=False):
        # mkdir for current training
        pwd = os.getcwd()
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(cur_dir)
        model_path = os.path.join(parent_dir, "params\car")
        os.chdir(model_path)
        new_dir = f"{time.time()}".replace(".", "_")
        model_path = os.path.join(model_path, new_dir)
        os.mkdir(model_path)
        os.chdir(pwd)  # back to the original working directory

        if self.notebook:
            from tqdm.notebook import trange
        else:
            from tqdm import trange
        pbar = trange(self.epochs, desc="epochs")

        train_losses, eval_losses, lr_rates = [], [], []
        best_eval_loss = float("inf")
        for epoch in pbar:
            train_loss, lr_rate = self._train()
            eval_loss = self._eval()
            train_losses.append(train_loss)
            lr_rates.append(lr_rate)
            eval_losses.append(eval_loss)
            desc = f"train loss: {train_loss[-1]:.4f}, eval_loss: {eval_loss: .4f}"
            pbar.set_description(desc)

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                for filename in os.listdir(model_path):
                    filename = os.path.join(model_path, filename)
                    if os.path.isfile(filename):
                        os.remove(filename)
                save_path = f"epoch_{epoch + 1}_eval_loss_{eval_loss:.4f}".replace(".", "_") + ".pt"
                save_path = os.path.join(model_path, save_path)
                torch.save(self.model.state_dict(), save_path)

            if if_plot:
                index = random.randint(0, len(self.eval_loader.dataset) - 1)
                img_tensor, mask_tensor = self.eval_loader.dataset[index]
                # for img_tensor, mask_tensor in self.eval_loader:
                #     break
                self._end_of_epoch_plot(img_tensor.unsqueeze(0), mask_tensor.unsqueeze(0))

        return train_losses, eval_losses, lr_rates

    @torch.no_grad()
    def _end_of_epoch_plot(self, img_tensor, mask_tensor):
        assert len(img_tensor.shape) == 4 and img_tensor.shape[0] == 1, "only supports (1, C, H, W) input"
        # img_tensor: (1, C, H, W)
        img, mask = predict(self.model, img_tensor, self.device)
        img, mask = img[0].permute(1, 2, 0).numpy(), mask[0].numpy()
        fig, axes = plt.subplots(1, 3)
        axes[0].imshow(img)
        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("predicted")
        axes[2].imshow(mask_tensor[0].detach().cpu().numpy(), cmap="gray")
        axes[2].set_title("original")
        fig.tight_layout()
        plt.show()


@torch.no_grad()
def predict(model, img_tensor, device=None):
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # img_tensor: (B, C, H, W)
    model.eval()
    mask_tensor = model(img_tensor.to(device))  # (B, K, H, W)
    mask_tensor = torch.argmax(mask_tensor, dim=1)  # (B, H, W)
    img, mask = img_tensor.detach().cpu(), mask_tensor.detach().cpu()

    return img, mask


def visualize_predictions(img_tensor, mask_tensor, mask_pred, indices=None):
    # input shape: (B, C, H, W), (B, H, W), (B, H, W)
    assert img_tensor.shape[0] == mask_tensor.shape[0] == mask_pred.shape[0], "all input tensors should have the " \
                                                                              "same length"
    if indices is None:
        indices = [random.randint(0, img_tensor.shape[0] - 1)]
    img, mask = img_tensor.permute(0, 2, 3, 1).numpy(), mask_tensor.numpy()
    fig, axes_all = plt.subplots(len(indices), 3)

    for i in range(len(indices)):
        if len(indices) == 1:
            axes = axes_all
        else:
            axes = axes_all[i, :]
        axes[0].imshow(img[indices[i]])
        axes[1].imshow(mask[indices[i]], cmap="gray")
        axes[1].set_title("original")
        axes[2].imshow(mask_pred[indices[i]].detach().cpu().numpy(), cmap="gray")
        axes[2].set_title("predicted")
    fig.tight_layout()
    plt.show()

######################################################################
## MADETrainer ##


class MADETrainer(object):
    def __init__(self, model, train_loader, eval_loader, optimizer,
                 lr_scheduler=None, device=None, epochs=20, notebook=True):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.device = device if device is not None else torch.device("cuda")
        self.epochs = epochs
        self.notebook = notebook

    def _train(self):
        self.model.train()
        if self.notebook:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        pbar = tqdm(enumerate(self.train_loader), desc="batch training", total=len(self.train_loader), leave=False)
        losses = []

        for i, (X, _) in pbar:
            X = (X >= 0.5)
            X_img = X.float().to(self.device)
            label = X.long().to(self.device)
            # loss = self.model.loss(X_img * 2 - 1, label)
            loss = self.model.loss(X_img, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            pbar.set_description(desc=f"training: batch {i + 1}/{len(self.train_loader)}, loss: {loss.item():.4f}")

        return losses

    @torch.no_grad()
    def _eval(self):
        self.model.eval()
        if self.notebook:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        pbar = tqdm(enumerate(self.eval_loader), desc="batch eval", total=len(self.eval_loader), leave=False)
        avg_loss, num_samples = 0, 0
        for i, (X, _) in pbar:
            X = (X >= 0.5)
            X_img = X.float().to(self.device)
            label = X.long().to(self.device)
            # loss = self.model.loss(X_img * 2 - 1, label)
            loss = self.model.loss(X_img, label)
            avg_loss += loss.item() * X.shape[0]
            num_samples += X.shape[0]
            pbar.set_description(desc=f"eval: batch {i + 1}/{len(self.eval_loader)}, loss: {loss.item():.4f}")

        return avg_loss / num_samples

    def train(self, if_plot=False):
        model_path = self._change_and_make_dir()
        if self.notebook:
            from tqdm.notebook import trange
        else:
            from tqdm import trange
        pbar = trange(self.epochs, desc=f"epoch 1/{self.epochs}")
        train_losses, eval_losses = [], []
        best_eval_loss = float("inf")

        for epoch in pbar:
            train_loss = self._train()
            train_losses.append(train_loss)
            eval_loss = self._eval()
            eval_losses.append(eval_loss)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            pbar.set_description(f"epoch {epoch + 1} loss: training: {np.array(train_loss).mean():.4f}, "
                                 f"eval: {eval_loss:.4f}")
            if if_plot:
                self._end_of_epoch_plot(title=f"epoch {epoch + 1}")

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                for filename in os.listdir(model_path):
                    filename = os.path.join(model_path, filename)
                    if os.path.isfile(filename):
                        os.remove(filename)
                save_path = f"epoch_{epoch + 1}_eval_loss_{eval_loss:.4f}".replace(".", "_") + ".pt"
                save_path = os.path.join(model_path, save_path)
                torch.save(self.model.state_dict(), save_path)

        return train_losses, eval_losses

    @torch.no_grad()
    def _end_of_epoch_plot(self, **kwargs):
        # for MNIST
        sample = self.model.sample(1)  # (1, H, W)
        fig, axis = plt.subplots()
        title = kwargs.get("title", "")
        handle = axis.imshow(sample[0, ...], cmap="gray")
        plt.colorbar(handle, ax=axis)
        axis.set_title(title)
        plt.show()

    def _change_and_make_dir(self):
        orginal_dir = os.getcwd()
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(cur_dir)
        model_path = os.path.join(parent_dir, "params\made_mnist")
        os.chdir(model_path)
        new_dir = f"{time.time()}".replace(".", "_")
        model_path = os.path.join(model_path, new_dir)
        os.mkdir(model_path)
        os.chdir(orginal_dir)

        # params\made_mnist\time_stamp
        return model_path

######################################################################
## RealNVPTrainer ##


class RealNVPTrainer(object):
    def __init__(self, model, train_loader, eval_loader, optimizer,
                 lr_scheduler=None, device=None, epochs=20, notebook=True):
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = torch.device("cuda") if device is None else device
        self.epochs = epochs
        self.notebook = notebook

    def _train(self):
        self.model.train()
        if self.notebook:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        pbar = tqdm(enumerate(self.train_loader), desc="training", total=len(self.train_loader), leave=False)
        losses = []
        for i, (X, _) in pbar:
            if i > 25:
                break
            X = X.float().to(self.device) * 255  # [0, 255]
            X, log_det = real_nvp_preprocess(X)  # (B, C, H, W), (B,)
            log_prob_model = self.model.log_prob(X)
            loss = -(log_prob_model + log_det).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            pbar.set_description(f"batch {i + 1}/{len(self.train_loader)}: training loss: {losses[-1]:.4f}")

        return losses

    @torch.no_grad()
    def _eval(self):
        self.model.eval()
        if self.notebook:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        pbar = tqdm(enumerate(self.eval_loader), desc="validation", total=len(self.eval_loader), leave=False)
        num_samples = 0
        loss_avg = 0
        for i, (X, _) in pbar:
            if i > 10:
                break
            X = X.float().to(self.device) * 255  # [0, 255]
            X, log_det = real_nvp_preprocess(X)  # (B, C, H, W), (B,)
            log_prob_model = self.model.log_prob(X)
            loss = -(log_prob_model + log_det).mean()
            loss_avg += loss * X.shape[0]
            num_samples += X.shape[0]
            pbar.set_description(f"batch {i + 1}/{len(self.eval_loader)}: eval loss: {loss.item():.4f}")

        return loss_avg / num_samples

    def train(self, if_plot=True, model_save_dir=None):
        """
        model_save_dir: root of all param folders
        """
        if self.notebook:
            from tqdm.notebook import trange
        else:
            from tqdm import trange
        time_stamp = self._create_save_folder(model_save_dir)

        pbar = trange(self.epochs, desc="epochs")
        train_losses, eval_losses = [], []
        lowest_loss = float("inf")
        for epoch in pbar:
            train_loss = self._train()
            eval_loss = self._eval()
            # save the best model so far
            if eval_loss < lowest_loss:
                lowest_loss = eval_loss
                if model_save_dir is not None:
                    self._save_latest_model(time_stamp, epoch, eval_loss, model_save_dir)

            train_losses.append(train_loss)
            eval_losses.append(eval_loss)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            pbar.set_description(f"epoch: {epoch + 1}/{self.epochs}, training loss: {train_loss[-1]:.4f}, "
                                 f"eval loss: {eval_loss[-1]:.4f}")
        if if_plot:
            self._eval_sample_plot()

        return train_losses, eval_losses

    def _create_save_folder(self, model_save_dir=None):
        """
        This should be called at the root of the project. model_save_dir is the root folder:
        model_save_dir/
        -time_stamp_and_additional_info/
        --model.pt (to save in self._save_latest_model(.))
        """
        assert model_save_dir is not None, "please specify a save directory"
        if not os.path.isdir(model_save_dir):
            os.mkdir(model_save_dir)
        original_wd = os.getcwd()
        os.chdir(model_save_dir)
        time_stamp = f"{time.time()}".replace(".", "_")
        os.mkdir(time_stamp)
        os.chdir(original_wd)

        return time_stamp

    def _save_latest_model(self, time_stamp, epoch, eval_loss, model_save_dir=None):
        assert model_save_dir is not None, "please specify a save directory"
        original_wd = os.getcwd()
        os.chdir(model_save_dir)
        os.chdir(time_stamp)
        # remove all old files
        for filename in os.listdir():
            if os.path.isfile(filename):
                os.remove(filename)
        filename = f"epoch_{epoch + 1}_eval_loss_{eval_loss:.4f}".replace(".", "_") + ".pt"
        torch.save(self.model.state_dict(), filename)
        os.chdir(original_wd)

    @torch.no_grad()
    def _eval_sample_plot(self):
        X = None
        for (X, _) in self.eval_loader:
            break
        sample = self.model.sample(1, X.shape[1:])
        sample_img, _ = real_nvp_preprocess(sample, if_reverse=True)  # (1, C, H, W)
        plt.imshow(sample_img[0, 0], cmap="gray")
        plt.colorbar()
        plt.show()

######################################################################
## MNISTVAETrainer ##
class MNISTVAETrainer(object):
    def __init__(self, vae, int_warper, shape_warper, train_loader, eval_loader, vae_optimizer,
                 int_warper_optimizer, shape_warper_optimizer, lr_scheduler=None,
                 epochs=20, warper_batch_portion=0.5, deriv_win_size=5, lamda_int_warper=1, lamda_shape_warper=1,
                 lamda_warp_recons=1, loss_type="MSE", device=None, notebook=True):
        assert loss_type in ["MSE", "CE"], "loss type should be 'MSE' or 'CE'"
        self.vae = vae
        self.int_warper = int_warper
        self.shape_warper = shape_warper
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.vae_optimizer = vae_optimizer
        self.int_warper_optimizer = int_warper_optimizer
        self.shape_warper_optimizer = shape_warper_optimizer
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs
        self.device = device if device is not None else torch.device("cuda")
        self.notebook = notebook
        self.warper_batch_portion = warper_batch_portion
        self.deriv_win_size = deriv_win_size
        self.lamda_int_warper = lamda_int_warper
        self.lamda_shape_warper = lamda_shape_warper
        self.lamda_warp_recons = lamda_warp_recons
        self.loss_type = loss_type
        assert (self.loss_type == "MSE" and not self.vae.if_cross_entropy and not self.int_warper.if_cross_entropy
                and not self.shape_warper.if_cross_entropy) or \
               (self.loss_type == "CE" and self.vae.if_cross_entropy and self.int_warper.if_cross_entropy and
                self.shape_warper.if_cross_entropy), "loss type doesn't match VAE and/or warper configuration"

    def _train(self):
        self.vae.train()
        self.int_warper.train()
        self.shape_warper.train()
        if self.notebook:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        losses = dict(int_warper=[], shape_warper=[], int_warper_smooth=[], shape_warper_smooth=[],
                      recons=[], kl=[], int_warper_vae=[], shape_warper_vae=[])
        pbar = tqdm(enumerate(self.train_loader), total=int(len(self.train_loader) * self.warper_batch_portion),
                    desc="training", leave=False)
        # train warpers
        for i, (X1, X2) in pbar:
            if i > self.warper_batch_portion * len(self.train_loader):
                break
            X1, X2 = X1.float().to(self.device), X2.float().to(self.device)
            Z1_mu, Z1_log_sigma, Z2_mu, Z2_log_sigma, \
            X1_hat, X2_hat, X_int_interp_shape_1, X_int_interp_shape_2, X_int_1_shape_interp, X_int_2_shape_interp = \
                self.vae(X1, X2)  # (B, lat_dim) for Z, (B, C, H, W) for X

            # intensity warper
            loss_recons1, loss_smooth1 = self._compute_int_warper_loss(X_int_interp_shape_1, X1)
            loss_recons2, loss_smooth2 = self._compute_int_warper_loss(X_int_interp_shape_2, X2)
            loss_int_warper = (loss_recons1 + loss_recons2) / 2
            loss_int_warper_smooth = (loss_smooth1 + loss_smooth2) / 2
            loss_int_all = loss_int_warper + self.lamda_int_warper * loss_int_warper_smooth
            self.int_warper_optimizer.zero_grad()
            loss_int_all.backward(retain_graph=True)
            self.int_warper_optimizer.step()
            losses["int_warper"].append(loss_int_warper.item())
            losses["int_warper_smooth"].append(loss_int_warper_smooth.item())

            # shape warper
            loss_recons1, loss_smooth1 = self._compute_shape_warper_loss(X_int_1_shape_interp, X1)
            loss_recons2, loss_smooth2 = self._compute_shape_warper_loss(X_int_2_shape_interp, X2)
            loss_shape_warper = (loss_recons1 + loss_recons2) / 2
            loss_shape_warper_smooth = (loss_smooth1 + loss_smooth2) / 2
            loss_shape_all = loss_shape_warper + self.lamda_shape_warper * loss_shape_warper_smooth
            self.shape_warper_optimizer.zero_grad()
            loss_shape_all.backward()
            self.shape_warper_optimizer.step()
            losses["shape_warper"].append(loss_shape_warper.item())
            losses["shape_warper_smooth"].append(loss_shape_warper_smooth.item())

            pbar.set_description(f"int_warper: {loss_int_all.item():.4f}, shape_warper: {loss_shape_all.item():.4f}, "
                                 f"int_warper_smooth: {loss_int_warper_smooth.item():.4f}, "
                                 f"shape_warper_smooth: {loss_shape_warper_smooth.item():.4f}")

        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="training", leave=False)

        # train VAE
        for i, (X1, X2) in pbar:
            X1, X2 = X1.float().to(self.device), X2.float().to(self.device)
            Z1_mu, Z1_log_sigma, Z2_mu, Z2_log_sigma, \
            X1_hat, X2_hat, X_int_interp_shape_1, X_int_interp_shape_2, X_int_1_shape_interp, X_int_2_shape_interp = \
                self.vae(X1, X2)  # (B, lat_dim) for Z, (B, C, H, W) for X
            loss_recons1 = self._compute_vae_recons_loss(X1_hat, X1)
            loss_recons2 = self._compute_vae_recons_loss(X2_hat, X2)
            loss_recons = (loss_recons1 + loss_recons2) / 2
            loss_kl = (-Z1_log_sigma - Z2_log_sigma - 0.5 * 2 +
                       0.5 * ((2 * Z1_log_sigma).exp() + (2 * Z2_log_sigma).exp() + Z1_mu ** 2) + Z2_mu ** 2) / 2
            loss_kl = loss_kl.sum(dim=1).mean()  # (B, lam_dim) -> (B,) -> float

            # int_warper loss
            loss_recons1, _ = self._compute_int_warper_loss(X_int_interp_shape_1, X1)
            loss_recons2, _ = self._compute_int_warper_loss(X_int_interp_shape_2, X2)
            loss_int_warper = (loss_recons1 + loss_recons2) / 2

            # shape_warper loss
            loss_recons1, _ = self._compute_shape_warper_loss(X_int_1_shape_interp, X1)
            loss_recons2, _ = self._compute_shape_warper_loss(X_int_2_shape_interp, X2)
            loss_shape_warper = (loss_recons1 + loss_recons2) / 2
            loss_vae_all = loss_recons + loss_kl + self.lamda_warp_recons * (loss_int_warper + loss_shape_warper)
            self.vae_optimizer.zero_grad()
            loss_vae_all.backward()
            self.vae_optimizer.step()
            losses["recons"].append(loss_recons.item())
            losses["kl"].append(loss_kl.item())
            losses["int_warper_vae"].append(loss_int_warper.item())
            losses["shape_warper_vae"].append(loss_shape_warper.item())

            pbar.set_description(f"loss_all: {loss_vae_all.item():.4f}, recons: {loss_recons.item():.4f}, "
                                 f"kl: {loss_kl.item():.4f}, int_warper: {loss_int_warper.item():.4f}, "
                                 f"shape_warper: {loss_shape_warper.item():.4f}")

        return losses

    @torch.no_grad()
    def _eval(self):
        self.vae.eval()
        self.int_warper.eval()
        self.shape_warper.eval()

        if self.notebook:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm

        pbar = tqdm(enumerate(self.eval_loader), total=len(self.eval_loader), desc="validation", leave=False)
        num_samples = 0
        losses = dict(recons=0, kl=0, int_warper_vae=0, shape_warper_vae=0)
        # train VAE
        for i, (X1, X2) in pbar:
            X1, X2 = X1.float().to(self.device), X2.float().to(self.device)
            Z1_mu, Z1_log_sigma, Z2_mu, Z2_log_sigma, \
            X1_hat, X2_hat, X_int_interp_shape_1, X_int_interp_shape_2, X_int_1_shape_interp, X_int_2_shape_interp = \
                self.vae(X1, X2)  # (B, lat_dim) for Z, (B, C, H, W) for X
            loss_recons1 = self._compute_vae_recons_loss(X1_hat, X1)
            loss_recons2 = self._compute_vae_recons_loss(X2_hat, X2)
            loss_recons = (loss_recons1 + loss_recons2) / 2
            loss_kl = (-Z1_log_sigma - Z2_log_sigma - 0.5 * 2 +
                       0.5 * ((2 * Z1_log_sigma).exp() + (2 * Z2_log_sigma).exp() + Z1_mu ** 2) + Z2_mu ** 2) / 2
            loss_kl = loss_kl.sum(dim=1).mean()  # (B, lam_dim) -> (B,) -> float

            # int_warper loss
            loss_recons1, _ = self._compute_int_warper_loss(X_int_interp_shape_1, X1)
            loss_recons2, _ = self._compute_int_warper_loss(X_int_interp_shape_2, X2)
            loss_int_warper = (loss_recons1 + loss_recons2) / 2

            # shape_warper loss
            loss_recons1, _ = self._compute_shape_warper_loss(X_int_1_shape_interp, X1)
            loss_recons2, _ = self._compute_shape_warper_loss(X_int_2_shape_interp, X2)
            loss_shape_warper = (loss_recons1 + loss_recons2) / 2

            loss_vae_all = loss_recons + loss_kl + self.lamda_warp_recons * (loss_int_warper + loss_shape_warper)

            pbar.set_description(f"loss_all: {loss_vae_all.item():.4f}, recons: {loss_recons.item():.4f}, "
                                 f"kl: {loss_kl.item():.4f}, int_warper: {loss_int_warper.item():.4f}, "
                                 f"shape_warper: {loss_shape_warper.item():.4f}")

            losses["recons"] += loss_recons * X1.shape[0]
            losses["kl"] += loss_kl * X1.shape[0]
            losses["int_warper_vae"] += loss_int_warper * X1.shape[0]
            losses["shape_warper_vae"] += loss_shape_warper * X1.shape[0]
            num_samples += X1.shape[0]

        for key in losses:
            losses[key] /= num_samples

        return losses

    def _compare_two_imgs(self, X_hat, X, loss_func="MSE"):
        assert loss_func in ["MSE", "CE"], "only support MSE and CE"
        # X_hat, X: (B, 3, 28, 28), [0, 1]
        if loss_func == "MSE":
            loss = nn.MSELoss(reduction="none")
            # loss = nn.L1Loss(reduction="none")
            loss_mse = loss(X_hat, X)
            return loss_mse.sum(dim=[1, 2, 3]).mean()
        else:
            loss = nn.CrossEntropyLoss(reduction="none")
            X = (X * 255).long()
            # plt.imshow(X[0].permute(1, 2, 0).detach().cpu().numpy())
            # plt.show()
            loss_ce = loss(X_hat, X)  # (B, C, H, W)
            # return loss_ce.sum(dim=[1, 2, 3]).mean()
            return loss_ce.mean()

    def _compute_int_warper_loss(self, X_pred, X_tgt):
        if self.loss_type == "MSE":
            # X_pred, X_tgt: (B, 3, 28, 28), float
            log_s, t = torch.chunk(self.int_warper(X_pred, X_tgt), 2, dim=1)  # (B, 1, H, W) each
            X_tgt_hat = X_pred * log_s.exp() + t  ### intensity warper: per pixel on all channels
            loss_recons = self._compare_two_imgs(X_tgt_hat, X_tgt, "MSE")
            loss_smooth = 0
            for img in [log_s, t]:
                Ix, Iy = compute_derivatives(img, self.deriv_win_size)  # (B, 1, H, W) each
                loss_smooth += torch.abs(Ix).sum(dim=[1, 2, 3]).mean() + torch.abs(Iy).sum(dim=[1, 2, 3]).mean()

            return loss_recons, loss_smooth
        else:
            # X_pred, X_tgt: (B, 3 * 256, 28, 28), (B, 3, 28, 28), float
            weights = self.int_warper(X_pred, X_tgt)  # (B, K, H, W) -> (B, 1, K, H, W) (next line)
            X_tgt_hat = X_pred.view(-1, 3, 256, 28, 28) * weights.unsqueeze(1)  # (B, C, K, H, W) * (B, 1, K, H, W)
            # (B, C, K, H, W) -> (B, K, C, H, W)
            # (X_tgt * 255).long(): converted in .compare_two_imgs(.)
            loss_recons = self._compare_two_imgs(X_tgt_hat.permute(0, 2, 1, 3, 4), X_tgt, "CE")
            Ix, Iy = compute_derivatives(weights, self.deriv_win_size, True)  # (B, K, H_valid, W_valid) each
            loss_smooth = torch.abs(Ix).sum(dim=[2, 3]).mean() + torch.abs(Iy).sum(dim=[1, 2, 3]).mean()

            return loss_recons, loss_smooth

    def _compute_shape_warper_loss(self, X_pred, X_tgt):
        if self.loss_type == "MSE":
            # X_pred, X_tgt: (B, 3, 28, 28), float
            uv = self.shape_warper(X_pred, X_tgt)  # (B, 2, H, W)
            X_tgt_hat = warp_optical_flow(X_pred, uv.permute(0, 2, 3, 1))  # (B, 2, H, W) -> (B, H, W, 2)
            loss_recons = self._compare_two_imgs(X_tgt_hat, X_tgt, "MSE")

        else:
            # X_pred, X_tgt: (B, 3 * 256, 28, 28), (B, 3, 28, 28), float
            uv = self.shape_warper(X_pred, X_tgt)  # (B, 2, H, W)
            X_tgt_hat = warp_optical_flow(X_pred, uv.permute(0, 2, 3, 1))  # (B, 2, H, W) -> (B, H, W, 2)
            X_tgt_hat = X_tgt_hat.view(-1, 3, 256, 28, 28)
            loss_recons = self._compare_two_imgs(X_tgt_hat.permute(0, 2, 1, 3, 4), X_tgt, "CE")

        loss_smooth = 0
        for i in range(uv.shape[1]):
            img = uv[:, i:i + 1, ...]  # (B, 1, H, W)
            Ix, Iy = compute_derivatives(img)
            loss_smooth += torch.abs(Ix).sum(dim=[1, 2, 3]).mean() + torch.abs(Iy).sum(dim=[1, 2, 3]).mean()

        return loss_recons, loss_smooth

    def _compute_vae_recons_loss(self, X_pred, X_tgt):
        if self.loss_type == "MSE":
            loss_recons = self._compare_two_imgs(X_pred, X_tgt, "MSE")
        else:
            loss_recons = self._compare_two_imgs(X_pred.view(-1, 3, 256, 28, 28).permute(0, 2, 1, 3, 4), X_tgt, "CE")

        return loss_recons

    def train(self, if_plot=True):
        if self.notebook:
            from tqdm.notebook import trange
        else:
            from tqdm import trange

        model_save_dir = "./params/vae_warper_mnist"
        time_stamp = self._create_save_folder(model_save_dir=model_save_dir)
        pbar = trange(self.epochs, desc="epochs")
        train_losses, eval_losses = [], []
        lowest_loss = float("inf")
        for epoch in pbar:
            train_loss = self._train()
            eval_loss = self._eval()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            train_losses.append(train_loss)
            eval_losses.append(eval_loss)
            cur_loss = eval_loss["recons"] + eval_loss["kl"] + \
                       self.lamda_warp_recons * (eval_loss["int_warper_vae"] + eval_loss["shape_warper_vae"])
            if cur_loss < lowest_loss:
                lowest_loss = cur_loss
                self._save_latest_model(time_stamp, epoch, eval_loss, model_save_dir)

            if if_plot:
                print(f"epoch {epoch + 1}/{self.epochs}")
                self._eval_sample_plot()
                self._check_warper()

            cur_train_loss = train_loss["recons"][-1] + train_loss["kl"][-1] + \
                       self.lamda_warp_recons * (train_loss["int_warper_vae"][-1] + train_loss["shape_warper_vae"][-1])
            pbar.set_description(f"epoch: {epoch + 1}/{self.epochs}: training loss: {cur_train_loss:.4f}, eval loss:"
                                 f"{cur_loss:.4f}")

        return train_losses, eval_losses

    def _create_save_folder(self, model_save_dir=None):
        """
        This should be called at the root of the project. model_save_dir is the root folder:
        model_save_dir/
        -time_stamp_and_additional_info/
        --model.pt (to save in self._save_latest_model(.))
        """
        assert model_save_dir is not None, "please specify a save directory"
        if not os.path.isdir(model_save_dir):
            os.mkdir(model_save_dir)
        original_wd = os.getcwd()
        os.chdir(model_save_dir)
        time_stamp = f"{time.time()}".replace(".", "_")
        os.mkdir(time_stamp)
        os.chdir(original_wd)

        return time_stamp

    def _save_latest_model(self, time_stamp, epoch, eval_loss, model_save_dir=None):
        # eval_loss; dict
        assert model_save_dir is not None, "please specify a save directory"
        original_wd = os.getcwd()
        os.chdir(model_save_dir)
        os.chdir(time_stamp)
        # remove all old files
        for filename in os.listdir():
            if os.path.isfile(filename):
                os.remove(filename)
        filename_common = f"{self.loss_type}_epoch_{epoch + 1}"
        for key in eval_loss:
            filename_common += f"_{key}_{eval_loss[key]:.4f}"
        filename_common = filename_common.replace(".", "_")
        filename_common += ".pt"
        torch.save(self.vae.state_dict(), "vae_" + filename_common)
        torch.save(self.int_warper.state_dict(), "int_warper_" + filename_common)
        torch.save(self.shape_warper.state_dict(), "shape_warper_" + filename_common)
        os.chdir(original_wd)

    @torch.no_grad()
    def _eval_sample_plot(self, time_steps=5):
        X1, X2 = None, None
        for X1, X2 in self.eval_loader:
            # X1, X2: (B, 3, 28, 28)
            break
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(X1[0, ...].permute(1, 2, 0).detach().cpu().numpy())
        axes[1].imshow(X2[0, ...].permute(1, 2, 0).detach().cpu().numpy())
        plt.show()

        X1 = X1.to(self.device)
        X2 = X2.to(self.device)
        self.vae.interpolate(X1[0:1, ...], X2[0:1, ...], time_steps=time_steps)

    @torch.no_grad()
    def _check_warper(self):
        X1, X2 = None, None
        for X1, X2 in self.eval_loader:
            # X1, X2: (B, 3, 28, 28)
            break
        X1 = X1.to(self.device)
        X2 = X2.to(self.device)
        # fig, axes = plt.subplots(1, 2)
        # axes[0].imshow(X1[0, ...].permute(1, 2, 0).detach().cpu().numpy())
        # axes[1].imshow(X2[0, ...].permute(1, 2, 0).detach().cpu().numpy())
        # plt.show()

        img1, img2 = X1[0:1, ...], X2[0:1, ...]  # (1, 3, 28, 28) each
        _, _, _, _, \
        X1, X2, X_int_interp_shape_1, X_int_interp_shape_2, X_int_1_shape_interp, X_int_2_shape_interp = self.vae(img1, img2)

        # log_s_1, t_1 = torch.chunk(self.int_warper(X_int_interp_shape_1, X1), 2, dim=1)  # (B, 1, H, W) each
        # X1_hat_int = X_int_interp_shape_1 * log_s_1.exp() + t_1  ### intensity warper: per pixel on all channels
        # log_s_2, t_2 = torch.chunk(self.int_warper(X_int_interp_shape_2, X2), 2, dim=1)
        # X2_hat_int = X_int_interp_shape_2 * log_s_2.exp() + t_2
        if self.loss_type == "MSE":
            # X_pred, X_tgt: (B, 3, 28, 28), float
            log_s_1, t_1 = torch.chunk(self.int_warper(X_int_interp_shape_1, img1), 2, dim=1)  # (B, 1, H, W) each
            X1_hat_int = X_int_interp_shape_1 * log_s_1.exp() + t_1  ### intensity warper: per pixel on all channels
            log_s_2, t_2 = torch.chunk(self.int_warper(X_int_interp_shape_2, img2), 2, dim=1)
            X2_hat_int = X_int_interp_shape_2 * log_s_2.exp() + t_2

            uv_1 = self.shape_warper(X_int_1_shape_interp, img1)  # (B, 2, H, W)
            X1_hat_shape = warp_optical_flow(X_int_1_shape_interp, uv_1.permute(0, 2, 3, 1))
            uv_2 = self.shape_warper(X_int_2_shape_interp, img2)  # (B, 2, H, W)
            X2_hat_shape = warp_optical_flow(X_int_2_shape_interp, uv_2.permute(0, 2, 3, 1))

            warper_grid = torch.cat([log_s_1, t_1, log_s_2, t_2, *uv_1.chunk(2, dim=1), *uv_2.chunk(2, dim=1)], dim=0)

        else:
            # X_pred, X_tgt: (B, 3 * 256, 28, 28), (B, 3, 28, 28), float
            # print(f"X_int_interp_shape_1: {X_int_interp_shape_1.shape}")
            weights_1 = self.int_warper(X_int_interp_shape_1, img1)  # (B, K, H, W) -> (B, 1, K, H, W) (next line)
            weights_2 = self.int_warper(X_int_interp_shape_2, img2)  # (B, K, H, W) -> (B, 1, K, H, W) (next line)
            # (B, C, K, H, W) * (B, 1, K, H, W)
            X1_hat_int = X_int_interp_shape_1.view(-1, 3, 256, 28, 28) * weights_1.unsqueeze(1)
            X2_hat_int = X_int_interp_shape_2.view(-1, 3, 256, 28, 28) * weights_2.unsqueeze(1)
            X1_hat_int = torch.argmax(X1_hat_int, dim=2)
            X2_hat_int = torch.argmax(X2_hat_int, dim=2)
            # print(f"X1_hat_int: {X1_hat_int.shape}, X2_hat_int: {X2_hat_int.shape}")

            # print(f"X_int_1_shape_interp: {X_int_1_shape_interp.shape}, img1: {img1.shape}")
            uv_1 = self.shape_warper(X_int_1_shape_interp, img1)  # (B, 2, H, W)
            X1_hat_shape = warp_optical_flow(X_int_1_shape_interp, uv_1.permute(0, 2, 3, 1))
            uv_2 = self.shape_warper(X_int_2_shape_interp, img2)  # (B, 2, H, W)
            X2_hat_shape = warp_optical_flow(X_int_2_shape_interp, uv_2.permute(0, 2, 3, 1))
            X1_hat_shape = torch.argmax(X1_hat_shape.view(-1, 3, 256, 28, 28), dim=2)
            X2_hat_shape = torch.argmax(X2_hat_shape.view(-1, 3, 256, 28, 28), dim=2)
            # print(f"X1_hat_shape: {X1_hat_shape.shape}, X2_hat_shape: {X2_hat_shape.shape}")
            X1 = torch.argmax(X1.view(-1, 3, 256, 28, 28), dim=2)
            X2 = torch.argmax(X2.view(-1, 3, 256, 28, 28), dim=2)
            X_int_interp_shape_1 = torch.argmax(X_int_interp_shape_1.view(-1, 3, 256, 28, 28), dim=2)
            X_int_interp_shape_2 = torch.argmax(X_int_interp_shape_2.view(-1, 3, 256, 28, 28), dim=2)
            X_int_1_shape_interp = torch.argmax(X_int_1_shape_interp.view(-1, 3, 256, 28, 28), dim=2)
            X_int_2_shape_interp = torch.argmax(X_int_2_shape_interp.view(-1, 3, 256, 28, 28), dim=2)

            warper_grid = torch.cat([*uv_1.chunk(2, dim=1), *uv_2.chunk(2, dim=1)], dim=0)

        img_grid = torch.cat([img1, img2, X1, X2, X_int_interp_shape_1, X_int_interp_shape_2,
                              X_int_1_shape_interp, X_int_2_shape_interp, X1_hat_int, X2_hat_int,
                              X1_hat_shape, X2_hat_shape], dim=0)

        img_grid = make_grid(img_grid, nrow=2)
        warper_grid = make_grid(warper_grid, nrow=2)
        plt.imshow(img_grid.permute(1, 2, 0).detach().cpu().numpy())
        plt.show()
        plt.imshow(warper_grid[0].detach().cpu().numpy(), cmap="gray")  # (1, H', W')
        plt.colorbar()
        plt.show()


######################################################################
## Normalizer + U-Net Trainer ##
# This is a copy
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
        layers += [nn.ReLU(), nn.Conv2d(intermediate_channels, 1, kernel_size, padding=padding)]
        self.layers = nn.Sequential(*layers)
        self.conv_layers = [layer for layer in layers if isinstance(layer, nn.Conv2d)]

    def forward(self, x):
        # x: (B, C, H, W)

        # x_out: (B, C, H, W)
        return self.layers(x)


class AlternatingTrainer(object):
    def __init__(self, normalizer, u_net, train_loader, eval_loader, test_loader,
                 norm_optimizer, u_net_optimizer, lr_scheduler=None,
                 epochs=20, device=None, notebook=True, num_classes=4, smooth_weight=1,
                 img_save_dir=None, writer: SummaryWriter = None, ce_weight=1, dsc_weight=0, time_stamp=None):
        # assert loss_type in ["CE", "DSC"], "only supports CE and DSC"
        self.normalizer = normalizer
        self.u_net = u_net
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.test_loader = test_loader
        self.norm_optimizer = norm_optimizer
        self.u_net_optimizer = u_net_optimizer
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs
        self.device = torch.device("cuda") if device is None else device
        self.notebook = notebook
        self.ce_weight = ce_weight
        self.dsc_weight = dsc_weight
        self.loss_fn = lambda X, mask: ce_weight * cross_entropy_loss(X, mask) + dsc_weight * dice_loss(X, mask)
        self.num_classes = num_classes
        self.smooth_weight = smooth_weight
        self.img_save_dir = img_save_dir
        self.time_stamp = time_stamp
        self.writer = writer
        self.global_steps = {"train": 0, "eval": 0, "epoch": 0}
        if not self.notebook:
            assert self.img_save_dir is not None, "please specify a directory for saving images"

    def _train(self):
        if self.notebook:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        pbar = tqdm(enumerate(self.train_loader), desc="training", total=len(self.train_loader), leave=False)
        losses = {"norm": [], "seg-main": [], "seg-smooth": [], "seg-all": []}
        for i, (X, X_aug, mask) in pbar:
            # ###########
            # # TODO: Comment out
            # if i >= 2:
            #     break
            # ##########
            X = (2 * X - 1).float().to(self.device)
            X_aug = (2 * X_aug - 1).float().to(self.device)
            mask = mask.to(self.device)
            # update U-Net
            self.normalizer.eval()
            self.u_net.train()
            mask_pred = self.u_net(X)  # (B, K, H, W)
            loss_main = self.loss_fn(mask_pred, mask)
            X_norm = self.normalizer(X)
            mask_pred_norm = self.u_net(X_norm)
            mask_pred_aug = self.u_net(X_aug)
            loss_smooth = symmetric_loss(mask_pred_norm, mask_pred_aug, self.loss_fn)
            loss_all = loss_main + self.smooth_weight * loss_smooth

            self.u_net_optimizer.zero_grad()
            loss_all.backward()
            self.u_net_optimizer.step()
            losses["seg-main"].append(loss_main.item())
            losses["seg-smooth"].append(loss_smooth.item())
            losses["seg-all"].append(loss_all.item())

            # update Normalizer
            self.u_net.eval()
            self.normalizer.train()
            # mask_pred = self.u_net(X)  # (B, K, H, W)
            # loss_main = self.loss_fn(mask_pred, mask)
            X_norm = self.normalizer(X)
            mask_pred_norm = self.u_net(X_norm)
            loss_main = self.loss_fn(mask_pred_norm, mask)
            mask_pred_aug = self.u_net(X_aug)
            loss_smooth = symmetric_loss(mask_pred_norm, mask_pred_aug, self.loss_fn)
            loss_all = loss_main + self.smooth_weight * loss_smooth

            self.norm_optimizer.zero_grad()
            loss_all.backward()
            self.norm_optimizer.step()
            losses["norm"].append(loss_main.item())

            pbar.set_description(f"training batch: {i + 1}/{len(self.train_loader)}: norm: {losses['norm'][-1]:.4f}, "
                                 f"loss-main: {losses['seg-main'][-1]:.4f}, "
                                 f"loss-smooth: {losses['seg-smooth'][-1]:.4f}, "
                                 f"loss-all: {losses['seg-all'][-1]:.4f}")

            self.writer.add_scalar("train_norm", losses["norm"][-1], self.global_steps["train"])
            self.writer.add_scalar("train_loss_main", losses["seg-main"][-1], self.global_steps["train"])
            self.writer.add_scalar("train_loss_smooth", losses["seg-smooth"][-1], self.global_steps["train"])
            self.writer.add_scalar("train_loss_all", losses["seg-all"][-1], self.global_steps["train"])
            self.global_steps["train"] += 1

        return losses

    @torch.no_grad()
    def eval(self, loader="eval"):
        """
        After training for an epoch, alternating optimization should give currently optimal normalizer,
        so a simple segmentation loss is sufficient.
        """
        assert loader in ["eval", "test"]
        self.normalizer.eval()
        self.u_net.eval()

        if loader == "eval":
            data_loader = self.eval_loader
        else:
            data_loader = self.test_loader
        if self.notebook:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        pbar = tqdm(enumerate(data_loader), desc=f"{loader}", total=len(data_loader), leave=False)
        num_samples = 0
        loss_acc = 0

        for i, (X, mask) in pbar:
            # ###########
            # # TODO: Comment out
            # if i >= 2:
            #     break
            # ##########
            X = (2 * X - 1).float().to(self.device)
            mask = mask.to(self.device)

            X_norm = self.normalizer(X)
            mask_pred = self.u_net(X_norm)
            loss = self.loss_fn(mask_pred, mask)
            loss_acc += loss.item() * X.shape[0]
            num_samples += X.shape[0]

            mask_pred_debug = self.u_net(X)
            loss_debug = self.loss_fn(mask_pred_debug, mask)
            print(f"inside .eval(): loss: {loss_debug}")

            pbar.set_description(f"{loader} batch {i + 1}/{len(data_loader)}: loss-all: {loss.item():.4f}")

        loss_acc /= num_samples
        self.writer.add_scalar("eval_loss_main", loss_acc, self.global_steps["eval"])
        self.global_steps["eval"] += 1

        return loss_acc

    def train(self, model_save_dir=None, if_plot=True):
        # time_stamp = self._create_save_folder(model_save_dir)
        self._create_save_folder(model_save_dir)
        lowest_loss_all = float("inf")
        if self.notebook:
            from tqdm.notebook import trange
        else:
            from tqdm import trange
        pbar = trange(self.epochs, desc="epoch")
        train_losses = {"norm": [], "seg-main": [], "seg-smooth": [], "seg-all": []}
        eval_losses = {"seg-all": []}
        for epoch in pbar:
            # TODO: uncomment
            train_loss = self._train()
            eval_loss = self.eval()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            for key in train_loss:
                train_losses[key].append(train_loss[key])
            eval_losses["seg-all"].append(eval_loss)

            if lowest_loss_all > eval_loss:
                lowest_loss_all = eval_loss
                self._save_latest_model(self.time_stamp, epoch, eval_loss, model_save_dir)

            if if_plot:
                self._eval_sample_plot(epoch)

            self._write_norm_params()

            self.global_steps["epoch"] += 1
            print(f"epoch {epoch + 1}/{self.epochs}: train loss: {train_loss['seg-main'][-1]:.4f}, "
                  f"eval loss: {eval_loss:.4f}")

        return train_losses, eval_losses

    def _eval_sample_plot(self, epoch, figsize=(9.6, 4.8)):
        normalizer = Normalizer(self.normalizer.num_layers, self.normalizer.kernel_size, self.normalizer.in_channels,
                                self.normalizer.intermediate_channels).to(self.device)
        normalizer.load_state_dict(self.normalizer.state_dict())
        norm_optimizer = torch.optim.Adam(normalizer.parameters())
        img_ind = np.random.randint(len(self.test_loader.dataset))
        X, mask_gt = self.test_loader.dataset[img_ind]  # (1, H, W), (1, H, W)
        test_time_adapter = TestTimeAdapter(normalizer, self.u_net, norm_optimizer, self.loss_fn,
                                            device=self.device, writer=self.writer, epoch=epoch, notebook=self.notebook)
        # (1, H, W) -> (1, 1, H, W)
        X = X.unsqueeze(0)
        mask_gt = mask_gt.unsqueeze(0)
        mask_pred, loss = test_time_adapter.predict(X, mask=mask_gt)  # (1, H, W) -> (1, 1, H, W)
        # print(mask_pred.shape)
        loss_gt = dice_loss(mask_to_one_hot(mask_pred.unsqueeze(0), self.num_classes), mask_gt,
                            if_soft_max=False)

        figsize = plt.rcParams["figure.figsize"] if figsize is None else figsize
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        imgs = [X.detach().cpu().numpy()[0, 0], mask_pred[0], mask_gt[0, 0]]
        for axis, img in zip(axes, imgs):
            handle = axis.imshow(img, cmap="gray")
            plt.colorbar(handle, ax=axis)
        plt.suptitle(f"loss: {loss:.4f}, loss_gt: {loss_gt.item():.4f}")
        fig.tight_layout()

        if self.notebook:
            plt.show()
        else:
            plt.savefig(os.path.join(self.img_save_dir, self.time_stamp, f"epoch_{epoch + 1}.png"))

        self.writer.add_figure("epoch_norm_check", fig, self.global_steps["epoch"])

    def _create_save_folder(self, model_save_dir=None):
        """
        This should be called at the root of the project. model_save_dir is the root folder:
        model_save_dir/
        -time_stamp_and_additional_info/
        --model.pt (to save in self._save_latest_model(.))
        """
        assert model_save_dir is not None, "please specify a save directory"
        if not os.path.isdir(model_save_dir):
            os.mkdir(model_save_dir)
        original_wd = os.getcwd()
        os.chdir(model_save_dir)
        # time_stamp = f"{time.time()}".replace(".", "_")
        # self.time_stamp = time_stamp
        os.mkdir(self.time_stamp)
        os.chdir(original_wd)

        if not self.notebook:
            if not os.path.isdir(self.img_save_dir):
                os.mkdir(self.img_save_dir)
            original_wd = os.getcwd()
            os.chdir(self.img_save_dir)
            os.mkdir(self.time_stamp)
            os.chdir(original_wd)

        # return time_stamp

    def _save_latest_model(self, time_stamp, epoch, eval_loss, model_save_dir=None):
        # eval_loss; dict
        assert model_save_dir is not None, "please specify a save directory"
        original_wd = os.getcwd()
        os.chdir(model_save_dir)
        os.chdir(time_stamp)
        # remove all old files
        for filename in os.listdir():
            if os.path.isfile(filename):
                os.remove(filename)
        # filename_common = f"{self.loss_type}_epoch_{epoch + 1}_eval_loss_{eval_loss:.4f}"
        filename_common = f"ce_{self.ce_weight}_dsc_{self.dsc_weight}_epoch_{epoch + 1}_eval_loss_{eval_loss:.4f}"
        filename_common = filename_common.replace(".", "_")
        filename_common += ".pt"
        torch.save(self.normalizer.state_dict(), "norm_" + filename_common)
        torch.save(self.u_net.state_dict(), "u_net_" + filename_common)
        os.chdir(original_wd)

    def _write_norm_params(self):
        for i, layer in enumerate(self.normalizer.conv_layers):
            self.writer.add_histogram(f"norm_param_{i}", layer.weight, self.global_steps["epoch"])


class TestTimeAdapter(object):
    def __init__(self, normalizer, u_net, norm_optimizer, loss_fn, device=None,
                 writer: SummaryWriter = None, epoch=None, notebook=False):
        # assert loss_type in ["CE", "DSC"], "only supports CE and DSC"
        self.normalizer = normalizer
        self.u_net = u_net
        self.norm_optimizer = norm_optimizer
        # self.loss_type = loss_type
        self.loss_fn = loss_fn
        self.device = torch.device("cuda") if device is None else device
        self.writer = writer
        self.epoch = epoch
        self.global_steps = 0
        self.notebook = notebook

    def predict(self, X, mask=None, rel_eps=0.05, max_iters=15):
        # X: (1, 1, H, W)
        X = (2 * X - 1).to(self.device)
        self.normalizer.train()
        self.u_net.eval()
        cur_loss = None
        next_loss = self._compute_loss(X)
        num_iters = 1

        if self.writer is not None:
            with torch.no_grad():
                self.writer.add_scalar(f"epoch_{self.epoch}_adapt", next_loss.item(), self.global_steps)
                fig = make_summary_plot(self.u_net, self.normalizer, None, if_save=False,
                                        X_in=X[0], mask_in=mask[0], device=self.device,
                                        figsize=(10.8, 7.2), fraction=0.5)
                self.writer.add_figure(f"epoch_{self.epoch}_adapt_fig", fig, self.global_steps)
                self.global_steps += 1
                if self.notebook:
                    make_summary_plot(self.u_net, self.normalizer, None, if_save=False,
                                      X_in=X[0], mask_in=mask[0], device=self.device,
                                      figsize=(10.8, 7.2), fraction=0.5, if_show=True)

        while (cur_loss is None) or abs((next_loss.item() - cur_loss.item()) / cur_loss.item()) > rel_eps:
            if num_iters > max_iters:
                break
            self.norm_optimizer.zero_grad()
            next_loss.backward()
            self.norm_optimizer.step()
            cur_loss = next_loss
            next_loss = self._compute_loss(X)
            num_iters += 1
            print(f"rel: {abs((next_loss.item() - cur_loss.item()) / cur_loss.item())}")

            if self.writer is not None:
                with torch.no_grad():
                    self.writer.add_scalar(f"epoch_{self.epoch}_adapt", next_loss.item(), self.global_steps)
                    fig = make_summary_plot(self.u_net, self.normalizer, None, if_save=False,
                                            X_in=X[0], mask_in=mask[0], device=self.device,
                                            figsize=(10.8, 7.2), fraction=0.5)
                    self.writer.add_figure(f"epoch_{self.epoch}_adapt_fig", fig, self.global_steps)
                    self.global_steps += 1

        self.normalizer.eval()
        X_norm = self.normalizer(X)
        mask_pred = self.u_net(X_norm).argmax(dim=1)

        # mask_pred: (1, K, H, W) -> (1, H, W)
        return mask_pred.detach().cpu(), cur_loss

    def _compute_loss(self, X):
        # X: (1, 1, H, W)
        X = (2 * X - 1).float().to(self.device)
        X_norm = self.normalizer(X)
        mask_pred = self.u_net(X)  # (B, K, H, W)
        mask_norm_pred = self.u_net(X_norm)
        loss = symmetric_loss(mask_pred, mask_norm_pred, self.loss_fn)

        return loss


class UNetTrainer(object):
    def __init__(self, train_loader, eval_loader, model, optimizer, scheduler=None, ce_weight=1, dsc_weight=0,
                 epochs=20, notebook=True, device=None, writer: SummaryWriter = None):
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.epochs = epochs
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ce_weight = ce_weight
        self.dsc_weight = dsc_weight
        self.loss_fn = lambda X, mask: ce_weight * cross_entropy_loss(X, mask) + dsc_weight * dice_loss(X, mask)
        self.notebook = notebook
        self.device = torch.device("cuda") if device is None else device
        self.writer = writer
        self.global_steps = {"train": 0, "eval": 0, "epoch": 0}

    def _train(self):
        self.model.train()
        if self.notebook:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="training", leave=False)
        losses = []
        for i, (X, _, mask) in pbar:
            # X: (B, 1, 256, 256)
            X = X.float().to(self.device)
            mask = mask.to(self.device)
            X = 2 * X - 1
            mask_pred = self.model(X)
            loss = self.loss_fn(mask_pred, mask)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
            self.writer.add_scalar("train_loss", losses[-1], self.global_steps["train"])
            self.global_steps["train"] += 1

            pbar.set_description(f"training batch {i + 1}/{len(self.train_loader)}: loss: {losses[-1]}")

        return losses

    @torch.no_grad()
    def _eval(self):
        self.model.eval()
        if self.notebook:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        pbar = tqdm(enumerate(self.eval_loader), total=len(self.eval_loader), desc="eval", leave=False)
        loss_acc = 0
        num_samples = 0
        for i, (X, mask) in pbar:
            # X: (B, 1, 256, 256)
            X = X.float().to(self.device)
            mask = mask.to(self.device)
            X = 2 * X - 1
            mask_pred = self.model(X)
            loss = self.loss_fn(mask_pred, mask)
            loss_acc += loss.item() * X.shape[0]
            num_samples += X.shape[0]

            pbar.set_description(f"eval batch {i + 1}/{len(self.eval_loader)}: loss: {loss.item()}")

        loss_acc /= num_samples
        self.writer.add_scalar("eval_loss", loss_acc, self.global_steps["eval"])
        self.global_steps["eval"] += 1

        return loss_acc

    def train(self):
        if self.notebook:
            from tqdm.notebook import trange
        else:
            from tqdm import trange
        pbar = trange(self.epochs, desc="epochs")

        train_losses, eval_losses = [], []
        for epoch in pbar:
            train_loss = self._train()
            eval_loss = self._eval()
            if self.scheduler is not None:
                self.scheduler.step()

            train_losses.append(train_loss)
            eval_losses.append(eval_loss)

            pbar.set_description(f"epoch {epoch + 1}/{self.epochs}: train loss {train_loss[-1]:.4f}, "
                                 f"eval loss: {eval_loss:.4f}")

            self.writer.add_scalar("epoch_train", train_loss[-1], self.global_steps["epoch"])
            self.writer.add_scalar("epoch_eval", eval_loss, self.global_steps["epoch"])

            self._eval_plot()

            self.global_steps["epoch"] += 1

        return train_losses, eval_losses

    @torch.no_grad()
    def _eval_plot(self):
        index = np.random.randint(len(self.eval_loader.dataset))
        X, mask = self.eval_loader.dataset[index]  # (1, 256, 256) each
        X = X.float().to(self.device)
        mask = mask.to(self.device)
        X = 2 * X - 1
        X = X.unsqueeze(0)  # (1, 1, 256, 256)
        mask = mask.unsqueeze(0)
        mask_pred = self.model(X)
        loss = self.loss_fn(mask_pred, mask)  # (1, K, 256, 256), (1, 1, 256, 256)
        mask_pred = mask_pred.argmax(dim=1)  # (1, K, 256, 256) -> (1, 256, 256)
        fig, axes = plt.subplots(1, 3, figsize=(10.8, 7.2))
        for img_iter, axis in zip([X[0, 0], mask_pred[0], mask[0, 0]], axes):
            handle = axis.imshow(img_iter.detach().cpu().numpy(), cmap="gray")
            plt.colorbar(handle, ax=axis)
        fig.suptitle(f"loss: {loss:.4f}")
        if self.notebook:
            plt.show()
        self.writer.add_figure("epoch_img", fig, self.global_steps["epoch"])


class TestTimeEvaluator(object):
    def __init__(self, test_loader, normalizer, u_net, norm_optimizer, loss_fn, writer: SummaryWriter,
                 device=None, notebook=False):
        self.test_loader = test_loader
        self.normalizer = normalizer
        self.u_net = u_net
        self.norm_optimizer = norm_optimizer
        self.loss_fn = loss_fn
        self.device = torch.device("cuda") if device is None else device
        self.writer = writer
        self.notebook = notebook
        self.global_step = 0
        self.index = np.random.randint(len(self.test_loader.dataset))

    def adapt(self, max_iter=15, rel_error=1e-5):
        # Store both values of self.loss_fn and DICE; randomly sample one image
        # TODO: Save scalars (decide which to save)
        self.u_net.eval()
        self.normalizer.train()
        # index = np.random.randint(len(self.test_loader.dataset))
        X, mask = self.test_loader.dataset[self.index]
        X = X.float().to(self.device)  # (1, H, W)
        X_orig = X
        X = 2 * X - 1
        mask = mask.to(self.device)  # (1, H, W)
        mask_orig = mask
        X = X.unsqueeze(0)  # (1, 1, H, W)
        mask = mask.unsqueeze(0)

        loss_adapt, loss_norm, loss_no_norm = self._compute_loss(X, mask)
        self.writer.add_scalar("loss_norm", loss_norm.item(), self.global_step)
        self.writer.add_scalar("loss_no_norm", loss_no_norm.item(), self.global_step)
        self.writer.add_scalar("loss_adapt", loss_adapt.item(), self.global_step)
        suptitle = f"loss_norm: {loss_norm.item():.4f}, loss_no_norm: {loss_no_norm.item():.4f}, " \
                   f"loss_adapt: {loss_adapt.item():.4f}"
        with torch.no_grad():
            fig = make_summary_plot(self.u_net, self.normalizer, self.test_loader, suptitle=suptitle, if_save=False,
                                    X_in=X_orig, mask_in=mask_orig, if_show=True, figsize=(10.8, 7.2),
                                    fraction=0.5)
            self.writer.add_figure("img_adaption", fig, self.global_step)
        self.global_step += 1

        cur_loss = None
        next_loss = loss_adapt
        rel = None

        counter = 0
        while (cur_loss is None) or rel >= rel_error:
            if counter > max_iter:
                break
            counter += 1
            self.norm_optimizer.zero_grad()
            next_loss.backward()
            self.norm_optimizer.step()

            loss_adapt, loss_norm, loss_no_norm = self._compute_loss(X, mask)
            self.writer.add_scalar("loss_norm", loss_norm.item(), self.global_step)
            self.writer.add_scalar("loss_no_norm", loss_no_norm.item(), self.global_step)
            self.writer.add_scalar("loss_adapt", loss_adapt.item(), self.global_step)
            suptitle = f"loss_norm: {loss_norm.item():.4f}, loss_no_norm: {loss_no_norm.item():.4f}, " \
                       f"loss_adapt: {loss_adapt.item():.4f}"
            with torch.no_grad():
                fig = make_summary_plot(self.u_net, self.normalizer, self.test_loader, suptitle=suptitle, if_save=False,
                                        X_in=X_orig, mask_in=mask_orig, if_show=True, figsize=(10.8, 7.2),
                                        fraction=0.5)
                self.writer.add_figure("img_adaption", fig, self.global_step)

            cur_loss = next_loss
            next_loss = loss_adapt

            rel = abs((next_loss.item() - cur_loss.item()) / cur_loss.item())
            self.writer.add_scalar("rel_error", rel, self.global_step)
            self.global_step += 1

    def _compute_loss(self, X, mask):
        # X, mask: (1, 1, H, W); X: (2 * X - 1).float().to(self.device)
        with torch.no_grad():
            mask_pred_no_norm = self.u_net(X)  # (1, 1, H, W)
            loss_no_norm = dice_loss(mask_pred_no_norm, mask)
            mask_pred_norm = self.u_net(self.normalizer(X))
            loss_norm = dice_loss(mask_pred_norm, mask)

        mask_pred_no_norm_adapt = self.u_net(X)
        mask_pred_norm_adapt = self.u_net(self.normalizer(X))
        loss_adapt = symmetric_loss(mask_pred_no_norm_adapt, mask_pred_norm_adapt, self.loss_fn)

        return loss_adapt, loss_norm, loss_no_norm

    @torch.no_grad()
    def eval(self, if_normalize=True):
        # Only DICE; given gt
        # Store 2 values in SummaryWriter with and without normalizer in wrapper func
        if self.notebook:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        pbar = tqdm(enumerate(self.test_loader), desc="Evaluation", total=len(self.test_loader))

        loss_avg = 0
        num_samples = 0
        for i, (X, mask) in pbar:
            X = X.float().to(self.device)  # (B, 1, H, W)
            X = 2 * X - 1
            mask = mask.to(self.device)  # (B, 1, H, W)

            if if_normalize:
                X = self.normalizer(X)

            mask_pred = self.u_net(X)  # (B, 1, H, W)
            loss = dice_loss(mask_pred, mask)
            loss_avg += loss.item() * X.shape[0]
            num_samples += X.shape[0]

            pbar.set_description(f"Batch {i + 1}/{len(self.test_loader)}: loss: {loss.item():.4f}")

        return loss_avg / num_samples
