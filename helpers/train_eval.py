import os
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import time

from torchvision.utils import make_grid
from torch_lr_finder import LRFinder
from .utils import real_nvp_preprocess, compute_derivatives, warp_optical_flow


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
            X = X.float().to(self.device) * 255  # [0, 255]
            X, log_det = real_nvp_preprocess(X)  # (B, C, H, W), (B,)
            log_prob_model = self.model.log_prob(X)
            loss = -(log_prob_model + log_det)
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
            X = X.float().to(self.device) * 255  # [0, 255]
            X, log_det = real_nvp_preprocess(X)  # (B, C, H, W), (B,)
            log_prob_model = self.model.log_prob(X)
            loss = -(log_prob_model + log_det)
            loss_avg += loss * X.shape[0]
            num_samples += X.shape[0]
            pbar.set_description(f"batch {i + 1}/{len(self.eval_loader)}: eval loss: {loss.item:.4f}")

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
            # TODO
            self._eval_sample_plot()

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

    def _eval_sample_plot(self):
        # TODO: implementation
        pass


######################################################################
## MNISTVAETrainer ##
class MNISTVAETrainer(object):
    def __init__(self, vae, int_warper, shape_warper, train_loader, eval_loader, vae_optimizer,
                 int_warper_optimizer, shape_warper_optimizer, lr_scheduler=None,
                 epochs=20, warper_batch_portion=0.5, deriv_win_size=5, lamda_int_warper=1, lamda_shape_warper=1,
                 lamda_warp_recons=1, device=None, notebook=True):
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
        # mse_loss = nn.MSELoss()
        loss_type = "MSE"
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
            log_s_1, t_1 = torch.chunk(self.int_warper(X_int_interp_shape_1, X1), 2, dim=1)  # (B, 1, H, W) each
            X1_hat = X_int_interp_shape_1 * log_s_1.exp() + t_1  ### intensity warper: per pixel on all channels
            log_s_2, t_2 = torch.chunk(self.int_warper(X_int_interp_shape_2, X2), 2, dim=1)
            X2_hat = X_int_interp_shape_2 * log_s_2.exp() + t_2
            # loss_1 = mse_loss(X1_hat, X1)
            # loss_2 = mse_loss(X2_hat, X2)
            loss_1 = self._compare_two_imgs(X1_hat, X1, loss_type)
            loss_2 = self._compare_two_imgs(X2_hat, X2, loss_type)
            loss_int_warper = (loss_1 + loss_2) / 2
            loss_int_warper_smooth = 0
            for img in [log_s_1, t_1, log_s_2, t_2]:
                Ix, Iy = compute_derivatives(img, win_size=self.deriv_win_size)  # (B, 1, H, W) each
                loss_int_warper_smooth += torch.abs(Ix).sum(dim=1).mean() + torch.abs(Iy).sum(dim=1).mean()
            loss_int_warper_smooth /= 2
            loss_int_all = loss_int_warper + self.lamda_int_warper * loss_int_warper_smooth
            self.int_warper_optimizer.zero_grad()
            loss_int_all.backward(retain_graph=True)
            self.int_warper_optimizer.step()
            losses["int_warper"].append(loss_int_warper.item())
            losses["int_warper_smooth"].append(loss_int_warper_smooth.item())

            # shape warper
            uv_1 = self.shape_warper(X_int_1_shape_interp, X1)  # (B, 2, H, W)
            X1_hat = warp_optical_flow(X_int_1_shape_interp, uv_1.permute(0, 2, 3, 1))
            uv_2 = self.shape_warper(X_int_2_shape_interp, X2)  # (B, 2, H, W)
            X2_hat = warp_optical_flow(X_int_2_shape_interp, uv_2.permute(0, 2, 3, 1))
            # loss_1 = mse_loss(X1_hat, X1)
            # loss_2 = mse_loss(X2_hat, X2)
            loss_1 = self._compare_two_imgs(X1_hat, X1, loss_type)
            loss_2 = self._compare_two_imgs(X2_hat, X2, loss_type)
            loss_shape_warper = (loss_1 + loss_2) / 2
            loss_shape_warper_smooth = 0
            for img in [*uv_1.chunk(2, dim=1), *uv_2.chunk(2, dim=1)]:
                Ix, Iy = compute_derivatives(img, win_size=self.deriv_win_size)  # (B, 1, H, W) each
                loss_shape_warper_smooth += torch.abs(Ix).sum(dim=1).mean() + torch.abs(Iy).sum(dim=1).mean()
            loss_shape_warper_smooth /= 2
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
            # loss_recons = (mse_loss(X1_hat, X1) + mse_loss(X2_hat, X2)) / 2
            loss_recons = (self._compare_two_imgs(X1_hat, X1, loss_type) +
                           self._compare_two_imgs(X2_hat, X2, loss_type)) / 2
            loss_kl = (-Z1_log_sigma - Z2_log_sigma - 0.5 * 2 +
                       0.5 * ((2 * Z1_log_sigma).exp() + (2 * Z2_log_sigma).exp() + Z1_mu ** 2) + Z2_mu ** 2) / 2
            loss_kl = loss_kl.sum(dim=1).mean()  # (B, lam_dim) -> (B,) -> float

            # int_warper loss
            log_s_1, t_1 = torch.chunk(self.int_warper(X_int_interp_shape_1, X1), 2, dim=1)  # (B, 1, H, W) each
            X1_hat = X_int_interp_shape_1 * log_s_1.exp() + t_1  ### intensity warper: per pixel on all channels
            log_s_2, t_2 = torch.chunk(self.int_warper(X_int_interp_shape_2, X2), 2, dim=1)
            X2_hat = X_int_interp_shape_2 * log_s_2.exp() + t_2
            # loss_1 = mse_loss(X1_hat, X1)
            # loss_2 = mse_loss(X2_hat, X2)
            loss_1 = self._compare_two_imgs(X1_hat, X1, loss_type)
            loss_2 = self._compare_two_imgs(X2_hat, X2, loss_type)
            loss_int_warper = (loss_1 + loss_2) / 2
            # shape_warper loss
            uv_1 = self.shape_warper(X_int_1_shape_interp, X1)  # (B, 2, H, W)
            X1_hat = warp_optical_flow(X_int_1_shape_interp, uv_1.permute(0, 2, 3, 1))
            uv_2 = self.shape_warper(X_int_2_shape_interp, X2)  # (B, 2, H, W)
            X2_hat = warp_optical_flow(X_int_2_shape_interp, uv_2.permute(0, 2, 3, 1))
            # loss_1 = mse_loss(X1_hat, X1)
            # loss_2 = mse_loss(X2_hat, X2)
            loss_1 = self._compare_two_imgs(X1_hat, X1, loss_type)
            loss_2 = self._compare_two_imgs(X2_hat, X2, loss_type)
            loss_shape_warper = (loss_1 + loss_2) / 2

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

        # mse_loss = nn.MSELoss()
        loss_type = "MSE"
        pbar = tqdm(enumerate(self.eval_loader), total=len(self.eval_loader), desc="validation", leave=False)
        num_samples = 0
        losses = dict(recons=0, kl=0, int_warper_vae=0, shape_warper_vae=0)
        # train VAE
        for i, (X1, X2) in pbar:
            X1, X2 = X1.float().to(self.device), X2.float().to(self.device)
            Z1_mu, Z1_log_sigma, Z2_mu, Z2_log_sigma, \
            X1_hat, X2_hat, X_int_interp_shape_1, X_int_interp_shape_2, X_int_1_shape_interp, X_int_2_shape_interp = \
                self.vae(X1, X2)  # (B, lat_dim) for Z, (B, C, H, W) for X
            loss_recons = (self._compare_two_imgs(X1_hat, X1, loss_type) +
                           self._compare_two_imgs(X2_hat, X2, loss_type)) / 2
            loss_kl = (-Z1_log_sigma - Z2_log_sigma - 0.5 * 2 +
                       0.5 * ((2 * Z1_log_sigma).exp() + (2 * Z2_log_sigma).exp() + Z1_mu ** 2) + Z2_mu ** 2) / 2
            loss_kl = loss_kl.sum(dim=1).mean()  # (B, lam_dim) -> (B,) -> float

            # int_warper loss
            log_s_1, t_1 = torch.chunk(self.int_warper(X_int_interp_shape_1, X1), 2, dim=1)  # (B, 1, H, W) each
            X1_hat = X_int_interp_shape_1 * log_s_1.exp() + t_1  ### intensity warper: per pixel on all channels
            log_s_2, t_2 = torch.chunk(self.int_warper(X_int_interp_shape_2, X2), 2, dim=1)
            X2_hat = X_int_interp_shape_2 * log_s_2.exp() + t_2
            # loss_1 = mse_loss(X1_hat, X1)
            # loss_2 = mse_loss(X2_hat, X2)
            loss_1 = self._compare_two_imgs(X1_hat, X1, loss_type)
            loss_2 = self._compare_two_imgs(X2_hat, X2, loss_type)
            loss_int_warper = (loss_1 + loss_2) / 2

            # shape_warper loss
            uv_1 = self.shape_warper(X_int_1_shape_interp, X1)  # (B, 2, H, W)
            X1_hat = warp_optical_flow(X_int_1_shape_interp, uv_1.permute(0, 2, 3, 1))
            uv_2 = self.shape_warper(X_int_2_shape_interp, X2)  # (B, 2, H, W)
            X2_hat = warp_optical_flow(X_int_2_shape_interp, uv_2.permute(0, 2, 3, 1))
            # loss_1 = mse_loss(X1_hat, X1)
            # loss_2 = mse_loss(X2_hat, X2)
            loss_1 = self._compare_two_imgs(X1_hat, X1, loss_type)
            loss_2 = self._compare_two_imgs(X2_hat, X2, loss_type)
            loss_shape_warper = (loss_1 + loss_2) / 2

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
            loss_mse = loss(X_hat, X).view(X.shape[0], -1)
            return loss_mse.sum(dim=1).mean()
        else:
            # TODO: multi-channel output for CE
            loss = nn.CrossEntropyLoss()
            X = (X * 255).long()
            return loss(X_hat, X)

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
        filename_common = f"epoch_{epoch + 1}"
        for key in eval_loss:
            filename_common += f"_{key}_{eval_loss[key]:.4f}"
        filename_common = filename_common.replace(".", "_")
        filename_common += ".pt"
        torch.save(self.vae, "vae_" + filename_common)
        torch.save(self.int_warper, "int_warper_" + filename_common)
        torch.save(self.shape_warper, "shape_warper_" + filename_common)
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

        log_s_1, t_1 = torch.chunk(self.int_warper(X_int_interp_shape_1, X1), 2, dim=1)  # (B, 1, H, W) each
        X1_hat_int = X_int_interp_shape_1 * log_s_1.exp() + t_1  ### intensity warper: per pixel on all channels
        log_s_2, t_2 = torch.chunk(self.int_warper(X_int_interp_shape_2, X2), 2, dim=1)
        X2_hat_int = X_int_interp_shape_2 * log_s_2.exp() + t_2

        uv_1 = self.shape_warper(X_int_1_shape_interp, X1)  # (B, 2, H, W)
        X1_hat_shape = warp_optical_flow(X_int_1_shape_interp, uv_1.permute(0, 2, 3, 1))
        uv_2 = self.shape_warper(X_int_2_shape_interp, X2)  # (B, 2, H, W)
        X2_hat_shape = warp_optical_flow(X_int_2_shape_interp, uv_2.permute(0, 2, 3, 1))

        img_grid = torch.cat([img1, img2, X1, X2, X_int_interp_shape_1, X_int_interp_shape_2,
                              X_int_1_shape_interp, X_int_2_shape_interp, X1_hat_int, X2_hat_int,
                              X1_hat_shape, X2_hat_shape], dim=0)
        warper_grid = torch.cat([log_s_1, t_1, log_s_2, t_2, *uv_1.chunk(2, dim=1), *uv_2.chunk(2, dim=1)], dim=0)
        img_grid = make_grid(img_grid, nrow=2)
        warper_grid = make_grid(warper_grid, nrow=2)
        plt.imshow(img_grid.permute(1, 2, 0).detach().cpu().numpy())
        plt.show()
        plt.imshow(warper_grid[0].detach().cpu().numpy(), cmap="gray")  # (1, H', W')
        plt.colorbar()
        plt.show()
