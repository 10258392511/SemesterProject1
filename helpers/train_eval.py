import os
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import time

from torch_lr_finder import LRFinder
from .utils import real_nvp_preprocess


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
        pass
