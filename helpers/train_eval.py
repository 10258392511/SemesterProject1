import os
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import time

from torch_lr_finder import LRFinder


def lr_search(model, criterion, optimizer, train_loader, end_lr=1, device=None):
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    lr_finder = LRFinder(model, optimizer, criterion, device)
    lr_finder.range_test(train_loader, end_lr=end_lr, num_iter=100)
    lr_finder.plot()
    lr_finder.reset()

    return lr_finder.history


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
