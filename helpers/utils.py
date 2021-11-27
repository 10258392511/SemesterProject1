import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from torchvision.utils import make_grid


def get_car_filenames(mode="train"):
    assert mode in ["train", "test"], "mode must be 'train' or 'test'"
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cur_path = os.getcwd()
    os.chdir(project_root)

    # root_dir = None
    if mode == "train":
        root_dir = os.path.join(project_root, "data/cars")
    else:
        root_dir = os.path.join(project_root, "data/cars/Test")
    base_paths = os.listdir(os.path.join(root_dir, "Input/"))
    image_full_paths = [os.path.join(root_dir, "Input/", base_path) for base_path in base_paths]
    mask_full_paths = [os.path.join(root_dir, "Target/", base_path) for base_path in base_paths]

    os.chdir(cur_path)
    return image_full_paths, mask_full_paths


def convert_mask(mask: np.ndarray):
    """
    Convert a uint8 mask with C semantic classes to 0, 1, ..., C - 1

    Parameters
    ----------
    mask: np.ndarray

    Returns
    -------
    np.ndarray
    """
    labels = np.sort(np.unique(mask), axis=-1)
    # check whether the labels already satisfy 0, 1, ..., C - 1
    need_to_convert = False
    for i, label in enumerate(labels):
        if i != label:
            need_to_convert = True
            break
    if not need_to_convert:
        return mask

    mask_out = np.zeros_like(mask, dtype=np.uint8)
    for i, label in enumerate(labels):
        mask_out[mask==label] = i

    return mask_out


def show_img_mask(img_path: str = None, mask_path: str = None, figsize=None, **kwargs):
    """
    kwargs: {"img": ..., "mask": ...}
    """
    if figsize is None:
        figsize = plt.rcParams.get("figure.figsize")

    if img_path is not None and mask_path is not None:
        img, mask = cv.imread(img_path, cv.IMREAD_COLOR), cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        mask_converted = convert_mask(mask)
        imgs = [img, mask, mask_converted]
    elif kwargs.get("img") is not None and kwargs.get("mask") is not None:
        img, mask = kwargs["img"], kwargs["mask"]
        imgs = [img, mask]
    else:
        raise ValueError("No image or mask input")

    fig, axes = plt.subplots(1, len(imgs), figsize=figsize)
    for axis, img in zip(axes.flatten(), imgs):
        if img.ndim == 3:
            axis.imshow(img)
        elif img.ndim == 2:
            handle = axis.imshow(img, cmap="gray")
            plt.colorbar(handle, ax=axis, fraction=0.05)
    fig.tight_layout()
    plt.show()


def compute_resize_shape_and_max_depth(in_height, in_width, min_feature_dim=16):
    height_bits, width_bits = int(np.ceil(np.log2(in_height))), int(np.ceil(np.log2(in_width + 1)))
    min_dist = float("inf")
    out = None
    for height_log in range(height_bits + 1):
        for width_log in range(width_bits + 1):
            height_resize, width_resize = 2 ** height_log, 2 ** width_log
            candidate = abs(in_height - height_resize) + abs(in_width - width_resize)
            if candidate <= min_dist:
                min_dist = candidate
                out = (height_resize, width_resize)

    return out[0], out[1], int(min(np.log2(out[0] / 16), np.log2(out[1] / 16)))


def plot_training_history(train_loss, eval_loss, **kwargs):
    # create x-ticks for "train_loss"
    train_loss_ticks = []
    for i, epoch_loss in enumerate(train_loss):
        train_loss_ticks += np.arange(i, i + 1, 1 / len(epoch_loss)).tolist()

    eval_loss_ticks = list(range(1, 1 + len(eval_loss)))
    figsize = kwargs.get("figsize", plt.rcParams["figure.figsize"])
    fig, axis = plt.subplots(figsize=figsize)
    axis.plot(train_loss_ticks, np.array(train_loss).flatten(), label="train")
    axis.plot(eval_loss_ticks, eval_loss, label="eval")
    axis.grid(True)
    axis.legend()
    axis.set_xlabel("epoch")
    axis.set_ylabel("loss")
    axis.set_xticks(eval_loss_ticks)
    axis.set_title("Training Curve")
    plt.show()


def plot_lr(learning_rates, **kwargs):
    train_loss_ticks = []
    for i, epoch_loss in enumerate(learning_rates):
        train_loss_ticks += np.arange(i, i + 1, 1 / len(epoch_loss)).tolist()

    eval_loss_ticks = list(range(1, 1 + len(learning_rates)))
    figsize = kwargs.get("figsize", plt.rcParams["figure.figsize"])
    fig, axis = plt.subplots(figsize=figsize)
    axis.plot(train_loss_ticks, np.array(learning_rates).flatten())
    axis.grid(True)
    axis.set_xlabel("epoch")
    axis.set_ylabel("learning rate")
    axis.set_xticks(eval_loss_ticks)
    axis.set_title("Learning Rate")
    plt.show()


def MADE_sample_plot(model, num_rows=5, num_cols=5):
    model.eval()
    batch_size = num_rows * num_cols
    samples = model.sample(batch_size)  # (B, H, W)
    samples = samples.unsqueeze(1)  # (B, 1, H, W)
    grid_tensor = make_grid(samples, num_rows)  # (1, H', W')
    fig, axis = plt.subplots()
    handle = axis.imshow(grid_tensor.permute(1, 2, 0).numpy(), cmap="gray")
    plt.colorbar(handle, ax=axis)
    plt.show()


@torch.no_grad()
def real_nvp_preprocess(x, dequantize=True, alpha=0.95, max_value=256, reverse=False):
    """
    x: (B, C, H, W)
    x -> logit(alpha + (1 - alpha) * x / max_value), possibly with dequantization.
    """
    if reverse:
        x1 = 1 / (1 + torch.exp(-x))
        x_out = (x1 - alpha) / (1 - alpha)

        # x_out is in [0, 1]
        return x_out, None

    x = x.float()
    # print(f"x: {x}")
    if dequantize:
        x = x + torch.rand_like(x)
        # print(f"dequantized: {x}")

    x1 = x / max_value * (1 - alpha) + alpha
    x_out = torch.log(x1) - torch.log(1 - x1)
    log_det = F.softplus(torch.exp(-x_out)) + F.softplus(torch.exp(x_out)) + torch.log(torch.tensor(1 - alpha))\
              - torch.log(torch.tensor(max_value))

    # x_out: (B, C, H, W), log_det: (B,)
    return x_out, log_det.sum(dim=[1, 2, 3])


@torch.no_grad()
def real_nvp_interpolation_grid(num_rows=5, num_cols=5):
    # TODO
    pass


def compute_derivatives(X, win_size=5):
    # X: (B, 1, H, W)
    sobel, _ = cv.getDerivKernels(1, 1, win_size)
    gaussian = cv.getGaussianKernel(win_size, -1).astype(np.float32)
    deriv_x = sobel * gaussian.T
    deriv_y = deriv_x.T
    deriv_x, deriv_y = torch.FloatTensor(deriv_x), torch.FloatTensor(deriv_y)
    Ix = F.conv2d(X, deriv_x.view(1, 1, *deriv_x.shape))  # (B, 1, H_valid, W_valid)
    Iy = F.conv2d(X, deriv_y.view(1, 1, * deriv_y.shape))

    return Ix, Iy


def warp_optical_flow(X, flow):
    # X: (B, C, H, W), flow: (B, H, W, 2)
    B, C, H, W = X.shape
    ### future version of PyTorch will make indexing="ij" default
    yy, xx = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))  # (H, W) each
    xx = xx.float().to(X.device)
    yy = yy.float().to(X.device)
    flow[..., 0] += (xx - W / 2) / W * 2
    flow[..., 1] += (yy - H / 2) / H * 2
    # print(flow)

    X_out = F.grid_sample(X, flow)

    return X_out
