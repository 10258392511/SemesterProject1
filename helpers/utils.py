import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import albumentations as A
import random

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


def compute_derivatives(X, win_size=5, if_cross_entropy=False):
    # X: (B, 1, H, W) or (B, K, H, W)
    sobel, _ = cv.getDerivKernels(1, 1, win_size)
    gaussian = cv.getGaussianKernel(win_size, -1).astype(np.float32)
    deriv_x = sobel * gaussian.T
    deriv_y = deriv_x.T
    deriv_x, deriv_y = torch.FloatTensor(deriv_x).to(X.device), torch.FloatTensor(deriv_y).to(X.device)
    if not if_cross_entropy:
        Ix = F.conv2d(X, deriv_x.view(1, 1, *deriv_x.shape))  # (B, 1, H_valid, W_valid)
        Iy = F.conv2d(X, deriv_y.view(1, 1, * deriv_y.shape))
    else:
        # (B, K, H, W) -> (B, 1, K, H, W)
        X = X.unsqueeze(1)
        # kernel: (1, 1, 1, H', W'), out: (B, 1, K, H_valid, W_valid) -> (B, K, H_valid, W_valid)
        Ix = F.conv3d(X, deriv_x.view(1, 1, 1, *deriv_x.shape)).squeeze()
        Iy = F.conv3d(X, deriv_y.view(1, 1, 1, *deriv_y.shape)).squeeze()

    return Ix, Iy


def warp_optical_flow(X, flow, if_normalize=True):
    # X: (B, C, H, W), flow: (B, H, W, 2)
    B, C, H, W = X.shape
    ### future version of PyTorch will make indexing="ij" default
    yy, xx = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))  # (H, W) each
    xx = xx.float().to(X.device)
    yy = yy.float().to(X.device)
    flow[..., 0] += (xx - W / 2) / W * 2
    flow[..., 1] += (yy - H / 2) / H * 2
    if if_normalize:
        flow = torch.tanh(flow)
    # print(flow)

    X_out = F.grid_sample(X, flow)

    return X_out


def get_transforms():
    options = [A.Flip(), A.ElasticTransform(alpha=50, alpha_affine=10)]

    return options


def get_separated_transforms(p=0.1):
    # p = 0.1
    options = [A.RandomCrop(224, 224, p=p), A.Flip(p=p), A.ElasticTransform(alpha=50, alpha_affine=0, p=p),
               A.Affine(scale=(0.8, 1.0), translate_percent=(0.0, 0.3), rotate=(-25, 25), shear=(-25, 25), p=p),
               A.RandomGamma((50, 150), p=p), A.GaussNoise(var_limit=(0, 0.3), mean=0, p=p),
               A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0, p=p)]

    return options


@torch.no_grad()
def make_summary_plot(u_net, normalizer, test_loader, image_save_path=None, suptitle="",
                      if_save=True, X_in=None, mask_in=None, if_show=False, **kwargs):
    u_net.eval()
    normalizer.eval()
    figsize = kwargs.get("figsize", plt.rcParams["figure.figsize"])
    fraction = kwargs.get("fraction", 1)
    device = kwargs.get("device", "cuda")
    device = torch.device(device)

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    if (X_in is None or mask_in is None) and test_loader is not None:
        ind = np.random.randint(len(test_loader.dataset))
        X, mask = test_loader.dataset[ind]  # (1, 256, 256), (1, 256, 256)
    else:
        X, mask = X_in, mask_in
    X = X.to(device)
    mask = mask.to(device)
    mask_direct_pred = u_net(2 * X.unsqueeze(0) - 1).argmax(dim=1)  # (1, 256, 256)
    X_norm = normalizer(2 * X.unsqueeze(0) - 1)  # (1, 1, 256, 256)
    mask_norm_pred = u_net(X_norm).argmax(dim=1)  # (1, 256, 256)
    img_grid = [[X.cpu().numpy()[0], mask.cpu().numpy()[0], mask_direct_pred.cpu().numpy()[0]],
                [(X_norm.cpu().numpy()[0, 0] + 1) / 2, ((X_norm[0, 0] + 1) / 2 - X[0]).cpu().numpy(),
                mask_norm_pred.cpu().numpy()[0]]]
    title_grid = [["original", "gt", "direct seg"],
                  ["normed", "normed diff", "normed seg"]]
    for i in range(2):
        for j in range(3):
            handle = axes[i, j].imshow(img_grid[i][j], cmap="gray")
            plt.colorbar(handle, ax=axes[i, j], fraction=fraction)
            axes[i, j].set_title(title_grid[i][j])
    fig.suptitle(suptitle)
    fig.tight_layout()

    if if_save:
        assert image_save_path is not None, "please specify a saving path"
        plt.savefig(image_save_path)

    if if_show:
        plt.show()

    return fig


def normalize(img, norm_type="div_by_max", eps=1e-8):
    # img: (H, W)
    assert norm_type in ["zero_mean", "div_by_max"], "invalid mode"
    if norm_type == "zero_mean":
        img = (img - np.mean(img)) / (np.std(img) + eps)
    else:
        perc1 = np.percentile(img, 1)
        perc99 = np.percentile(img, 99)
        img = (img - perc1) / (perc99 - perc1 + eps)
        img = np.clip(img, 0, 1)

    return img


def random_gamma_transform(X, gamma_min=0.5, gamma_max=2):
    # X: (B, 1, H, W), [0, 1], after sent to gpu
    X_cpu = X.detach().cpu()
    X_out = torch.empty_like(X_cpu)
    for i in range(X.shape[0]):
        img = X_cpu[i, 0, ...].numpy()  # (H, W)
        gamma = np.round(np.random.uniform(gamma_min, gamma_max), 2)
        img_out = normalize(img ** gamma)
        X_out[i, 0] = torch.FloatTensor(img_out)

    return X_out.to(X.device)


def normalize_tensor(X):
    # X: (B, 1, H, W)
    B, C, H, W = X.shape
    X = X.view(B, C, -1)
    X_min, _ = torch.min(X, dim=-1, keepdim=True)  # (B, 1, 1)
    X_max, _ = torch.max(X, dim=-1, keepdim=True)  # (B, 1, 1)
    X = (X - X_min) / (X_max - X_min)
    X = X.view(B, C, H, W)

    return X


def random_contrast_transform(X, gamma_min=0.5, gamma_max=2, noise_min=0, noise_max=0.1,
                              brightness_min=-1, brightness_max=1):
    # X: (B, 1, H, W)
    B = X.shape[0]
    gamma = torch.rand((B, 1, 1, 1)) * (gamma_max - gamma_min) + gamma_min  # (B, 1, 1, 1)
    gamma = gamma.to(X.device)
    X = normalize_tensor(X ** gamma)

    noise = torch.randn_like(X) * noise_max + noise_min
    X = normalize_tensor(X + noise)

    brightness = torch.rand((B, 1, 1, 1)) * (brightness_max - brightness_min) + brightness_min  # (B, 1, 1, 1)
    brightness = brightness.to(X.device)
    X = normalize_tensor(X + brightness)

    return X


def sample_from_loader(loader):
    X, mask = None, None
    for X, mask in loader:
        break

    return X, mask


# @torch.no_grad()
# def augmentation_by_normalizer(X, normalizer_list, device=None):
#     # X: (B, 1, H, W), [0, 1]; normalizer: bottleneck, default: k = 5
#     device = torch.device("cuda") if device is None else device
#     normalizer = normalizer_list[np.random.randint(0, len(normalizer_list))]
#     print(normalizer.kernel_size)
#     X = 2 * X.to(device) - 1
#     normalizer.eval()
#     X_norm = normalizer(X)
#     X_aug = normalizer(X) - X
#     p = (torch.rand((X.shape[0], 1, 1, 1)).to(device) >= 0.5)
#     p = 2 * p - 1
#     # print(p)
#     X_aug *= p
#     X_norm *= p
#
#     return X_norm, X_aug


@torch.no_grad()
def augmentation_by_normalizer(X, normalizer_list, device=None, if_debug=False):
    # X: (B, 1, H, W), [0, 1]; normalizer: bottleneck, default: k = 5
    device = torch.device("cuda") if device is None else device
    ind1, ind2 = np.random.choice(len(normalizer_list), (2,), replace=False)
    normalizer1, normalizer2 = normalizer_list[ind1], normalizer_list[ind2]
    # print(f"{normalizer1.kernel_size}, {normalizer2.kernel_size}")
    X = 2 * X.to(device) - 1
    normalizer1.eval()
    normalizer2.eval()
    X_norm1 = normalizer1(X)
    X_norm2 = normalizer2(X)
    X_diff = X_norm1 - X_norm2
    p = (torch.rand((X.shape[0], 1, 1, 1)).to(device) >= 0.5)
    p = 2 * p - 1
    X_aug = X + X_diff
    # print(p)
    X_aug *= p
    X_aug = normalize_tensor(X_aug)

    if if_debug:
        return X_diff, X_aug

    return X_aug
