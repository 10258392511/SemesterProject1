import os

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


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
