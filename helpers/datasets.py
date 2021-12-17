import numpy as np
import cv2 as cv
import torch
import matplotlib.pyplot as plt
import csv
import nibabel as nib
import albumentations as A
import os
import h5py

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Resize
from albumentations import BasicTransform, Compose
from .utils import convert_mask, normalize


class CarDataset(Dataset):
    def __init__(self, images: list, masks: list, dataset_expand_ratio = 1, transforms: list = None):
        assert len(images) == len(masks), "number of images must be equal to that of masks"
        if transforms is not None:
            for transform in transforms:
                assert isinstance(transform, BasicTransform), "please use 'albumentations' transforms"
        assert isinstance(dataset_expand_ratio, int) and dataset_expand_ratio >= 1, \
            "can only expand the dataset by an integer ratio"
        super(CarDataset, self).__init__()
        self.images = images  # list[filename]
        self.masks = masks
        self.dataset_expand_ratio = dataset_expand_ratio
        self.transforms = transforms

    def __len__(self):
        return int(len(self.images) * self.dataset_expand_ratio)

    def __getitem__(self, index):
        if self.dataset_expand_ratio != 1:
            index = index % len(self.images)
        image_path, mask_path = self.images[index], self.masks[index]
        image, mask = cv.imread(image_path, cv.IMREAD_COLOR), cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        mask = convert_mask(mask)
        if self.transforms is not None:
            transformer = Compose(self.transforms)
            transformed = transformer(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]
        image, mask = torch.FloatTensor(image / 255.), torch.LongTensor(convert_mask(mask))
        # print(f"image: {image.shape}, mask: {mask.shape}")

        # (H, W, C) -> (C, H, W)
        return image.permute(2, 0, 1), mask


class MNISTForMADE(Dataset):
    def __init__(self, data_tensor, transform=None):
        super(Dataset, self).__init__()
        self.dataset = data_tensor
        self.transform = transform

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, index):
        img = self.dataset[index, ...].unsqueeze(0).float() / 255  # (28, 28) -> (1, 28, 28)
        label = (img >= 0.5)
        # label = self.dataset[index, ...].unsqueeze(0).float() / 255.
        # img = label
        # print(f"inside MNITforMADE: {img.max()}")
        if self.transform is not None:
            img = self.transform(img)
        return 2 * img - 1, label.long()


class ColorMNIST(Dataset):
    def __init__(self, root=None, train=True, transform=None, class_label=9):
        assert root is not None, "Please specify where to store MNIST"
        assert class_label in list(range(10)), "class_label should be 0 to 9"
        super(ColorMNIST, self).__init__()
        if transform is None:
            transform = ToTensor()
        self.dataset = MNIST(root=root, train=train, transform=transform, download=False)
        self.y = class_label
        X, y = self.dataset.data, self.dataset.targets
        self.X = X[y==class_label] / 255.

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        num_samples = self.X.shape[0]
        img1 = self.X[index]  # (28, 28)
        another_ind = np.random.randint(0, num_samples)
        while another_ind == index:
            another_ind = np.random.randint(0, num_samples)
        img2 = self.X[another_ind]
        # add background
        img1_out = torch.zeros((3, *img1.shape))
        img2_out = torch.zeros((3, *img1.shape))
        rgb1, rgb2 = torch.rand((3, 1, 1)), torch.rand((3, 1, 1))
        img1_out = torch.clip(img1_out + img1.unsqueeze(0) + rgb1, 0, 1)
        img2_out = torch.clip(img2_out + img2.unsqueeze(0) + rgb2, 0, 1)

        return img1_out, img2_out

    def plot_pair(self, img1, img2):
        # img1, img2: (3, 28, 28)
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(img1.permute(1, 2, 0))
        axes[1].imshow(img2.permute(1, 2, 0))
        plt.show()


class MnMsDataset(Dataset):
    def __init__(self, source_name, data_path_root, transforms: list, mode,
                 gamma_limit=(50, 150), target_size=(256, 256)):
        assert mode in ["train", "eval", "test"], "mode should be 'train', 'eval' or 'test'"
        assert source_name.lower() in ["csf", "hvhd", "uhe"], "invalid source name"
        super(MnMsDataset, self).__init__()
        self.source_name = source_name
        self.data_path_root = data_path_root
        self.csv_path = os.path.join(data_path_root, f"mnms_{source_name.lower()}.csv")
        self.patient_info = {}
        self.patient_id = []
        self.mode = mode
        self.transforms = transforms  # TODO: experiment on different combination of augmentations
        self.split_info = {"csf": {"train": 31, "eval": 41},
                           "hvhd": {"train": 56, "eval": 66},
                           "uhe": {"train": 11, "eval": 16}}
        self.gamma_limit = gamma_limit
        self.target_size = target_size
        self._read_patient_info()
        self.num_samples = []
        # self.num_samples = -1
        # self.img_paths = []
        # self.mask_paths = []
        self._compute_num_samples()
        # self._get_img_mask_paths()

    def _read_patient_info(self):
        with open(self.csv_path, "r", newline="") as rf:
            csv_reader = csv.reader(rf, delimiter=",", quotechar="|")
            for i, row in enumerate(csv_reader):
                if self.mode == "train":
                    if i <= self.split_info[self.source_name]["train"]:
                        self.patient_info[row[0]] = {"ED": int(row[1]), "ES": int(row[2])}
                    else:
                        break

                elif self.mode == "eval":
                    if self.split_info[self.source_name]["train"] < i <= self.split_info[self.source_name]["eval"]:
                        self.patient_info[row[0]] = {"ED": int(row[1]), "ES": int(row[2])}
                    elif i > self.split_info[self.source_name]["eval"]:
                        break

                else:
                    if i > self.split_info[self.source_name]["eval"]:
                        self.patient_info[row[0]] = {"ED": int(row[1]), "ES": int(row[2])}

                self.patient_id = list(self.patient_info.keys())

    def _compute_num_samples(self):
        # read in all data and document filename
        for patient_id in self.patient_info:
            # directory = os.path.join(self.data_path_root, f"Labeled/{patient_id}")
            # img_path = os.path.join(directory, f"{patient_id}_sa.nii.gz")
            directory = os.path.join(self.data_path_root, f"images_corrected")
            img_path = os.path.join(directory, f"ed_{patient_id}_sa.nii.gz")
            # mask_path = os.path.join(directory, f"{patient_id}_sa_gt.nii.gz")
            img_data = nib.load(img_path)
            # mask_data = nib.load(mask_path)
            img_data_array = img_data.get_fdata()
            # mask_data_array = mask_data.get_fdata()
            # H, W, D, T = img_data.shape
            H, W, D = img_data.shape
            self.num_samples.append(2 * D)

    # def _get_img_mask_paths(self):
    #     base_dir = os.path.join(self.data_path_root, "images_corrected")
    #     for patient_id in self.patient_id:
    #         self.img_paths += [os.path.join(base_dir, f"es_{patient_id}_sa.nii.gz"),
    #                            os.path.join(base_dir, f"ed_{patient_id}_sa.nii.gz")]
    #         self.mask_paths += [os.path.join(base_dir, f"es_{patient_id}_sa_gt.nii.gz"),
    #                            os.path.join(base_dir, f"ed_{patient_id}_sa_gt.nii.gz")]

    def __len__(self):
        return sum(self.num_samples)
        # return self.num_samples
        # return len(self.patient_id) * 2

    def __getitem__(self, index):
        assert 0 <= index < self.__len__(), "invalid index"
        # img, mask = self.eval_test_array["image"][index, ...], self.eval_test_array["mask"][index, ...]
        counter = index
        img, mask, t, img_data_array, mask_data_array = None, None, None, None, None
        for i, patient_id in enumerate(self.patient_id):
            num_samples = self.num_samples[i]
            # if counter < num_samples:
            #     directory = os.path.join(self.data_path_root, f"Labeled/{patient_id}")
            #     img_path = os.path.join(directory, f"{patient_id}_sa.nii.gz")
            #     mask_path = os.path.join(directory, f"{patient_id}_sa_gt.nii.gz")
            #     img_data = nib.load(img_path)
            #     mask_data = nib.load(mask_path)
            #     img_data_array = img_data.get_fdata()
            #     mask_data_array = mask_data.get_fdata()
            #     H, W, D, T = img_data_array.shape
            #     img_data_array = (img_data_array / 255.).astype(np.float32)
            #     ed, es = self.patient_info[patient_id]["ED"], self.patient_info[patient_id]["ES"]
            #     if counter < D:
            #         img, mask = img_data_array[..., counter, ed], mask_data_array[..., counter, ed]
            #         t = ed
            #     else:
            #         counter -= D
            #         img, mask = img_data_array[..., counter - D, es], mask_data_array[..., counter, es]
            #         t = es
            #     break

            if counter < num_samples:
                if counter < num_samples // 2:
                    img_path = os.path.join(self.data_path_root, "images_corrected", f"ed_{patient_id}_sa.nii.gz")
                    mask_path = os.path.join(self.data_path_root, "images_corrected", f"ed_{patient_id}_sa_gt.nii.gz")
                else:
                    img_path = os.path.join(self.data_path_root, "images_corrected", f"es_{patient_id}_sa.nii.gz")
                    mask_path = os.path.join(self.data_path_root, "images_corrected", f"es_{patient_id}_sa_gt.nii.gz")
                    counter -= num_samples // 2
                break
            counter -= num_samples

        # print(f"img_paths: {len(self.img_paths)}, mask_paths: {len(self.mask_paths)}")
        # img_path, mask_path = self.img_paths[index], self.mask_paths[index]
        img_data = nib.load(img_path)
        img = img_data.get_fdata()[..., counter]
        mask_data = nib.load(mask_path)
        mask = mask_data.get_fdata()[..., counter]

        # Normalize img
        img = normalize(img, norm_type="div_by_max")

        if self.mode == "train":
            # H, W, D, T = img_data_array.shape
            # another_t = np.random.randint(T)
            # while another_t == t:
            #     another_t = np.random.randint(T)
            # img1 = img_data_array[..., counter, another_t]
            # # self.plot_triple(img, img1, mask)
            # transform = A.Compose(self.transforms, additional_targets={"image1": "image"})
            # transformed = transform(image=img, mask=mask, image1=img1)
            # img, mask, img1 = transformed["image"], transformed["mask"], transformed["image1"]
            # self.plot_triple(img, img1, mask)
            transform = A.Compose(self.transforms)
            transformed = transform(image=img, mask=mask)
            img, mask = transformed["image"], transformed["mask"]

            gamma_transform = A.RandomGamma(gamma_limit=self.gamma_limit, always_apply=True)
            # img1 = gamma_transform(image=img1)["image"]
            img1 = gamma_transform(image=img)["image"]

            cropper = A.Compose([A.RandomResizedCrop(height=self.target_size[0],
                                                     width=self.target_size[1], always_apply=True)],
                                additional_targets={"image1": "image"})
            transformed = cropper(image=img, mask=mask, image1=img1)
            img, mask, img1 = transformed["image"], transformed["mask"], transformed["image1"]

            # (1, H, W), (1, H, W), (1, H, W)
            return torch.FloatTensor(img).unsqueeze(0), torch.FloatTensor(img1).unsqueeze(0), \
                           torch.LongTensor(mask).unsqueeze(0)

        else:
            resizer = A.Resize(*self.target_size, always_apply=True)
            transformed = resizer(image=img, mask=mask)
            img, mask = transformed["image"], transformed["mask"]

            return torch.FloatTensor(img).unsqueeze(0), torch.LongTensor(mask).unsqueeze(0)

    def plot_triple(self, img, img1, mask, figsize=None):
        figsize = plt.rcParams["figure.figsize"] if figsize is None else figsize
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        imgs = [img, img1, mask]
        for i, img_iter in enumerate(imgs):
            handle = axes[i].imshow(img_iter, cmap="gray")
            plt.colorbar(handle, ax=axes[i], fraction=0.1)
        fig.tight_layout()
        plt.show()


class MnMsHDF5Dataset(Dataset):
    """
    File hierarchy:

    root
    -csf
    --train
    ---img0
    ---mask0
    --eval
    --test
    ...
    """
    def __init__(self, data_path, source_name, mode, transforms: list,
                 gamma_limit=(50, 150), target_size=(256, 256)):
        super(MnMsHDF5Dataset, self).__init__()
        assert ".h5" in data_path, "data_path must be a hdf5 file"
        assert source_name in ["csf", "hvhd", "uhe"], "invalid source name"
        assert mode in ["train", "eval", "test"], "invalid mode"

        self.data_path = data_path
        self.source_name = source_name
        self.mode = mode
        self.transforms = transforms
        self.gamma_limit = gamma_limit
        self.target_size = target_size

    def __len__(self):
        with h5py.File(self.data_path, "r") as hdf:
            group_source = hdf.get(self.source_name)
            group_data_mode = group_source.get(self.mode)

            return group_data_mode.attrs["SIZE"]

    def __getitem__(self, index):
        assert 0 <= index < self.__len__()
        with h5py.File(self.data_path, "r") as hdf:
            group_source = hdf.get(self.source_name)
            group_data_mode = group_source.get(self.mode)

            img, mask = np.array(group_data_mode.get(f"img{index}")), np.array(group_data_mode.get(f"mask{index}"))

        # Normalize img
        img = normalize(img, norm_type="div_by_max")

        if self.mode == "train":
            transform = A.Compose(self.transforms)
            transformed = transform(image=img, mask=mask)
            img, mask = transformed["image"], transformed["mask"]

            gamma_transform = A.RandomGamma(gamma_limit=self.gamma_limit, always_apply=True)
            img1 = gamma_transform(image=img)["image"]

            cropper = A.Compose([A.RandomResizedCrop(height=self.target_size[0],
                                                     width=self.target_size[1], always_apply=True)],
                                additional_targets={"image1": "image"})
            transformed = cropper(image=img, mask=mask, image1=img1)
            img, mask, img1 = transformed["image"], transformed["mask"], transformed["image1"]

            # (1, H, W), (1, H, W), (1, H, W)
            return torch.FloatTensor(img).unsqueeze(0), torch.FloatTensor(img1).unsqueeze(0), \
                   torch.LongTensor(mask).unsqueeze(0)

        else:
            resizer = A.Resize(*self.target_size, always_apply=True)
            transformed = resizer(image=img, mask=mask)
            img, mask = transformed["image"], transformed["mask"]

            return torch.FloatTensor(img).unsqueeze(0), torch.LongTensor(mask).unsqueeze(0)

    def plot_triple(self, img, img1, mask, figsize=None):
        figsize = plt.rcParams["figure.figsize"] if figsize is None else figsize
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        imgs = [img, img1, mask]
        for i, img_iter in enumerate(imgs):
            handle = axes[i].imshow(img_iter, cmap="gray")
            plt.colorbar(handle, ax=axes[i], fraction=0.1)
        fig.tight_layout()
        plt.show()


class MnMsHDF5SimplifiedDataset(Dataset):
    """
    File hierarchy:

    root
    -csf
    --train
    ---img0
    ---mask0
    --eval
    --test
    ...
    """
    def __init__(self, data_path, source_name, mode, transforms: list, target_size=(256, 256), if_augment=False):
        super(MnMsHDF5SimplifiedDataset, self).__init__()
        assert ".h5" in data_path, "data_path must be a hdf5 file"
        assert source_name in ["csf", "hvhd", "uhe"], "invalid source name"
        assert mode in ["train", "eval", "test"], "invalid mode"

        self.data_path = data_path
        self.source_name = source_name
        self.mode = mode
        self.transforms = transforms
        self.target_size = target_size
        self.if_augment = if_augment

    def __len__(self):
        with h5py.File(self.data_path, "r") as hdf:
            group_source = hdf.get(self.source_name)
            group_data_mode = group_source.get(self.mode)

            return group_data_mode.attrs["SIZE"]

    def __getitem__(self, index):
        assert 0 <= index < self.__len__()
        with h5py.File(self.data_path, "r") as hdf:
            group_source = hdf.get(self.source_name)
            group_data_mode = group_source.get(self.mode)

            img, mask = np.array(group_data_mode.get(f"img{index}")), np.array(group_data_mode.get(f"mask{index}"))

        # Normalize img
        img = normalize(img, norm_type="div_by_max")

        if self.mode == "train" and self.if_augment:
            transform = A.Compose(self.transforms)
            transformed = transform(image=img, mask=mask)
            img, mask = transformed["image"], transformed["mask"]

            resizer = A.Resize(*self.target_size, always_apply=True)
            transformed = resizer(image=img, mask=mask)
            img, mask = transformed["image"], transformed["mask"]

            # (1, H, W), (1, H, W)
            return torch.FloatTensor(img).unsqueeze(0), torch.LongTensor(mask).unsqueeze(0)

        else:
            resizer = A.Resize(*self.target_size, always_apply=True)
            transformed = resizer(image=img, mask=mask)
            img, mask = transformed["image"], transformed["mask"]

            return torch.FloatTensor(img).unsqueeze(0), torch.LongTensor(mask).unsqueeze(0)

    def plot_pair(self, img, mask, figsize=None):
        figsize = plt.rcParams["figure.figsize"] if figsize is None else figsize
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        imgs = [img, mask]
        for i, img_iter in enumerate(imgs):
            handle = axes[i].imshow(img_iter, cmap="gray")
            plt.colorbar(handle, ax=axes[i], fraction=0.1)
        fig.tight_layout()
        plt.show()
