import numpy as np
import cv2 as cv
import torch
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from albumentations import BasicTransform, Compose
from .utils import convert_mask


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
