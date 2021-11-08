import numpy as np
import cv2 as cv
import torch

from torch.utils.data import Dataset, DataLoader
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
        label = self.dataset[index, ...].unsqueeze(0).float() / 255. # (28, 28) -> (1, 28, 28)
        img = label
        # print(f"inside MNITforMADE: {img.max()}")
        if self.transform is not None:
            img = self.transform(img)
        return 2 * img - 1, label.long()
