import torch
import numpy as np
import torch.nn.functional as F
import cv2
from torch.utils.data import Dataset
from data.transform import get_transforms
import natsort
import glob
from PIL import Image
import os


def normalize(data_array):
    """
    Normalize the data array to the range [0, 1].
    """
    normalized_data = []
    valid_masks= []
    for i in range(data_array.shape[2]):
        band_data = data_array[:, :, i]
        valid_mask = (band_data > 0)
        result = band_data.copy().astype(np.float32)
        result[valid_mask] = result[valid_mask] / 10000
        result[valid_mask] = np.clip(result[valid_mask], 0, 1)
        result[~valid_mask] = 0.0
        normalized_data.append(result)
        valid_masks.append(valid_mask)
    return np.dstack(normalized_data), np.dstack(valid_masks)

# def normalize(data_array):
#     """
#     Normalize the data array to the range [0, 1].
#     """
#     normalized_data = []
#     valid_masks= []
#     for i in range(data_array.shape[2]):
#         band_data = data_array[:, :, i]
#         valid_mask = (band_data > 0)
#         valid_pixels = band_data[valid_mask]
#         min_val = np.min(valid_pixels)
#         max_val = np.max(valid_pixels)
#         #lower = np.percentile(valid_pixels, lower_percent)
#         #upper = np.percentile(valid_pixels, upper_percent)
#         # result[valid_mask] = np.clip((band[valid_mask] - lower) / (upper - lower), 0, 1)

#         result = band_data.copy().astype(np.float32)
#         result[valid_mask] = (valid_pixels - min_val) / (max_val - min_val)
#         # result[valid_mask] = result[valid_mask] / 10000
#         result[~valid_mask] = 0.0
#         normalized_data.append(result)
#         valid_masks.append(valid_mask)
#     return np.dstack(normalized_data), np.dstack(valid_masks)


def read_images(product_paths):
    images = []
    for path in product_paths:
        data = Image.open(path)
        data = np.array(data)
        images.append(data)

    # image : - > H x W x C
    images = np.dstack(images)
    return images


class Sentinel2Dataset(Dataset):

    def __init__(self, df_x, df_y, train, augmentation, img_size):
        self.df_x = df_x
        self.df_y = df_y
        self.train = train
        self.augmentation = augmentation
        self.img_size = img_size
        # self.transform = get_transforms(train=self.train, augmentation=True, aug_prob=0.5)

    def __getitem__(self, index):
        x_paths = natsort.natsorted(glob.glob(os.path.join(self.df_x["path"][index], "*.png"), recursive=False))
        x_data = read_images(x_paths)
        x_data, x_mask = normalize(x_data)

        y_paths = natsort.natsorted(glob.glob(os.path.join(self.df_y["path"][index], "*.png"), recursive=False))
        y_data = read_images(y_paths)
        y_data, y_mask = normalize(y_data)

        # Apply the same augmentation to both input and target
        # if self.train and self.augmentation:
        #     transformed = self.transform(image=x_data, mask=y_data)
        #     x_data = transformed["image"]
        #     y_data = transformed["mask"]

        # Handle resizing separately from augmentations
        x_data = cv2.resize(x_data, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        y_data = cv2.resize(y_data, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # Resize masks to match image size
        x_mask = cv2.resize(x_mask.astype(np.uint8), (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST).astype(bool)
        y_mask = cv2.resize(y_mask.astype(np.uint8), (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST).astype(bool)

        # Final valid mask is intersection of x and y
        valid_mask = torch.from_numpy(y_mask).bool()
        valid_mask = torch.permute(valid_mask, (2, 0, 1))  # HWC to CHW

        x_data = torch.from_numpy(x_data).float()
        x_data = torch.permute(x_data, (2, 0, 1))  # HWC to CHW

        y_data = torch.from_numpy(y_data).float()
        y_data = torch.permute(y_data, (2, 0, 1))  # HWC to CHW

        return x_data, y_data, valid_mask

    def __len__(self):
        return len(self.df_x)

    # def __init__(self, df_x, df_y, train, augmentation, img_size):
    #     self.df_x = df_x
    #     self.df_y = df_y
    #     self.train = train
    #     self.augmentation = augmentation
    #     self.img_size = img_size
    #     # self.transform = get_transforms(train=self.train, augmentation=self.augmentation)

    # def __getitem__(self, index):
    #     x_paths = natsort.natsorted(glob.glob(os.path.join(self.df_x["path"][index], "*.png"), recursive=False))
    #     x_data = read_images(x_paths)
    #     x_data, x_mask = normalize(x_data)
    #     x_data = cv2.resize(x_data, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
    #     x_mask = cv2.resize(x_mask.astype(np.uint8), (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST).astype(bool)

    #     y_paths = natsort.natsorted(glob.glob(os.path.join(self.df_y["path"][index], "*.png"), recursive=False))
    #     y_data = read_images(y_paths)
    #     y_data, y_mask  = normalize(y_data)
    #     y_data = cv2.resize(y_data, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
    #     y_mask = cv2.resize(y_mask.astype(np.uint8), (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST).astype(bool)

    #     # Final valid mask is intersection of x and y
    #     valid_mask = torch.from_numpy(y_mask).bool()
    #     valid_mask = torch.permute(valid_mask, (2, 0, 1))  # HWC to CHW

    #     x_data = torch.from_numpy(x_data).float()
    #     x_data = torch.permute(x_data, (2, 0, 1))  # HWC to CHW

    #     y_data = torch.from_numpy(y_data).float()
    #     y_data = torch.permute(y_data, (2, 0, 1))  # HWC to CHW

    #     # transformed = self.transform(image=x_data, mask=y_data)
    #     # y_data = transformed["mask"]
    #     # x_data = transformed["image"]

    #     return x_data, y_data, valid_mask

    # def __len__(self):
    #     return len(self.df_x)


class Sentinel2TCIDataset(Dataset):
    def __init__(self, df_path,
                 train,
                 augmentation,
                 img_size):

        self.df_path = df_path
        self.train = train
        self.augmentation = augmentation
        self.img_size = img_size
        self.transform = get_transforms(train=self.train,
                                        augmentation=self.augmentation)

    def __getitem__(self, index):
        # Load images
        x_path = self.df_path.l1c_path.iloc[index]
        x_data = cv2.imread(x_path)
        x_data = cv2.cvtColor(x_data, cv2.COLOR_BGR2RGB)
        x_data = cv2.resize(x_data, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        x_data = np.array(x_data).astype(np.float32) / 255.0
        x_data = torch.from_numpy(x_data).float()
        x_data = torch.permute(x_data, (2, 0, 1))  # HWC to CHW

        y_path = self.df_path.l2a_path.iloc[index]
        y_data = cv2.imread(y_path)
        y_data = cv2.cvtColor(y_data, cv2.COLOR_BGR2RGB)
        y_data = cv2.resize(y_data, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        y_data = np.array(y_data).astype(np.float32) / 255.0
        y_data = torch.from_numpy(y_data).float()
        y_data = torch.permute(y_data, (2, 0, 1))  # HWC to CHW

        # transformed = self.transform(image=x_data, mask=y_data)
        # y_data = transformed["mask"]
        # x_data = transformed["image"]


        return x_data, y_data

    def __len__(self):
        return len(self.df_path)

