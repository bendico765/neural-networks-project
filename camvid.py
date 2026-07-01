from torch.utils.data import Dataset
from torchvision.io import decode_image, ImageReadMode
import pandas as pd
import os
import numpy as np
import torch

class CAMVID_Dataset(Dataset):
    def __init__(self, images_folder_path: str, masks_folder_path: str, labels_filepath: str, transform=None):
        """
        :param images_folder_path: path to the folder containing the image files
        :param masks_folder_path: path to the folder containing the mask files
        :param transform: transformation applied to the image and mask
        """
        self.transform = transform
        self.images_paths = [
            f"{images_folder_path}/{filename}"
            for filename in os.listdir(images_folder_path)
        ]

        self.masks_paths = [
            f"{masks_folder_path}/{filename}"
            for filename in os.listdir(masks_folder_path)
        ]

        # initialize pixel labels
        df = pd.read_csv(labels_filepath)
        colors = df[["r", "g", "b"]].values
        # class_names = df["name"].tolist()

        self.color2label = {
            tuple(color): idx
            for idx, color in enumerate(colors)
        }

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image = decode_image(self.images_paths[idx], mode=ImageReadMode.RGB).to(torch.float32) # Tensor: (C, H, W)
        mask = decode_image(self.masks_paths[idx], mode=ImageReadMode.RGB) # Tensor: (C, H, W)

        # mask to category colors
        mask = mask.permute(1, 2, 0).numpy() # Numpy ndarray: (H, W, C)
        label_mask = np.zeros(mask.shape[:2], dtype=np.int64)  # Numpy ndarray: (H, W)
        for rgb, label in self.color2label.items():
            matches = np.all(mask == rgb, axis = -1) # match all pixels with that class rgb color
            label_mask[matches] = label

        mask = label_mask # numpy ndarray (H, W)
        mask = torch.from_numpy(mask) # tensor (H,W)

        if self.transform:
            # Conversion (C,H,W) -> (H,W,C)
            image = image.permute(1, 2, 0).numpy() # numpy ndarray (H, W, C)
            mask = mask.numpy() # numpy ndarray (H, W)
            
            transformed = self.transform(
                image = image,
                mask = mask
            )

            # Conversion (H,W,C) -> (C,H,W)
            image = torch.from_numpy(transformed["image"]).permute(2, 0, 1)
            mask = torch.from_numpy(transformed["mask"]).to(torch.long) # .permute(2, 0, 1)

        return image, mask # ( Tensor (C,H,W), Tensor(H,W) )
