# src/dataloader.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os


class ResNetDataset(Dataset):
    """Custom Dataset for ResNet classification."""

    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["filepaths"]
        label = self.df.iloc[idx]["labels"]

        try:
            image = Image.open(img_path).convert("RGB")
        except (IOError, FileNotFoundError):
            print(f"Warning: Could not read image {img_path}. Skipping.")
            return None, None

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def create_resnet_dataloaders(batch_size, data_base_path):
    """Creates DataLoaders for the ResNet classification model."""
    data_dir = os.path.join(data_base_path, "resnet")

    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ["train", "val", "test"]
    }

    dataloaders = {
        x: DataLoader(
            image_datasets[x],
            batch_size=batch_size,
            shuffle=(x == "train"),
            num_workers=2,
        )
        for x in ["train", "val", "test"]
    }

    class_names = image_datasets["train"].classes

    return dataloaders["train"], dataloaders["val"], dataloaders["test"], class_names
