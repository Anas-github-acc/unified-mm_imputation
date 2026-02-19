"""
IXI Dataset Loader for MM-GAN Training

Loads pre-extracted 2D axial slices (.npy files) from Phase 1 preprocessing.
Each .npy file contains a stacked array of shape (3, 224, 224) = [T1, T2, PD].

Provides:
  - IXISliceDataset: PyTorch Dataset for loading individual slices
  - create_dataloaders: Helper to build train/val/test DataLoaders
"""

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


class IXISliceDataset(Dataset):
    """
    Dataset for loading pre-extracted IXI axial slices.

    Each .npy file is shape (3, H, W) with channels [T1, T2, PD].
    Data is already normalized to [0, 1] by the extraction pipeline.
    """

    def __init__(self, data_dir, split="train", target_size=(256, 256), augment=False):
        """
        Args:
            data_dir: Root directory containing split subdirs (train/val/test)
            split: 'train', 'val', or 'test'
            target_size: Resize slices to this size (H, W)
            augment: Apply data augmentation (horizontal flip)
        """
        self.data_dir = Path(data_dir) / split
        self.target_size = target_size
        self.augment = augment

        # Collect all .npy files
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.data_dir}")

        self.file_list = sorted(list(self.data_dir.glob("*.npy")))

        if len(self.file_list) == 0:
            raise RuntimeError(f"No .npy files found in {self.data_dir}")

        print(f"[IXISliceDataset] {split}: {len(self.file_list)} slices from {self.data_dir}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        Returns:
            image: (3, H, W) float32 tensor, channels = [T1, T2, PD]
            filename: str, name of the .npy file (for tracking)
        """
        filepath = self.file_list[idx]
        data = np.load(filepath).astype(np.float32)  # (3, H, W)

        # Convert to tensor
        image = torch.from_numpy(data)

        # Resize if needed
        if self.target_size is not None:
            h, w = image.shape[1], image.shape[2]
            if (h, w) != self.target_size:
                image = TF.resize(
                    image, list(self.target_size),
                    interpolation=TF.InterpolationMode.BILINEAR,
                    antialias=True,
                )

        # Augmentation
        if self.augment and torch.rand(1).item() > 0.5:
            image = TF.hflip(image)

        return {
            "image": image,
            "filename": filepath.stem,
        }


def create_dataloaders(
    data_dir,
    batch_size=8,
    target_size=(256, 256),
    num_workers=2,
    augment_train=True,
):
    """
    Create train, val, test DataLoaders.

    Args:
        data_dir: Root directory with train/val/test subdirs
        batch_size: Batch size
        target_size: Resize to (H, W)
        num_workers: DataLoader workers
        augment_train: Apply augmentation to training set

    Returns:
        dict of {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """
    loaders = {}

    for split in ["train", "val", "test"]:
        split_dir = Path(data_dir) / split
        if not split_dir.exists():
            print(f"[WARN] Split directory missing: {split_dir}")
            continue

        augment = augment_train if split == "train" else False
        shuffle = split == "train"

        dataset = IXISliceDataset(
            data_dir=data_dir,
            split=split,
            target_size=target_size,
            augment=augment,
        )

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )

    return loaders
