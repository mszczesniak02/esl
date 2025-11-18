
import torch.nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from hparams import *


class FashionMNISTDataset(Dataset):
    """_summary_ Fashion ***MNIST*** dataset, based on torch.nn
    """

    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = int(row.iloc[0])
        pixels = row.iloc[1:].values.astype(np.float32)
        image = pixels.reshape(28, 28)

        image = image / 255.0

        if self.transform:
            image = self.transform(image)

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)

        return image, label


def dataset_visualize(dataset: pd.DataFrame, file_title: str = "dataset_visualize.png", file_path: str = FIGURES_PATH) -> None:

    label_names = ["T-shirt/top", "Trouser", "Pullover", "Dress",
                   "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    fig, axes = plt.subplots(2, 5, figsize=(12, 8))
    axes = axes.flatten()

    label_col = dataset.columns[0]
    labels = sorted(dataset[label_col].unique())[:10]

    for i, label in enumerate(labels):
        subset = dataset[dataset[label_col] == label]
        if subset.empty:
            axes[i].axis('off')
            continue

        sample = subset.sample(n=1).iloc[0]
        pixels = sample.iloc[1:].values
        img = pixels.reshape(28, 28).astype(np.uint8)

        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"Label: {label} ({label_names[label]})")
        axes[i].axis('off')

    plt.tight_layout(h_pad=0)
    plt.savefig(file_path + file_title)


def dataset_load(train_path: str = TRAIN_DATA_PATH, test_path: str = TEST_DATA_PATH, bsize: int = BATCH_SIZE, epochs: int = EPOCHS, print_info: bool = False) -> tuple[DataLoader, DataLoader]:
    """Load Fashion-MNIST dataset from CSV files and return train/test DataLoaders

    Args:
        train_path (str, optional): Path to training CSV file. Defaults to TRAIN_DATA_PATH.
        test_path (str, optional): Path to test CSV file. Defaults to TEST_DATA_PATH.
        bsize (int, optional): Batch size for DataLoaders. Defaults to BATCH_SIZE.
        epochs (int, optional): Number of epochs (unused in function). Defaults to EPOCHS.
        print_info (bool, optional): Print dataset statistics. Defaults to False.

    Returns:
        tuple[DataLoader, DataLoader]: Train and test DataLoaders
    """
    train_dataset = pd.read_csv(train_path)
    test_dataset = pd.read_csv(test_path)

    train_ds = FashionMNISTDataset(train_dataset)
    test_ds = FashionMNISTDataset(test_dataset)

    train_loader = DataLoader(
        train_ds, batch_size=bsize, shuffle=True, num_workers=WORKERS)
    test_loader = DataLoader(test_ds, batch_size=bsize,
                             shuffle=False, num_workers=WORKERS)

    if print_info:
        print(f"          Epochs: {epochs}")
        print(f"      Batch size: {bsize}")
        print(f"   Train samples: {len(train_ds)}")
        print(f"    Test samples: {len(test_ds)}")
        print(f"   Train batches: {len(train_loader)}")
        print(f"    Test batches: {len(test_loader)}")

    return train_loader, test_loader
