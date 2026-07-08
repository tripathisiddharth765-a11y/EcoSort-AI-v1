import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from data import train_path,test_path
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

def get_dataloaders():
    train_dataset = datasets.ImageFolder(
        train_path,
        transform=train_transformer
    )

    test_dataset = datasets.ImageFolder(
        test_path,
        transform=test_transformer
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )

    return train_loader, test_loader
classes = [
    "Cardboard",
    "Glass",
    "Metal",
    "Paper",
    "Plastic",
    "Trash"
]































