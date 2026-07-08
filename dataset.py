import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from data import train_path,test_path
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np


train_transformer = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.Resize(size=(224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])
test_transformer=transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset=datasets.ImageFolder(train_path,transform=train_transformer)
test_dataset=datasets.ImageFolder(test_path,transform=test_transformer)

train_loader=DataLoader(dataset=train_dataset,batch_size=32,shuffle=True)
test_loader=DataLoader(dataset=test_dataset,batch_size=32,shuffle=False)
classes=train_dataset.classes































