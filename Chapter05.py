# 如何利用torch实现图形分类，使用convolution和隐藏层。
import random
import numpy as np
from PIL import Image

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import Compose, Normalize

from data_generation.image_classification import generate_dataset
from helpers import index_splitter, make_balanced_sampler
from stepbystep.v2 import StepByStep

class TransformedTensorDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.x[index]
        
        if self.transform:
            x = self.transform(x)
        
        return x, self.y[index]
        
    def __len__(self):
        return len(self.x)

np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Generation
images, labels = generate_dataset(img_size=10, n_images=1000, binary=False, seed=17)

# Data prepare
## Builds tensors from numpy arrays BEFORE split Modifies the scale of pixel values from [0, 255] to [0, 1]
x_tensor = torch.as_tensor(images / 255).float()
y_tensor = torch.as_tensor(labels).long()

## Uses index_splitter to generate indices for training and validation sets
train_idx, val_idx = index_splitter(len(x_tensor), [80, 20])
x_train_tensor = x_tensor[train_idx]
y_train_tensor = y_tensor[train_idx]
x_val_tensor = x_tensor[val_idx]
y_val_tensor = y_tensor[val_idx]

## composed transforms
train_composer = Compose([Normalize(mean=(.5,), std=(.5,))])
val_composer = Compose([Normalize(mean=(.5,), std=(.5,))])
train_dataset = TransformedTensorDataset(x_train_tensor, y_train_tensor, transform=train_composer)
val_dataset = TransformedTensorDataset(x_val_tensor, y_val_tensor, transform=val_composer)
sampler = make_balanced_sampler(y_train_tensor) # Builds a weighted random sampler to handle imbalanced classes
train_loader = DataLoader(dataset=train_dataset, batch_size=16, sampler=sampler)# Uses sampler in the training set to get a balanced data loader
val_loader = DataLoader(dataset=val_dataset, batch_size=16)

# Model configuration
model_cnn1 = nn.Sequential()
## Featurizer Block 1: 1@10x10 -> n_channels@8x8 -> n_channels@4x4
n_channels = 1
model_cnn1.add_module('conv1', nn.Conv2d(in_channels=1, out_channels=n_channels, kernel_size=3))
model_cnn1.add_module('relu1', nn.ReLU())
model_cnn1.add_module('maxp1', nn.MaxPool2d(kernel_size=2))
model_cnn1.add_module('flatten', nn.Flatten()) # Flattening: n_channels * 4 * 4

## Classification
model_cnn1.add_module('fc1', nn.Linear(in_features=n_channels*4*4, out_features=10)) # Hidden Layer
model_cnn1.add_module('relu2', nn.ReLU())
model_cnn1.add_module('fc2', nn.Linear(in_features=10, out_features=3))# Output Layer
lr = 0.1
multi_loss_fn = nn.CrossEntropyLoss(reduction='mean')
optimizer_cnn1 = optim.SGD(model_cnn1.parameters(), lr=lr)

## training
sbs_cnn1 = StepByStep(model_cnn1, multi_loss_fn, optimizer_cnn1)
sbs_cnn1.set_loaders(train_loader, val_loader)
sbs_cnn1.train(20)
res = StepByStep.loader_apply(sbs_cnn1.val_loader, sbs_cnn1.correct)
print(res)