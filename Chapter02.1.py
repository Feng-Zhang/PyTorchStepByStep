# 如何利用torch实现线性回归的梯度下降算法，对一个线性回归求w和b的最优解。此外，还需要使用类来改善Chapter01代码结构，使其更易读和维护
import os, sys
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc

from stepbystep.v0 import StepByStep

np.random.seed(42)
torch.manual_seed(13)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# generate data
true_b = 1
true_w = 2
N = 100

## Data Generation
np.random.seed(42)
x = np.random.rand(N, 1)
epsilon = (.1 * np.random.randn(N, 1))
y = true_b + true_w * x + epsilon

## data load
x_tensor = torch.from_numpy(x).float()
y_tensor = torch.from_numpy(y).float()
dataset = TensorDataset(x_tensor, y_tensor) # Builds dataset containing ALL data points

### Performs the split
ratio = .8
n_total = len(dataset)
n_train = int(n_total * ratio)
n_val = n_total - n_train
train_data, val_data = random_split(dataset, [n_train, n_val])

### Builds a loader of each set
train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=16)


# put it together 
## 2 model configuration 
lr = 0.1
model = nn.Sequential()
model.add_module('linear', nn.Linear(1, 1))
optimizer = optim.SGD(model.parameters(), lr=lr) # Defines a SGD optimizer to update the parameters
loss_fn = nn.MSELoss(reduction='mean')

## 3 model training
n_epochs = 100
sbs = StepByStep(model, loss_fn, optimizer)
sbs.set_loaders(train_loader, val_loader)
sbs.train(n_epochs)
print(model.state_dict()) 
