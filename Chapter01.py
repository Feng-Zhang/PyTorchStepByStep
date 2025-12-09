# 如何利用torch实现线性回归的梯度下降算法，对一个线性回归求w和b的最优解
import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch import nn, optim

np.random.seed(42)
torch.manual_seed(13)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# creating tensors in PyTorch, sending them to a device, and making parameters out of them
true_b = 1
true_w = 2
N = 100

## Data Generation
np.random.seed(42)
x = np.random.rand(N, 1)
epsilon = (.1 * np.random.randn(N, 1))
y = true_b + true_w * x + epsilon

## Shuffles the indices
idx = np.arange(N)
np.random.shuffle(idx)

## Uses first 80 random indices for train
train_idx = idx[:int(N*.8)]
val_idx = idx[int(N*.8):] # Uses the remaining indices for validation

## Generates train and validation sets
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]
x_train_tensor,y_train_tensor = torch.tensor(x_train,dtype=torch.float32).to(device), torch.tensor(y_train,dtype=torch.float32).to(device)
print(f'The type of x_train, x_train_tensor, and x_train_tensor are,{type(x_train), type(x_train_tensor), x_train_tensor.type()}')

## create parameters out of them
b = torch.randn(1, dtype=torch.float32, device=device, requires_grad=True)
w = torch.randn(1, dtype=torch.float32, device=device, requires_grad=True)
print(f'The initail valure of b and w are {b}, and {w}')

# understanding PyTorch’s main feature, autograd, to perform automatic differentiation using its associated properties and methods, like backward(),grad, zero_(), and no_grad()
## Step 1 - Computes our model's predicted output - forward pass
yhat = b + w * x_train_tensor
## Step 2 - Computes the loss
error = (yhat - y_train_tensor)
loss = (error ** 2).mean()
## Step 3 - Computes gradients for both "b" and "w" parameters
# No more manual computation of gradients! 
# b_grad = 2 * error.mean()
# w_grad = 2 * (x_tensor * error).mean()
loss.backward()
print(f'The gradient of b and w are {b.grad}, {w.grad}')

## visualizing the dynamic computation graph associated with a sequence of operations


## creating an optimizer to simultaneously update multiple parameters, using its step() and zero_grad() methods


## creating a loss function using PyTorch’s corresponding higher-order function (more on that topic in the next chapter)


## understanding PyTorch’s Module class and creating your own models, implementing __init__() and forward() methods, and making use of its builtin parameters() and state_dict() methods


# transforming the original Numpy implementation into a PyTorch one using the elements above


# realizing the importance of including model.train() inside the training loop (never forget that!)


# implementing nested and sequential models using PyTorch’s layers


# # putting it all together into neatly organized code divided into three distinct parts: data preparation, model configuration, and model training
# ## data preparation
# x_train_tensor,y_train_tensor = torch.tensor(x_train,dtype=torch.float32).to(device), torch.tensor(y_train,dtype=torch.float32).to(device)
# x_val_tensor,y_val_tensor = torch.tensor(x_val,dtype=torch.float32).to(device), torch.tensor(y_val,dtype=torch.float32).to(device)

# ## model configuration
# lr=0.1
# model = nn.Sequential(nn.Linear(1, 1)).to(device) # defines a simple linear model
# optimizer = optim.SGD(model.parameters(), lr=lr)  # Defines a SGD optimizer to update the parameters 
# loss_fn = nn.MSELoss(reduction='mean')            # Defines a MSE loss function

# ## model training
# epochs = 1000
# for epoch in range(epochs):
#     print(f"Epoch {epoch+1}/{epochs}")
#     model.train()  # Puts the model in training mode
#     # Step 1: Compute Model's Predictions
#     yhat = model(x_train_tensor)
#     # Step 2: Compute the Loss
#     loss = loss_fn(yhat, y_train_tensor)
#     # Step 3: Compute the Gradients
#     optimizer.zero_grad()  # Zeros out all the gradients for the parameters
#     loss.backward()        # Computes the gradients for each parameter
#     # Step 4 - Updates parameters using gradients and the  
#     optimizer.step()       # Updates each parameter based on its gradient
#     print(f"Loss: {loss.item()}")
    
# print(f'The final estimated b and w are : {model.state_dict()}')