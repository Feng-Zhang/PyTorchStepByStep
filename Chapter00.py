# 如何利用numpy实现线性回归的梯度下降算法，对一个线性回归求w和b的最优解
import os, sys
import numpy as np, pandas as pd

# step 0: Data Generation Parameters
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


# Random Initialization
b = np.random.randn(1)
w = np.random.randn(1)
print(b, w)

epochs = 100
# learning rate
lr = 0.1
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    # Step 1: Compute Model's Predictions
    yhat  = b + w * x_train
    # print(yhat )
    # Step 2: Compute the Loss
    error = yhat - y_train
    loss = np.mean(error**2)
    # Step 3: Compute the Gradients
    b_grad = 2 * error.mean()
    w_grad = 2 * (x_train * error).mean()
    # print(b_grad, w_grad)

    # Step 4 - Updates parameters using gradients and the  
    b = b - lr * b_grad
    w = w - lr * w_grad
    print(b, w)

