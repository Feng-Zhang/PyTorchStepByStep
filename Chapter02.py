# 如何利用torch实现线性回归的梯度下降算法，对一个线性回归求w和b的最优解。此外，还需要改善Chapter01代码结构，使其更易读和维护
import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

def make_train_step_fn(model, loss_fn, optimizer):
# Builds function that performs a step in the train loop
    def perform_train_step_fn(x, y):
        # Sets model to TRAIN mode
        model.train()
        # Step 1 - Computes model's predictions - forward pass
        yhat = model(x)
        # Step 2 - Computes the loss
        loss = loss_fn(yhat, y)
        # Step 3 - Computes gradients for "b" and "w" parameters
        loss.backward()
        # Step 4 - Updates parameters using gradients and the learning rate
        optimizer.step()
        optimizer.zero_grad()

        # Returns the loss
        return loss.item()
    # Returns the function that will be called inside the train loop
    return perform_train_step_fn

def make_val_step_fn(model, loss_fn):
# Builds function that performs a step in the validation loop
    def perform_val_step_fn(x, y):
    # Sets model to EVAL mode
        model.eval() 

        # Step 1 - Computes our model's predicted output
        # forward pass
        yhat = model(x)
        # Step 2 - Computes the loss
        loss = loss_fn(yhat, y)
        # There is no need to compute Steps 3 and 4,
        # since we don't update parameters during evaluation
        return loss.item()
    return perform_val_step_fn

def mini_batch(device, data_loader, step_fn):
    mini_batch_losses = []
    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        mini_batch_loss = step_fn(x_batch, y_batch)
        mini_batch_losses.append(mini_batch_loss)
    loss = np.mean(mini_batch_losses)
    return loss

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

## use 
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
lr = 0.1 # Sets learning rate - this is "eta" ~ the "n" like Greek letter
model = nn.Sequential(nn.Linear(1, 1)).to(device) # Now we can create a model and send it at once to the device
optimizer = optim.SGD(model.parameters(), lr=lr) # Defines a SGD optimizer to update the parameters (now retrieved directly from the model)
loss_fn = nn.MSELoss(reduction='mean') # Defines a MSE loss function
train_step_fn = make_train_step_fn(model, loss_fn, optimizer) # Creates the train_step function for our model, loss function and optimizer
val_step_fn = make_val_step_fn(model, loss_fn) # Creates the val_step function for our model and loss function
# writer = SummaryWriter('runs/test') # Creates a Summary Writer to interface with TensorBoard

# # Fetches a single mini-batch so we can use add_graph
# x_sample, y_sample = next(iter(train_loader))
# writer.add_graph(model, x_sample.to(device))

## 3 model training
n_epochs = 200
losses = []
val_losses = []
for epoch in range(n_epochs):
    # inner loop
    loss = mini_batch(device, train_loader, train_step_fn)
    losses.append(loss)
    
    # VALIDATION no gradients in validation!
    with torch.no_grad():
        val_loss = mini_batch(device, val_loader, val_step_fn)
        val_losses.append(val_loss)
    
#     # Records both losses for each epoch under the main tag "loss"
#     writer.add_scalars(main_tag='loss',
#                        tag_scalar_dict={'training': loss, 'validation': val_loss},
#                        global_step=epoch)
# writer.close()

## save model
checkpoint = {'epoch': n_epochs,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'loss': losses,
              'val_loss': val_losses}
torch.save(checkpoint, 'model_checkpoint.pth')
