import math
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from torch import nn, sigmoid, no_grad, mul, add
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.autograd import Variable

x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]))
y_data = Variable(torch.Tensor([[1.0], [4.0], [9.0], [16.0], [25.0], [36.0]]))

class LinearRegressionModel(nn.Module):
 
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        self.input_size = input_size
        self.linear1 = nn.Linear(self.input_size, 10)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(10, 1)
        self.loss_func = nn.MSELoss()
 
    def forward(self, x):
        output = self.linear1(x)
        if self.training:
            output = self.dropout(output)
        output = self.linear2(output)
        return output

# our model
our_model = LinearRegressionModel(1)
 
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(our_model.parameters(), lr = 0.001)
 
for epoch in range(10000):
 
    # Forward pass: Compute predicted y by passing
    # x to the model
    pred_y = our_model(x_data)
 
    # Compute and print loss
    loss = our_model.loss_func(pred_y, y_data)
 
    # Zero gradients, perform a backward pass,
    # and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('epoch {}, loss {}'.format(epoch, loss.item()))
 
new_var = Variable(torch.Tensor([[2.5]]))
pred_y = our_model(new_var)
print("predict (after training)", 2.5, our_model(new_var).item())