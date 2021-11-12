#!/usr/bin/env python
# coding: utf-8

# In[10]:


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import torchvision.models as models


# In[2]:


class Lenet5(nn.Module):
    def __init__(self):
        super(Lenet5, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = (5, 5), stride = (1, 1), padding = (0, 0))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = (5, 5), stride = (1, 1), padding = (0, 0))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))
        self.linear1 = nn.Linear(400, 120)
        self.relu3 = nn.ReLU()
        self.linear2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.linear3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.linear1(x.reshape(x.shape[0], -1))
        x = self.relu3(x)
        x = self.linear2(x)
        x = self.relu4(x)
        x = self.linear3(x)
        return x

