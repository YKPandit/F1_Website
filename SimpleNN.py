import fastf1
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np


class SimpleNN(nn.Module):

    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        # Initializing the Neural Network
        super(SimpleNN, self).__init__()

        # Initalizingn the layers
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, output_size)

        # Relu functions
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        # Sigmoid Function
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)

        # predictions = self.sigmoid(x)

        return x