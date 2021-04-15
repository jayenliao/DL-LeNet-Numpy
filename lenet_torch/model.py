import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, channels:int, filter_size:int, hidden_act:str, hidden_sizes:list, pooling_size:int, output_size:int):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=filter_size)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=filter_size)
        self.conv2_drop = nn.Dropout2d()
        self.fc0_size0 = hidden_sizes[0]*13*13
        self.fc0 = nn.Linear(self.fc0_size0, hidden_sizes[0])
        self.fc1 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc2 = nn.Linear(hidden_sizes[1], output_size)
        self.pooling_size = pooling_size
        self.hidden_sizes = hidden_sizes
        if hidden_act.lower() == 'sigmoid':
            self.Activation = torch.sigmoid
            self.Activation_out = F.softmax
        elif hidden_act.lower() == 'relu':
            self.Activation = F.relu
            self.Activation_out = F.log_softmax
        elif hidden_act.lower() == 'tanh':
            self.Activation = F.tanh
            self.Activation_out = F.log_softmax
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, self.pooling_size)
        x = self.Activation(x)

        x = self.conv2(x)
        x = F.max_pool2d(x, self.pooling_size)
        x = self.Activation(x)

        x = x.view(-1, self.fc0_size0)
        x = self.fc0(x)
        x = self.Activation(x)
        x = self.fc1(x)
        x = self.Activation(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return self.Activation_out(x, dim=1) 
    
    