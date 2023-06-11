import torch
import torch.nn as nn

class AutoregressiveModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, dilation_rate):
        super(AutoregressiveModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.conv1 = nn.Conv1d(self.input_dim, self.hidden_dim, self.kernel_size, dilation=self.dilation_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(self.hidden_dim, self.input_dim, self.kernel_size, dilation=self.dilation_rate)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x
