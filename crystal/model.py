import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable
torch.backends.cudnn.enabled = False

class Flatten(nn.Module):
    def forward(self, inp):
        return inp.reshape(inp.shape[0], -1)

class Net_cifar10(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)
        self.output = None

    def forward(self, x):
        x1 = self.pool(F.relu(self.conv1(x)))
        x2 = self.pool(F.relu(self.conv2(x1)))
        x2 = x2.view(-1, 32 * 8 * 8)
        x3 = F.relu(self.fc1(x2))
        x4 = F.relu(self.fc2(x3))
        x5 = self.fc3(x4)

        return x5
