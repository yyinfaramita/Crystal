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
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

        self.output = None

    def forward(self, x):
        x1 = self.pool(F.relu(self.conv1(x)))
        x2 = self.pool(F.relu(self.conv2(x1)))
        x2 = x2.view(-1, 32 * 8 * 8)
        x3 = F.relu(self.fc1(x2))
        x4 = self.fc2(x3)

        self.output = torch.cat((x1.view(-1, 16 * 16 * 16), x2, x3, x4), 1)
        return x4

class White_attack(nn.Module):

    def __init__(self, len1, len2):
        super(White_attack, self).__init__()

        self.output = nn.Sequential(
            nn.Linear(len1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.loss = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.label = nn.Sequential(
            nn.Linear(len2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.encodes = nn.Sequential(
            nn.Linear(64 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x1 = self.output(x[0])
        x2 = self.loss(x[1])
        x3 = self.label(x[2])

        y = torch.cat((x1, x2, x3), 1)
        z = self.encodes(y)

        return z