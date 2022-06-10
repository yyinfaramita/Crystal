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
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc3 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

        self.output = None

    def forward(self, x):
        x1 = self.pool(torch.relu(self.conv1(x)))
        x2 = self.pool(torch.relu(self.conv2(x1)))
        x2 = x2.view(-1, 16 * 5 * 5)
        x3 = torch.relu(self.fc1(x2))
        x4 = torch.relu(self.fc3(x3))
        x5 = self.fc2(x4)

        return x5

class AlexNet(nn.Module):
  def __init__(self, classes=100):
    super(AlexNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
    self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
    self.conv2 = nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1)
    self.conv3 = nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1)
    self.conv4 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
    self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    self.fc1 = nn.Sequential(
      # nn.Dropout(),
      nn.Linear(256 * 1 * 1, 4096),
      nn.ReLU(inplace=True)
    )
    self.fc3 = nn.Sequential(
        # nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True)
    )
    self.fc2 = nn.Linear(4096, classes)
    self.output = None

  def forward(self, x):
      x1 = self.pool(torch.relu(self.conv1(x)))
      x2 = self.pool(torch.relu(self.conv2(x1)))
      x3 = torch.relu(self.conv3(x2))
      x4 = torch.relu(self.conv4(x3))
      x5 = self.pool(torch.relu(self.conv5(x4)))

      x5 = torch.flatten(x5, 1)
      x6 = self.fc1(x5)
      x7 = self.fc3(x6)
      x8 = self.fc2(x7)

      return x8
