import torch
import torchvision
from torchvision.datasets import CIFAR10
import numpy as np
import random
from PIL import Image
from sklearn.utils import shuffle
import torch.utils.data

class custum_CIFAR10(CIFAR10):
    def __init__(self, target, num, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        data_sizes = config.general.train_target_size

        # self.data, self.targets = shuffle(self.data, self.targets, random_state=config.general.seed)
        if target:
            if self.train:
                self.data = self.data[:data_sizes]
                self.targets = self.targets[:data_sizes]
            else:
                index1 = (data_sizes * (num))
                index2 = (data_sizes * (num + 1))

                self.data = self.data[index1:index2]
                self.targets = self.targets[index1:index2]

            datasets = []
            for img, target in zip(self.data, self.targets):
                img = Image.fromarray(img)
                img = self.transform(img)
                # target = self.target_transform(t)

                datasets.append((img, target))

            self.datasets = datasets
        else:
            self.data = self.data[data_sizes:]
            self.targets = self.targets[data_sizes:]

            tmp = random.sample(range(0, len(self.data)), data_sizes * 2)
            train_tmp = tmp[:data_sizes]
            test_tmp = tmp[data_sizes:]

            train_data = [self.data[i] for i in train_tmp]
            test_data = [self.data[i] for i in test_tmp]
            train_targets = [self.targets[i] for i in train_tmp]
            test_targets = [self.targets[i] for i in test_tmp]

            trains, tests = [], []
            for d, target in zip(train_data, train_targets):
                img = Image.fromarray(d)
                img = self.transform(img)
                # target = self.target_transform(t)

                trains.append((img, target))

            for d, target in zip(test_data, test_targets):
                img = Image.fromarray(d)
                img = self.transform(img)
                # target = self.target_transform(t)

                tests.append((img, target))

            self.trains = trains
            self.tests = tests

