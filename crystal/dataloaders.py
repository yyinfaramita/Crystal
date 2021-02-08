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

        self.data, self.targets = shuffle(self.data, self.targets, random_state=config.general.seed)

        if self.train:
            if target:
                self.data = self.data[:config.general.train_target_size]
                self.targets = self.targets[:config.general.train_target_size]
            else:

                index1 = (config.general.train_target_size * (num + 1)) % len(self.data)
                index2 = (config.general.train_target_size * (
                        num + 2)) % len(self.data)
                if index2 < index1:
                    self.data = np.append(self.data[index1:], self.data[
                                                              config.general.train_target_size: config.general.train_target_size + index2],
                                          axis=0)
                    self.targets = np.append(self.targets[index1:], self.targets[
                                                                    config.general.train_target_size: config.general.train_target_size + index2],
                                             axis=0)
                    self.targets = self.targets.astype(int)
                else:
                    self.data = self.data[index1:index2]
                    self.targets = self.targets[index1:index2]
        else:
            if target:
                self.data = self.data[:config.general.test_target_size]
                self.targets = self.targets[:config.general.test_target_size]

            else:

                index1 = (config.general.test_target_size * (num + 1)) % len(self.data)
                index2 = (config.general.test_target_size * (
                        num + 2)) % len(self.data)
                # print(index1, index2)
                if index2 < index1:
                    self.data = np.append(self.data[index1:], self.data[
                                                              config.general.test_target_size: config.general.test_target_size + index2],
                                          axis=0)
                    self.targets = np.append(self.targets[index1:], self.targets[
                                                                    config.general.test_target_size: config.general.test_target_size + index2],
                                             axis=0)
                    self.targets = self.targets.astype(int)
                else:
                    self.data = self.data[index1:index2]
                    self.targets = self.targets[index1:index2]

        # self.data.dtype = int


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            index = index % self.config.general.train_target_size
        else:
            index = index % self.config.general.test_target_size
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

