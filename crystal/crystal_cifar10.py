import random
import os
import numpy as np
import torch
from dataloaders import custum_CIFAR10
from model import Net_cifar10
from trainer_cifar import train_model
from utils.utils import softmax, get_middle
from sklearn.utils import shuffle
import torch.nn as nn
from sklearn.metrics import accuracy_score, recall_score, precision_score
import lightgbm as lgb

def crystal_cifar10(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.general.cuda)

    print("START TRAINING TARGET MODEL")
    data_train = custum_CIFAR10(True, 0, config, '../data', train=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ]))
    data_val = custum_CIFAR10(True, 0, config, '../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ]))
    data_test = custum_CIFAR10(True, 1, config, '../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))
    data_train_target = data_train.datasets
    data_val_target = data_val.datasets
    data_test_target = data_test.datasets
    
    criterion = nn.CrossEntropyLoss()
    
    train_loader_target = torch.utils.data.DataLoader(data_train_target, batch_size=config.learning.batch_size, shuffle=True, drop_last=True)
    val_loader_target = torch.utils.data.DataLoader(data_val_target, batch_size=config.learning.batch_size, shuffle=True, drop_last=True)
    test_loader_target = torch.utils.data.DataLoader(data_test_target, batch_size=config.learning.batch_size, shuffle=False, drop_last=True)

    dataloaders_target = {"train": train_loader_target, "val": val_loader_target, "test": test_loader_target}
    dataset_sizes_target = {"train": len(data_train_target), "val": len(data_val_target), "test": len(data_test_target)}

    model_target = Net_cifar10().cuda()
    optimizer_target = torch.optim.Adm(model_target.parameters(), lr=config.learning.learning_rate)

    config.set_subkey("general", "iftarget", "yes") # this mode is the target classifier
    model_target, times = train_model(model_target, config, criterion, optimizer_target, dataloaders_target,
                       dataset_sizes_target)

    data_test_set, label_test_set, class_test_set = eval_classifiers(True, model_target, dataloaders_target)
    print("Target Training time: " + times)

    train_accuracy = eval_target_net(model_target, train_loader_target, config.general.train_target_size)
    test_accuracy = eval_target_net(model_target, test_loader_target, config.general.test_target_size)
    print("Target——train and test accuracy: " + str(train_accuracy) + ", " + str(test_accuracy))


    print("END TARGET TRAINING")


