import torch.optim
from dataloaders import *
from model import *
from torchvision import transforms
from trainer_cifar import train_model, train_attack, eval_attack
from utils.utils import softmax, get_middle
from sklearn.utils import shuffle
from torch.optim import lr_scheduler
import time
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score
import lightgbm as lgb
from warnings import simplefilter
import csv
import os

def crystal_cifar10(config):
    print("START CIFAR10")

    use_cuda = config.general.use_cuda and torch.cuda.is_available()

    device = torch.device("cuda:" + str(config.general.cuda) if use_cuda else "cpu")
    print(device)

    print("START TRAINING TARGET MODEL")
    data_train_target = custum_CIFAR10(True, 0, config, '../data', train=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ]))
    data_test_target = custum_CIFAR10(True, 0, config, '../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ]))
    criterion = nn.CrossEntropyLoss()
    train_loader_target = torch.utils.data.DataLoader(data_train_target, batch_size=config.learning.batch_size, shuffle=True)
    test_loader_target = torch.utils.data.DataLoader(data_test_target, batch_size=config.learning.batch_size)
    dataloaders_target = {"train": train_loader_target, "val": test_loader_target}
    dataset_sizes_target = {"train": len(data_train_target), "val": len(data_test_target)}
    print(len(data_train_target), str(config.general.train_target_size))

    model_target = Net_cifar10().to(device)
    optimizer_target = torch.optim.SGD(model_target.parameters(), lr=config.learning.learning_rate,
                              momentum=config.learning.momentum)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_target, step_size=config.learning.decrease_lr_factor, gamma=config.learning.decrease_lr_every)

    config.set_subkey("general", "iftarget", "yes") # this mode is the target classifier
    model_target, best_acc_target, times = train_model(model_target, config, criterion, optimizer_target, exp_lr_scheduler,dataloaders_target,dataset_sizes_target,
                       num_epochs=config.learning.epochs, types=10)

    _, data_test_set, label_test_set, class_test_set = eval_classifiers(model_target, dataloaders_target, device)

    print("Target——best_accuracy——time: " + str(best_acc_target) + "," + times)

    train_accuracy = eval_target_net(model_target, train_loader_target, config.general.train_target_size,
                                     device=device)

    test_accuracy = eval_target_net(model_target, test_loader_target, config.general.test_target_size, device=device)

    print("Target——train and test accuracy: " + str(train_accuracy) + ", " + str(test_accuracy))

    print("START TRAINING SHADOW MODEL")
    all_shadow_models = []
    all_dataloaders_shadow = []
    data_train_set = []
    label_train_set = []
    class_train_set = []

    if config.general.fed == 1:
        config.set_subkey("general", "iftarget", "yes") # crystal
    else:
        config.set_subkey("general", "iftarget", "no")

    for num_model_sahdow in range(config.general.number_shadow_model):
        print("num_" + str(num_model_sahdow))
        criterion = nn.CrossEntropyLoss()
        data_train_shadow = custum_CIFAR10(False, num_model_sahdow, config, '../data', train=True,
                                           download=True,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                           ]))
        data_test_shadow = custum_CIFAR10(False, num_model_sahdow, config, '../data', train=False,
                                          transform=transforms.Compose([
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                          ]))
        train_loader_shadow = torch.utils.data.DataLoader(data_train_shadow,
                                                          batch_size=config.learning.batch_size, shuffle=True)
        test_loader_shadow = torch.utils.data.DataLoader(data_test_shadow,
                                                         batch_size=config.learning.batch_size)
        dataloaders_shadow = {"train": train_loader_shadow, "val": test_loader_shadow}
        dataset_sizes_shadow = {"train": len(data_train_shadow), "val": len(data_test_shadow)}

        model_shadow = Net_cifar10().to(device)

        optimizer_shadow = torch.optim.SGD(model_shadow.parameters(), lr=config.learning.learning_rate,
                                               momentum=config.learning.momentum)

        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_shadow, step_size=config.learning.decrease_lr_factor,
                                               gamma=config.learning.decrease_lr_every)

        model_shadow, _, _ = train_model(model_shadow, config, criterion, optimizer_shadow,
                                         exp_lr_scheduler, dataloaders_shadow,
                                         dataset_sizes_shadow, num_epochs=config.learning.epochs,
                                         types=10)

        _, data_train_set_unit, label_train_set_unit, class_train_set_unit = eval_classifiers(model_shadow,
                                                                                              dataloaders_shadow,
                                                                                              device)

        data_train_set.append(data_train_set_unit)
        label_train_set.append(label_train_set_unit)
        class_train_set.append(class_train_set_unit)
        all_shadow_models.append(model_shadow)
        all_dataloaders_shadow.append(dataloaders_shadow)

    print("START GETTING DATASET ATTACK MODEL")
    data_train_set = np.concatenate(data_train_set)
    label_train_set = np.concatenate(label_train_set)
    class_train_set = np.concatenate(class_train_set)

    data_train_sets, label_train_sets, class_train_sets = shuffle(data_train_set, label_train_set,
                                                                  class_train_set,
                                                                  random_state=config.general.seed)
    data_test_sets, label_test_sets, class_test_sets = shuffle(data_test_set, label_test_set, class_test_set,
                                                               random_state=config.general.seed)

    if config.general.white == 0: # black-box
        print("Taille dataset train", len(label_train_sets))
        print("Taille dataset test", len(label_test_sets))
        print("START FITTING ATTACK MODEL")

        model = lgb.LGBMClassifier(
            boosting_type='gbdt', num_leaves=55, reg_alpha=0.0, reg_lambda=10,
            max_depth=15, n_estimators=10000, objective='binary',
            subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
            learning_rate=0.06, min_child_weight=1, random_state=20, n_jobs=4
        )
        model.fit(data_train_sets, label_train_sets)
        y_pred_lgbm = model.predict(data_test_sets)

        precision_general = precision_score(y_true=label_test_sets, y_pred=y_pred_lgbm)
        recall_general = recall_score(y_true=label_test_sets, y_pred=y_pred_lgbm)
        accuracy_general = accuracy_score(y_true=label_test_sets, y_pred=y_pred_lgbm)

        precision_per_class, recall_per_class, accuracy_per_class = [], [], []
        for idx in range(10):
            all_index_class = np.where(class_test_sets == idx)

            precision = precision_score(y_true=label_test_sets[all_index_class],
                                        y_pred=y_pred_lgbm[all_index_class])
            recall = recall_score(y_true=label_test_sets[all_index_class],
                                  y_pred=y_pred_lgbm[all_index_class])
            accuracy = accuracy_score(y_true=label_test_sets[all_index_class],
                                      y_pred=y_pred_lgbm[all_index_class])

            precision_per_class.append(precision)
            recall_per_class.append(recall)
            accuracy_per_class.append(accuracy)

        print("Attack(general)——accuracy——precision——recall: " + str(
            accuracy_general) + ", " + str(
            precision_general) + ", " + str(
            recall_general))
        print("Attack(media)——accuracy——precision——recall: " + str(
            get_middle(accuracy_per_class)) + ", " + str(
            get_middle(precision_per_class)) + ", " + str(
            get_middle(recall_per_class)))
        print("Attack——accuracy_per_class: ")
        print(accuracy_per_class)

    else: # white-box
        model, _, times = train_attack(all_shadow_models, all_dataloaders_shadow, config, config.attack.epochs)
        accuracy, precision, recall = eval_attack(model, model_target, dataloaders_target, config)
        print("accuracy = %.2f, precision = %.2f, recall = %.2f" % (accuracy, precision, recall))

    print("END CIFAR10")

def eval_target_net(net, testloader, total, device):

    correct = 0
    with torch.no_grad():
        net.eval()
        for i, (imgs, lbls) in enumerate(testloader):

            imgs, lbls = imgs.to(device), lbls.to(device)
            output = net(imgs)
            predicted = output.argmax(dim=1)

            correct += predicted.eq(lbls).sum().item()


    accuracy = 100 * (correct / total)
    print("\nAccuracy = %.4f %%\n\n" % (accuracy))

    return accuracy


def eval_classifiers(model, dataloaders, device):
    X = []
    Y = []
    C = []

    # Each epoch has a training and validation phase
    model.eval()  # Set model to evaluate mode

    attack_train = dataloaders["train"]
    attack_out = dataloaders["val"]

    for i, ((train_imgs, train_lbls)) in enumerate(attack_train):

        train_imgs, train_lbls = train_imgs.to(device), train_lbls.to(device)
        train_outputs = model(train_imgs)

        for j, out in enumerate(train_outputs.cpu().detach().numpy()):
            X.append(out)
            Y.append(1)

        for cla in train_lbls.cpu().detach().numpy():
            C.append(cla)

    for i, ((train_imgs, train_lbls)) in enumerate(attack_out):

        train_imgs, train_lbls = train_imgs.to(device), train_lbls.to(device)
        train_outputs = model(train_imgs)

        for j, out in enumerate(train_outputs.cpu().detach().numpy()):
            X.append(out)
            Y.append(0)

        for cla in train_lbls.cpu().detach().numpy():
            C.append(cla)

    X = softmax(np.array(X))
    return model, np.array(X), np.array(Y), np.array(C)


