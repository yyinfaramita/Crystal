import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torch
import torch.utils.data as Data
from sklearn.utils import shuffle
import time
from torch.autograd import Variable
import os
import copy
import random
from warnings import simplefilter


def train_model(model, thres, config, criterion, optimizer, dataloaders, dataset_sizes):
    print("DATASET SIZE", dataset_sizes)
    since = time.time()

    num_epochs = config.learning.epochs
    types = config.general.classes

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        dgroup = []
        dtarget = []

        model.train()  # Set model to training mode

        # Iterate over data.
        for batch_idx, (data, target) in enumerate(dataloaders["train"]):
            # if crystal-box or target
            if config.general.iftarget == "yes":
                # if GA
                if "GA" in str(config.general.type):
                    w_old = copy.deepcopy(model.state_dict())

                # if DB or DC
                if ("DB" in str(config.general.type)) or ("DC" in str(config.general.type)):
                    # the order of DB/DC in DB+DC is arbitrary 
                    arbitrary = random.choice([True, False])
                    if arbitrary:
                        if "DC" in str(config.general.type):
                            config.set_subkey("edit", "lamda", random.randint(1, 10) / 10)
                            config.set_subkey("edit", "size", random.randint(1, 10) / 10)
                            config.set_subkey("edit", "size_other", random.randint(1, 10) / 10)

                            data, target = data_confusion(data.cuda(), target.cuda(), types, config)

                        if "DB" in str(config.general.type):
                            config.set_subkey("large", "lamda", random.randint(1, 10) / 10)
                            config.set_subkey("large", "N", random.randint(1, 10) / 10)
                            config.set_subkey("large", "M", random.randint(1, 10) / 10)
                            data, target = data_balance(data.cuda(), target.cuda(), dgroup, dtarget,
                                                        types, config)
                    else:
                        if "DB" in str(config.general.type):
                            config.set_subkey("large", "lamda", random.randint(1, 10) / 10)
                            config.set_subkey("large", "N", random.randint(1, 10) / 10)
                            config.set_subkey("large", "M", random.randint(1, 10) / 10)
                            data, target = data_balance(data.cuda(), target.cuda(), dgroup, dtarget,
                                                        types, config)

                        if "DC" in str(config.general.type):
                            config.set_subkey("edit", "lamda", random.randint(1, 10) / 10)
                            config.set_subkey("edit", "size", random.randint(1, 10) / 10)
                            config.set_subkey("edit", "size_other", random.randint(1, 10) / 10)

                            data, target = data_confusion(data.cuda(), target.cuda(), types, config)

                    # save data and target of batch k
                    dgroup = []
                    for i in range(types):
                        tmp = []
                        for j in range(len(target)):
                            if i == target[j]:
                                tmp.append(data[j])
                        dgroup.append(tmp)
                    dtarget = copy.deepcopy(target)

            


            inputs, labels = data.cuda(), target.cuda()

            optimizer.zero_grad()
            # forward
            # track history if only in train
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # backward + optimize
            loss.backward()
            optimizer.step()

            # if crystal-box or target
            if config.general.iftarget == "yes":
                if "GA" in str(config.general.type):
                    w_new = copy.deepcopy(model.state_dict())
                    gradient_anonymity(model, w_old, w_new, config)

    time_elapsed = time.time() - since

    print("DONE TRAIN")
    times = 'Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60)

    return model, times


def data_balance(data, target, dgroup, dtarget, types, config):
    if len(dgroup) == 0:
        return data, target
    else:
        if target.size(0) != dtarget.size(0):
            return dgroup, dtarget, data, target

        shapes = data[0].shape
        tcenters = []
        tgroup = []

        counts = 0
        lamda = config.large.lamda
        arr_target = target.detach().cpu().numpy()
        for i in range(types):
            tmp = []
            center = torch.zeros(shapes).cuda()
            ranks = np.where(arr_target == i)[0]
            # print(ranks)
            for j in ranks:
                tmp.append(data[j])
                center = center + data[j]

            tgroup.append(tmp)

            if len(tmp) != 0:
                tcenters.append(center / len(tmp))
            else:
                tcenters.append(center)

            ori = len(dgroup[i])
            now = len(tmp)

            if ori > now:
                N = config.large.N
                number = int((ori - now) * N)

                for t in range(number):
                    k = random.randint(0, ori - 1)
                    new_point = dgroup[i][k]

                    if now != 0:
                        new_point = new_point * lamda + tcenters[i] * (1 - lamda)
                    tgroup[i].append(new_point)
                    counts += 1

            elif ori < now:
                M = config.large.M
                number = int((now - ori) * M)
                for t in range(number):

                    choose_number = random.randint(0, len(tgroup[i]) - 1)
                    for k in range(len(tgroup[i])):
                        if k != choose_number:
                            tgroup[i][k] = tgroup[i][k] * (1 - lamda) + tgroup[i][choose_number] * lamda
                    tgroup[i].pop(choose_number)
                    counts -= 1
                    
        # Make sure the quantity of each batch doesn't change
        if counts < 0:
            # print("count < 0")
            for t in range(-counts):
                types_tmp = random.randint(0, types - 1)
                while len(dgroup[types_tmp]) == 0:
                    # print("1" + types_tmp)
                    types_tmp = random.randint(0, types - 1)

                choose_number = random.randint(0, len(dgroup[types_tmp]) - 1)
                new_point = dgroup[types_tmp][choose_number]
                if len(tgroup[types_tmp]) != 0:
                    new_point = new_point * lamda + tcenters[types_tmp] * (1 - lamda)
                tgroup[types_tmp].append(new_point)

        elif counts > 0:
            # print("count > 0")
            for t in range(counts):
                types_tmp = random.randint(0, types - 1)
                while len(tgroup[types_tmp]) == 0:
                    # print("0" + types_tmp)
                    types_tmp = random.randint(0, types - 1)

                choose_number = random.randint(0, len(tgroup[types_tmp]) - 1)
                for k in range(len(tgroup[types_tmp])):
                    if k != choose_number:
                        tgroup[types_tmp][k] = tgroup[types_tmp][k] * (1 - lamda) + tgroup[types_tmp][
                            choose_number] * lamda
                tgroup[types_tmp].pop(choose_number)

        data = torch.Tensor().cuda()
        j = 1

        tmp1 = torch.Tensor().cuda()
        tmp2 = torch.Tensor().cuda()

        target = dtarget

        for i in dtarget:
            types_tmp = i
            while len(tgroup[types_tmp]) == 0:
                types_tmp = random.randint(0, len(tgroup) - 1)

            if j % 2 == 0:
                new_point = tgroup[types_tmp][0]

                tmp2 = new_point
                tmp = torch.stack([tmp1, tmp2], 0)
                data = torch.cat([data, tmp], 0)
            else:
                new_point = tgroup[types_tmp][0]

                tmp1 = new_point

            tgroup[types_tmp].pop(0)
            target[j - 1] = types_tmp
            j += 1

        return data, target


def data_confusion(data, target, types, config):
    tgroup = []
    tcenters = []
    shapes = data[0].shape
    arr_target = target.detach().cpu().detach()
    for i in range(types):
        center = torch.zeros(shapes).cuda()
        ranks = np.where(arr_target == i)[0]
        tmp = []

        for t in ranks:
            tmp.append(data[t])
            center += data[t]

        tgroup.append(tmp)
        if len(tmp) != 0:
            tcenters.append(center / len(tmp))
        else:
            tcenters.append(center)

    lamda = config.edit.lamda
    for i in range(types):
        chooses = int(len(tgroup[i]) * float(config.edit.size))
        for t in range(chooses):
            new_point = tgroup[i][t]
            size_other = config.edit.size_other

            others = random.sample(range(types), int(types * size_other))
            tmps = torch.zeros(shapes).cuda()
            for k in others:
                tmps += tcenters[k]

            tgroup[i][t] = new_point * (1 - lamda) + tmps * lamda

    data = torch.Tensor().cuda()
    j = 1

    tmp1 = torch.Tensor().cuda()
    tmp2 = torch.Tensor().cuda()
    for i in target:
        if j % 2 == 0:
            tmp2 = tgroup[i][0]
            tmp = torch.stack([tmp1, tmp2], 0)
            data = torch.cat([data, tmp], 0)
            j += 1
        else:
            tmp1 = tgroup[i][0]
            j += 1
        tgroup[i].pop(0)

    return data, target


def gradient_anonymity(net, w_old, w_new):
    w = w_new
    k = "fc.weight" # the
    for n in w_new.keys():
        if "fc.weight" in n:
            k = n
    # print(k)
    w[k] = w_new[k] - w_old[k]

    tmp = w[k]

    aves = tmp.mean().float()
    number = torch.ones(tmp.size(0), tmp.size(1)).cuda() * aves

    w[k] = number.float() + w_old[k]

    net.load_state_dict(w)
