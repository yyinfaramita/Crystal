import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torch
import torch.utils.data as Data
from torch.optim import lr_scheduler
from model import White_attack
from sklearn.utils import shuffle
import time
from torch.autograd import Variable
import os
import copy
import random
from warnings import simplefilter

def train_model(model, config, criterion, optimizer, dataloaders, dataset_sizes):
    print("DATASET SIZE", dataset_sizes)
    since = time.time()

    num_epochs = config.learning.epochs
    types = config.general.classes

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        dgroup = []
        dtarget = []

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                w_old = copy.deepcopy(model.state_dict())
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

                # Iterate over data.
            for batch_idx, (data, target) in enumerate(dataloaders[phase]):
                oridata = data.detach()
                oritarget = target.detach()

                if phase == 'train':

                    if config.general.iftarget == "yes":
                        if "DC" in str(config.general.type):
                            config.set_subkey("large", "lamda", random.randint(1, 10) / 10)
                            config.set_subkey("large", "N", random.randint(1, 10) / 10)
                            config.set_subkey("large", "M", config.large.N)

                            dgroup, dtarget, data, target = data_confusion(data, target, dgroup, dtarget, types, config)

                        if "DB" in str(config.general.type):
                            config.set_subkey("edit", "lamda", random.randint(1, 10) / 10)
                            config.set_subkey("edit", "size", random.randint(1, 10) / 10)
                            config.set_subkey("edit", "size_other",
                                              random.randint(1, 10) / 10)

                            data, target = data_balance(data, target, types, config)


                inputs, labels = data.cuda(), target.cuda()

                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                        oriinputs, labels = oridata.cuda(), oritarget.cuda()
                        orioutputs = model(oriinputs)

                        _, preds = torch.max(orioutputs, 1)
                        loss = criterion(orioutputs, labels)


                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)


            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                w_new = copy.deepcopy(model.state_dict())
                # print(w_new)
                if config.general.iftarget == "yes":
                    if str(config.general.type).find("GA") != -1:
                        gradient_anonymity(model, w_old, w_new, config)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

        print()
    time_elapsed = time.time() - since

    print("DONE TRAIN")
    times = 'Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60)

    return model, times


def data_confusion(data, target, dgroup, dtarget, types, config):
    if len(dgroup) == 0:
        arr_target = target.detach().cpu().numpy()
        for i in range(types):
            tmp = []
            ranks = np.where(arr_target == i)[0]
            for j in ranks:
                tmp.append(data[j])
            dgroup.append(tmp)
        dtarget = target

        return dgroup, dtarget, data, target
    else:

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

            # print(len(tgroup[i]))
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

        # data = torch.Tensor(data)
        dgroup = []
        for i in range(types):
            tmp = []
            for j in range(len(target)):
                if i == target[j]:
                    tmp.append(data[j])
            dgroup.append(tmp)
        dtarget = target

        return dgroup, dtarget, data, target


def data_balance(data, target, types, config):
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

def gradient_anonymity(net, w_old, w_new, device, config):
    w = w_new
    k = "fc2.weight"
    for n in w_new.keys():
        if "fc2.weight" in n:
            k = n
    # print(k)
    w[k] = w_new[k] - w_old[k]

    tmp = w[k]

    aves = tmp.mean().float()
    number = torch.ones(tmp.size(0), tmp.size(1)).cuda() * aves

    w[k] = number.float() + w_old[k]

    net.load_state_dict(w)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
        torch.nn.init.constant_(m.bias.data, 0.0)

def train_attack(shadow_list, data_list, config, num_epochs=100):

    device = torch.device("cuda:" + str(config.general.cuda) if torch.cuda.is_available() else "cpu")
    one_hot = torch.eye(config.general.classes)
    shadow_criterion = nn.CrossEntropyLoss()
    first, second, third, fourth = [], [], [], []

    len1 = 0
    for idx, (model_shadow, data_shadow) in enumerate(zip(shadow_list, data_list)):
        model_shadow.eval()

        for phase in ['train', 'val']:
            for _, (data, target) in enumerate(data_shadow[phase]):
                data, target = data.cuda(), target.cuda()

                outputs = model_shadow(data)
                weights = model_shadow.output

                len1 = weights[0].size(0)

                if phase == 'train':
                    for (weis, outs, tars) in zip(weights, outputs, target):
                        first.append(Variable(weis, requires_grad=True).cuda())
                        second.append(
                            Variable(shadow_criterion(outs.unsqueeze(0), tars.unsqueeze(0)).data, requires_grad=True)
                        .unsqueeze(0).cuda()
                        )
                        third.append(one_hot[tars].cuda())
                        fourth.append(1)

                else:
                    for (weis, outs, tars) in zip(weights, outputs, target):

                        first.append(Variable(weis, requires_grad=True).cuda())
                        second.append(
                            Variable(shadow_criterion(outs.unsqueeze(0), tars.unsqueeze(0)).data, requires_grad=True)
                                .unsqueeze(0).cuda()
                        )
                        third.append(one_hot[tars].cuda())
                        fourth.append(0)

    first, second, third, fourth = shuffle(first, second, third, fourth, random_state=config.general.seed)
    data_sizes = len(first)

    first = torch.utils.data.DataLoader(first, batch_size=config.attack.batch_size, shuffle=False)
    second = torch.utils.data.DataLoader(second, batch_size=config.attack.batch_size, shuffle=False)
    third = torch.utils.data.DataLoader(third, batch_size=config.attack.batch_size, shuffle=False)
    fourth = torch.utils.data.DataLoader(fourth, batch_size=config.attack.batch_size, shuffle=False)

    since = time.time()

    best_accuracy = 0.0
    best_state = None

    model = White_attack(len1, config.general.classes).cuda()
    model.apply(weights_init_normal)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.attack.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config.attack.decrease_lr_factor,
                                           gamma=config.attack.decrease_lr_every)

    correct = 0.0
    total = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        scheduler.step()
        model.train()  # Set model to training mode

        running_loss = 0.0

        epoch_correct = 0.0
        epoch_total = 0
                # Iterate over data.
        for batch_idx, (f, s, t, target) in enumerate(zip(first, second, third, fourth)):
            labels = target.cuda()

            total += labels.size(0)
            epoch_total += labels.size(0)

            optimizer.zero_grad()
            # forward
            outputs = torch.squeeze(model([f, s, t]))

            outputs[outputs < 0.0] = 0.0
            outputs[outputs > 1.0] = 1.0

            # print(outputs.size(), labels.size())
            loss = criterion(outputs, labels.float())

            loss.backward()
            optimizer.step()

            c1 = len([k for k in np.where(outputs.detach().cpu().numpy() >= 0.5)[0] if
                            k in np.where(labels.detach().cpu().numpy() == 1)[0]])
            c2 = len(
                [k for k in np.where(outputs.detach().cpu().numpy() < 0.5)[0]
                 if k in np.where(labels.detach().cpu().numpy() == 0)[0]])

            correct += c1 + c2
            epoch_correct += c1 + c2

            running_loss += loss.item() * labels.size(0)

        epoch_loss = running_loss / data_sizes
        epoch_accuracy = 100 * epoch_correct / epoch_total

        print('Loss: {:.4f} Accuracy: {:.2f}'.format(
            epoch_loss, epoch_accuracy))

            # deep copy the model
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            best_state = model.state_dict()

        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


    print("DONE TRAIN")
    times = 'Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60)

    model.load_state_dict(best_state)

    train_accuracy = 100 * correct / total
    print("Training Accuracy: {:.2f} Best Accuracy: {:.2f}".format(float(train_accuracy), float(best_accuracy)))

    return model, best_accuracy, times

def eval_attack(model_attack, model_target, data_target, config):
    simplefilter(action='ignore', category=FutureWarning)

    device = torch.device("cuda:" + str(config.general.cuda) if torch.cuda.is_available() else "cpu")
    one_hot = torch.eye(config.general.classes)

    model_target.eval()

    attack_train, attack_out = [], []
    target_criterion = nn.CrossEntropyLoss()
    for phase in ['train', 'val']:
        for idx, (data, target) in enumerate(data_target[phase]):

            data, target = data.cuda(), target.cuda()

            outputs = model_target(data)
            weights = model_target.output

            if phase == 'train':
                # data_attack.append(([weights.detach(), losses.detach(), one_hots.detach()], 1))
                for (weis, outs, tars) in zip(weights.detach(), outputs.detach(), target.detach()):
                    attack_train.append(([weis, target_criterion(outs.unsqueeze(0), tars.unsqueeze(0)).unsqueeze(0),
                                          one_hot[tars].cuda()], 1))
            else:
                # data_attack.append(([weights.detach(), losses.detach(), one_hots.detach()], 0))
                for (weis, outs, tars) in zip(weights.detach(), outputs.detach(), target.detach()):
                    attack_out.append(([weis, target_criterion(outs.unsqueeze(0), tars.unsqueeze(0)).unsqueeze(0),
                                          one_hot[tars].cuda()], 0))

    attack_train, attack_out = shuffle(attack_train, attack_out, random_state=config.general.seed)
    attack_train = torch.utils.data.DataLoader(attack_train, batch_size=config.attack.batch_size, shuffle=True)
    attack_out = torch.utils.data.DataLoader(attack_out, batch_size=config.attack.batch_size, shuffle=True)

    model_attack.eval()  # Set model to training mode

    total = 0
    correct = 0

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Iterate over data.
    for i, ((train_imgs, train_lbls), (out_imgs, out_lbls)) in enumerate(zip(attack_train, attack_out)):

        train_inference = torch.squeeze(model_attack(train_imgs))
        out_inference = torch.squeeze(model_attack(out_imgs))

        total += train_inference.size(0) + out_inference.size(0)

        train_inference = torch.squeeze(model_attack(train_imgs)).detach().cpu().numpy()
        out_inference = torch.squeeze(model_attack(out_imgs)).detach().cpu().numpy()

        true_positives += len([k for k in np.where(train_inference >= 0.5)[0] if k in np.where(train_lbls.detach().cpu().numpy() == 1)[0]])
        false_positives += len([k for k in np.where(out_inference >= 0.5)[0] if k in np.where(out_lbls.detach().cpu().numpy() == 0)[0]])
        false_negatives += len([k for k in np.where(train_inference < 0.5)[0] if k in np.where(train_lbls.detach().cpu().numpy() == 1)[0]])

        correct += len([k for k in np.where(train_inference >= 0.5)[0] if k in np.where(train_lbls.detach().cpu().numpy() == 1)[0]])
        correct += len([k for k in np.where(out_inference < 0.5)[0] if k in np.where(out_lbls.detach().cpu().numpy() == 0)[0]])


    accuracy = 100 * correct / total
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives != 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives != 0 else 0
    print("accuracy = %.2f, precision = %.2f, recall = %.2f" % (accuracy, precision, recall))


    return accuracy, precision, recall
