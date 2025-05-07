#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import FedAvg_LocalUpdate, FedProx_LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, LogisticRegression, ResNet18
from utils import get_dataset, average_weights, exp_details, FedCC_average_weights




if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('/Users/michael/PycharmProjects/pythonProject/Federated-Learning-PyTorch-al')
    logger = SummaryWriter('./logs')

    args = args_parser()
    exp_details(args)

    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'lr':
        # Logistic Regression
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = LogisticRegression(dim_in=len_in, dim_out=10)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=128,
                               dim_out=10)
    elif args.model == 'resnet':
        global_model = ResNet18()
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy, train_epoch = [], [], []
    ans_loss, ans_acc = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    local_model = FedAvg_LocalUpdate(args=args, dataset=train_dataset,
                                     idxs=user_groups[10], logger=logger)
    acc, loss = local_model.inference(model=global_model)
    print(f' \nAvg Training Stats after 0 global rounds:')
    print(f'Training Loss : {loss}')
    print('Train Accuracy: {:.2f}% \n'.format(100 * acc))
    ans_loss.append(loss)
    ans_acc.append(acc)
    train_epoch.append(0)

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            if args.algorithm == 'FedAvg' or args.algorithm == 'FedCC':
                local_model = FedAvg_LocalUpdate(args=args, dataset=train_dataset,
                                                 idxs=user_groups[idx], logger=logger)
            elif args.algorithm == 'FedProx' or args.algorithm == 'FedProx-CC':
                local_model = FedProx_LocalUpdate(args=args, dataset=train_dataset,
                                                  idxs=user_groups[idx], logger=logger)
            # for key, value in global_model.state_dict().items():
            #     print(key, value.dtype)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        if args.algorithm == 'FedCC' or args.algorithm == 'FedProx-CC':
            global_weights = FedCC_average_weights(local_weights, args)
        else:
            global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)
        train_epoch.append(epoch)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = FedAvg_LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch+1) % 1 == 0:
            ans_loss.append(np.mean(np.array(train_loss)))
            ans_acc.append(100*train_accuracy[-1])
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    # file_name = '../save/objects/{}_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].txt'.\
    #     format(args.dataset, args.model, args.epochs, args.frac, args.iid,
    #            args.local_ep, args.local_bs)

    # with open(file_name, 'wb') as f:
    #     pickle.dump([train_loss, train_accuracy], f)
    data = {
        'epoch': train_epoch,
        'loss': ans_loss,
        'accuracy': ans_acc
    }
    df = pd.DataFrame(data)
    csv_path = './save/objects/{}_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss_{}.csv'.format(args.algorithm, args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs, datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    # 写入 CSV 文件
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    # Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(train_loss)), train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss_{}.png'.
                format(args.algorithm, args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs))

    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
                format(args.algorithm, args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs))