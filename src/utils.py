#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import math
from collections import Counter

import numpy as np
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid
from sklearn.cluster import KMeans


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)
        # 修改 train_dataset 的标签
        # train_dataset.targets = torch.where(train_dataset.targets >= 5, 1.0, 0.0).float()

        # 修改 test_dataset 的标签
        # test_dataset.targets = torch.where(test_dataset.targets >= 5, 1.0, 0.0).float()

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    # print(w_avg)
    return w_avg

def FedCC_average_weights(w, args):
    """
    Returns the average of the weights.
    """
    cluster_num = args.cluster_size
    _, labels = k_cluster(w, cluster_num, args)
    counter = Counter(labels)
    cluster_size = []
    value_sum = 0
    for i in range(cluster_num):
        cluster_size.append(counter[i])
        value_sum += counter[i] ** 2
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = torch.mul(w_avg[key], (cluster_size[labels[0]]))
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] * (cluster_size[labels[i]])
        w_avg[key] = torch.div(w_avg[key], value_sum)

    # print(w_avg)
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return


# K是聚类个数
def k_cluster(weights, K, max_iter=100):
    # layer_input_weights = []
    # layer_input_biases = []
    # layer_hidden_weights = []
    # layer_hidden_biases = []
    weights_list = []

    for w in weights:
        # tmp_w_input_weight = copy.deepcopy(w['layer_input.weight'])
        # tmp_w_input_bias = copy.deepcopy(w['layer_input.bias'])
        # if args.model == 'mlp':
        #     tmp_w_hidden_weight = copy.deepcopy(w['layer_hidden.weight'])
        #     tmp_w_hidden_bias = copy.deepcopy(w['layer_hidden.bias'])
        # layer_input_weights.append(tmp_w_input_weight.numpy().flatten())
        # layer_input_biases.append(tmp_w_input_bias.numpy().flatten())
        # if args.model == 'mlp':
        #     layer_hidden_weights.append(tmp_w_hidden_weight.numpy().flatten())
        #     layer_hidden_biases.append(tmp_w_hidden_bias.numpy().flatten())
        weight_list = []
        for _, w_single in w.items():
            t = w_single
            # weights_list.append()
            weight_list = np.hstack((weight_list, copy.deepcopy(w_single).flatten()))
        weights_list.append(copy.deepcopy(weight_list))

    # layer_input_weights = np.array(layer_input_weights)
    # layer_input_biases = np.array(layer_input_biases)
    # layer_hidden_weights = np.array(layer_hidden_weights)
    # layer_hidden_biases = np.array(layer_hidden_biases)
    #
    # weights_list = np.hstack((layer_input_weights, layer_input_biases, layer_hidden_weights, layer_hidden_biases))
    # for i in range(len(layer_input_biases)):
    #     if args.model == 'mlp':
    #         tmp = np.hstack((layer_input_weights[i], layer_input_biases[i], layer_hidden_weights[i], layer_hidden_biases[i]))
    #     else:
    #         tmp = np.hstack((layer_input_weights[i], layer_input_biases[i]))
    #     weights_list.append(tmp)
    # weights_list = np.array(weights_list)
    #
    kmeans = KMeans(n_clusters=K, random_state=42)
    kmeans.fit(weights_list)
    #
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    # labels = SpectualCluster(dis_matrix, weights, n_clusters=K)
    # centers = None

    return centers, labels


def Calculate_Matrix_L_sym(W):  # 计算标准化的拉普拉斯矩阵
    degreeMatrix = np.sum(W, axis=1)  # 按照行对W矩阵进行求和
    L = np.diag(degreeMatrix) - W  # 计算对应的对角矩阵减去w
    # 拉普拉斯矩阵标准化，就是选择Ncut切图
    sqrtDegreeMatrix = np.diag(1.0 / (degreeMatrix ** (0.5)))  # D^(-1/2)
    L_sym = np.dot(np.dot(sqrtDegreeMatrix, L), sqrtDegreeMatrix)  # D^(-1/2) L D^(-1/2)
    return L_sym


def normalization(matrix):  # 归一化
    sum = np.sqrt(np.sum(matrix ** 2, axis=1, keepdims=True))  # 求数组的正平方根
    nor_matrix = matrix / sum  # 求平均
    return nor_matrix

def SpectualCluster(W, data, n_clusters):
    L_sym = Calculate_Matrix_L_sym(W)  # 依据W计算标准化拉普拉斯矩阵
    lam, H = np.linalg.eig(L_sym)  # 特征值分解

    t = np.argsort(lam)  # 将lam中的元素进行排序，返回排序后的下标
    H = np.c_[H[:, t[0]], H[:, t[1]]]  # 0和1类的两个矩阵按行连接，就是把两矩阵左右相加，要求行数相等。
    H = normalization(H)  # 归一化处理

    model = KMeans(n_clusters)  # 新建20簇的Kmeans模型
    model.fit(H)  # 训练
    labels = model.labels_  # 得到聚类后的每组数据对应的标签类型

    res = np.c_[data, labels]  # 按照行数连接data和labels
    return labels
