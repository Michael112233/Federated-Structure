import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import normalize
import scipy.io as sio
import pandas as pd

import meta_data as md


def isSparse(dataset_name):
    if dataset_name == 'rcv1' or dataset_name == 'fashion_mnist':
        return True
    else:
        return False


class data:
    def __init__(self):
        # meta_data
        self.batch_size = md.batch_size
        self.dataset_name = md.dataset_name
        self.sampling_kind = md.sampling_kind
        self.client_num = md.client_num
        self.training_sample_ratio = md.training_samples_ratio
        self.model_name = md.model_name
        # dataset
        self.feature = None
        self.label = None
        self.train_feature = None
        self.train_label = None
        self.test_feature = None
        self.test_label = None
        self.dataset_size = 0
        self.train_dataset_size = 0
        # for non-iid
        self.sort_index = None
        self.non_iid_bar = md.non_iid_bar

    #######################
    # for loading dataset #
    #######################
    def get_dataset(self):
        total_feature, total_label = self.load_dataset()
        self.feature = total_feature
        self.label = total_label
        # divide_training_sample
        self.dataset_size = len(total_label)
        self.train_dataset_size = int(self.dataset_size * self.training_sample_ratio)
        # iid setting
        rand_idxs = np.random.permutation(self.dataset_size)
        self.train_feature = self.feature[rand_idxs[:self.train_dataset_size]]
        self.train_label = self.label[rand_idxs[:self.train_dataset_size]]
        self.test_feature = self.feature[rand_idxs[self.train_dataset_size:]]
        self.test_label = self.label[rand_idxs[self.train_dataset_size:]]

    def load_dataset(self):
        if isSparse(self.dataset_name):
            return self.load_sparse_dataset()
        else:
            return self.load_dense_dataset()

    def load_sparse_dataset(self):
        if self.dataset_name == 'rcv1':
            feature, label = load_svmlight_file('../dataset/rcv1/rcv1_test.binary')
            label = label.reshape(-1, 1)
            if self.model_name == 'logistic' or self.dataset_name == 'neural':
                label = (label + 1) / 2
        elif self.dataset_name == 'fashion_mnist':
            fmnist_data = pd.read_csv('../dataset/fashion_mnist/fashion-mnist_train.csv')
            fmnist_data = np.array(fmnist_data)
            label = fmnist_data[:, 0]
            feature = fmnist_data[:, 1:]
            label = (label.astype(int) >= 5) * 1  # 将数字>=5样本设为正例，其他数字设为负例
            label = label.reshape(-1, 1)
            if self.model_name == 'svm':
                label = label * 2 - 1
        else:
            print('Cannot find dataset')
            exit(0)
        bias_column = np.ones(feature.shape[0])
        bias_column_sparse = csr_matrix(bias_column).transpose()
        feature = hstack([feature, bias_column_sparse])
        feature = normalize(feature, axis=1, norm='l2')
        return feature, label

    def load_dense_dataset(self):
        if self.dataset_name == 'mnist':
            mnist_data = sio.loadmat('../dataset/mnist/mnist.mat')
            feature = mnist_data['Z']
            label = mnist_data['y']
            label = (label.astype(int) >= 5) * 1
            if self.model_name == 'svm':
                label = label * 2 - 1
        elif self.dataset_name == 'cifar10':
            cifar_data = sio.loadmat("../dataset/cifar10/cifar10.mat")
            feature = cifar_data['dataset']
            label = cifar_data['labels']
            label = (label.astype(int) >= 5) * 1
            if self.model_name == 'svm':
                label = label * 2 - 1
            feature = feature.reshape((feature.shape[0], feature.shape[1]))
        else:
            print('Cannot find dataset')
            exit(0)

        feature = np.hstack((feature, np.ones((feature.shape[0], 1))))
        feature = normalize(feature, axis=1, norm='l2')
        return feature, label

    ########################
    # for sampling dataset #
    ########################
    def sample(self, chosen_index):
        batch_indices = np.random.choice(chosen_index, self.batch_size, replace=True)
        feature_batch = self.train_feature[batch_indices]
        label_batch = self.train_label[batch_indices]
        return feature_batch, label_batch

    def sample_client_data(self, chosen_index):
        feature_data = self.train_feature[chosen_index]
        label_data = self.train_label[chosen_index]
        return feature_data, label_data

    #########################
    # for partition dataset #
    #########################
    def sort(self):
        train_label = self.train_label[:, 0]
        self.sort_index = np.argsort(train_label)

    def partition(self):
        if self.sampling_kind == 'iid':
            dict_index = self.iid_partition()
        elif self.sampling_kind == 'non-iid':
            self.sort()
            dict_index = self.non_iid_partition()
        else:
            print("Partition Error")
            exit(0)
        return dict_index

    def iid_partition(self):
        per_data_length = int(self.train_dataset_size / self.client_num)
        rand_idx = np.random.permutation(self.train_dataset_size)
        dict_idx = {}
        for i in range(self.client_num):
            dict_idx[i] = rand_idx[i * per_data_length: (i + 1) * per_data_length]
        return dict_idx

    def non_iid_partition(self):
        per_data_length = int(self.train_dataset_size / self.client_num / self.non_iid_bar)
        dict_index = {}
        for i in range(self.client_num):
            a = np.arange(i * per_data_length, (i + 1) * per_data_length)
            b = np.arange(int(self.train_dataset_size / 3) + i * per_data_length, int(self.train_dataset_size / 3) + (i + 1) * per_data_length)
            c = np.arange(int(self.train_dataset_size * 2 / 3) + i * per_data_length, int(self.train_dataset_size * 2 / 3) + (i + 1) * per_data_length)
            index = np.hstack((a, b))
            index = np.hstack((index, c))
            dict_index[i] = self.sort_index[index]
        return dict_index

    #############################
    # getting basic information #
    #############################
    def get_train_dataset(self):
        return self.train_feature, self.train_label

    def length(self):
        return self.train_dataset_size

    def width(self):
        return self.train_feature.shape[1]











