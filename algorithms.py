import copy
import time
import random

import numpy as np
import meta_data as md
from util import judge_whether_print


class algorithm:
    def __init__(self, model, dataset, decay):
        self.model = model
        self.weight = self.init_weight()
        self.dataset = dataset
        self.global_round = 0
        self.total_grad = 0
        self.client_index = []
        self.current_time = []
        self.current_grad_times = []
        self.current_loss = []
        self.current_round = []
        self.decay = decay

        for i in range(md.client_num):
            self.client_index.append(i)

    def init_weight(self):
        if self.model.modelName() == 'svm':
            return np.ones(self.model.len()).reshape(-1, 1)
        else:
            return np.zeros(self.model.len()).reshape(-1, 1)

    def get_loss(self):
        feature, label = self.dataset.get_train_dataset()
        loss = self.model.loss(self.weight, feature, label)
        accuracy = self.model.acc(self.weight, feature, label)
        if md.verbose:
            print("After iteration {}: loss is {}, accuracy is {:.2f}%".format(self.global_round, loss, accuracy))
        return loss

    def save_info(self, start_time):
        current_loss = self.get_loss()
        current_time = time.time()
        self.current_time.append(copy.deepcopy(current_time - start_time))
        self.current_grad_times.append(self.total_grad)
        self.current_loss.append(current_loss)
        self.current_round.append(self.global_round)

    def end_info(self, start_time):
        end_time = time.time()
        print("total time is {:.3f}".format(end_time - start_time))

class FedAvg_SGD(algorithm):
    def __init__(self, model, dataset, decay):
        super().__init__(model, dataset, decay)

    def alg_run(self, start_time):
        partition_index = self.dataset.partition()
        self.save_info(start_time)
        for i in range(md.global_iter):
            self.global_round += 1
            weights_list = []
            chosen_client_num = int(max(md.participate_ratio * md.client_num, 1))
            chosen_client = random.sample(self.client_index, chosen_client_num)

            for k in chosen_client:
                weight_tmp = copy.deepcopy(self.weight)
                weight_of_client = self.update_client(weight_tmp, partition_index[k])
                weights_list.append(copy.deepcopy(weight_of_client))

            self.weight = self.average(weights_list)
            if judge_whether_print(self.global_round):
                self.save_info(start_time)

    def update_client(self, current_weight, chosen_index):
        for i in range(md.local_iter):
            sample_feature, sample_label = self.dataset.sample(chosen_index)
            direction = np.random.randn(self.model.len(), 1)
            upper_val = self.model.loss((current_weight + md.radius * direction), sample_feature, sample_label)
            lower_val = self.model.loss((current_weight - md.radius * direction), sample_feature, sample_label)
            grad = (upper_val - lower_val) * (1 / (2 * md.radius)) * direction
            self.total_grad += 2 * md.batch_size
            eta = self.decay(md.eta, self.global_round)
            current_weight -= eta * grad
        return current_weight

    def average(self, weights_list):
        new_weights = sum(weights_list) / len(weights_list)
        return new_weights




#####################
# choose algorithms #
#####################
def choose_algorithm(model, dataset, decay):
    if md.algorithm_name == 'FedAvg_SGD':
        return FedAvg_SGD(model, dataset, decay)
    else:
        print("The algorithm is not defined")
        exit(0)
