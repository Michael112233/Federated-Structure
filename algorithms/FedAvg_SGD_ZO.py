import random
import copy

import numpy as np

from algorithms.BaseAlgorithm import BaseAlgorithm
from src import meta_data as md
from src.util import judge_whether_print

class FedAvg_SGD(BaseAlgorithm):
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
            grad = (upper_val - lower_val) * direction / (2 * md.radius)
            self.total_grad += 2 * md.batch_size
            eta = self.decay(md.eta, self.global_round)
            current_weight -= eta * grad
        return current_weight

    def average(self, weights_list):
        new_weights = sum(weights_list) / len(weights_list)
        return new_weights
