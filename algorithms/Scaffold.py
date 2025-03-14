import random
import copy

import numpy as np
from src import meta_data as md
from .BaseAlgorithm import BaseAlgorithm
from src.util import judge_whether_print

class Scaffold(BaseAlgorithm):
    def __init__(self, model, dataset, decay):
        super().__init__(model, dataset, decay)
        self.client_variate = np.zeros((md.client_num, dataset.width()))
        self.server_variate = np.zeros((dataset.width(), 1))
        self.client_variate_difference = np.zeros((dataset.width(), 1))

    def alg_run(self, start_time):
        partition_index = self.dataset.partition()
        self.save_info(start_time)
        for i in range(md.global_iter):
            self.global_round += 1
            weights_list = []
            chosen_client_num = int(max(md.participate_ratio * md.client_num, 1))
            chosen_client = random.sample(self.client_index, chosen_client_num)

            self.client_variate_difference = 0
            for k in chosen_client:
                weight_tmp = copy.deepcopy(self.weight)
                weight_of_client = self.update_client(k, weight_tmp, partition_index[k])
                weights_list.append(copy.deepcopy(weight_of_client))

            self.weight, self.server_variate = self.average(weights_list)
            if judge_whether_print(self.global_round):
                self.save_info(start_time)

    def get_grad(self, current_weight, sample_feature, sample_label):
        direction = np.random.randn(self.model.len(), 1)
        upper_val = self.model.loss((current_weight + md.radius * direction), sample_feature, sample_label)
        lower_val = self.model.loss((current_weight - md.radius * direction), sample_feature, sample_label)
        grad = (upper_val - lower_val) * (1 / (2 * md.radius)) * direction
        # grad = self.model.grad(current_weight, sample_feature, sample_label)
        return grad

    def update_client(self, client_index, current_weight, chosen_index):
        original_weight = copy.deepcopy(current_weight)
        original_variate = copy.deepcopy(self.client_variate[client_index].reshape((self.dataset.width(), 1)))
        eta = md.eta
        for i in range(md.local_iter):
            sample_feature, sample_label = self.dataset.sample(chosen_index)
            self.total_grad += 2 * md.batch_size
            grad = self.get_grad(current_weight, sample_feature, sample_label)
            eta = self.decay(md.eta, self.global_round)
            current_weight -= eta * (grad + self.server_variate - original_variate)
        # modify client variate
        if md.scaffold_kind == 0:
            client_feature, client_label = self.dataset.sample_client_data(chosen_index)
            new_variate = self.get_grad(original_weight, client_feature, client_label)
            self.client_variate[client_index] = new_variate.squeeze()
        elif md.scaffold_kind == 1:
            new_variate = original_variate - self.server_variate + (original_weight - current_weight) / (md.local_iter * eta)
            self.client_variate[client_index] = new_variate.squeeze()
        else:
            exit('Error: Scaffold kind is null.')

        self.client_variate_difference += new_variate - original_variate

        return current_weight

    def average(self, weights_list):
        new_weights = sum(weights_list) / len(weights_list)
        new_server_variate = self.server_variate + self.client_variate_difference / md.client_num
        # print(self.client_variate_difference)
        return new_weights, new_server_variate
