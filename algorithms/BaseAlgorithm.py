from src import meta_data as md
import numpy as np

import time
import copy


class BaseAlgorithm:
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
        model_len = self.model.len()
        if self.model.modelName() == 'neural':
            # model_len = md.hidden_dim * (self.model.len() + 1)
            return np.zeros(model_len).reshape(-1, 1)
        elif self.model.modelName() == 'svm':
            return np.ones(model_len).reshape(-1, 1)
        else:
            return np.zeros(model_len).reshape(-1, 1)

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

