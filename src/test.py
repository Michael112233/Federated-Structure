from math import sqrt

import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class model:
    def __init__(self, length):
        self.length = length
        self.model_name = ''

    def modelName(self):
        return self.model_name

    def len(self):
        return self.length

    def predict(self, w, x):
        return 0

    def acc(self, w, x, y):
        y_hat = self.predict(w, x)
        y_hat = (y_hat >= 0.5) * 1
        corrent_array = y_hat - y
        corrent_index = np.where(corrent_array == 0)
        accuracy = len(corrent_index[0]) / len(y)
        return accuracy * 100


class logistic(model):
    def modelName(self):
        return 'logistic'

    def predict(self, w, x):
        y_hat = np.minimum(1 - 1e-15, np.maximum(1e-15, sigmoid(x.dot(w))))
        return y_hat

    def loss(self, w, x, y):
        y_hat = self.predict(w, x, y)
        y_hat_bar = np.minimum(1 - 1e-15, np.maximum(1e-15, (1 - y_hat)))
        # 计算损失函数
        loss = (-1 / len(y)) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(y_hat_bar))
        return loss


class svm(model):
    def modelName(self):
        return 'svm'

    def predict(self, w, x):
        y_hat = x.dot(w)
        return y_hat

    def loss(self, weight, x, y):
        y_hat = self.predict(weight, x)
        loss = np.mean(np.maximum(0.0, 1 - y * y_hat) ** 2) / 2
        return loss


class neural_network(model):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__(0)
        self.input = input_dim
        self.hidden = hidden_dim
        self.output = output_dim

    def modelName(self):
        return 'NeuralNetwork'

    # transform weight vector into two matrix
    def weight_transform(self, weight):
        weight1_tmp = weight[0: self.input * self.hidden]
        weight2_tmp = weight[self.input * self.hidden:]
        weight1 = weight1_tmp.reshape((self.input, self.hidden))
        weight2 = weight2_tmp.reshape((self.hidden, self.output))
        return weight1, weight2

    def predict(self, w, x):
        weight1, weight2 = self.weight_transform(w)
        hidden_input = x.dot(weight1)
        hidden_output = sigmoid(hidden_input)
        output_input = hidden_output.dot(weight2)
        output_output = sigmoid(output_input)
        return output_output

    def loss(self, w, x, y):
        y_pred = self.predict(w, x, y)
        y_pred_bar = np.maximum(1e-15, (1 - y_pred))
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(y_pred_bar))

    def len(self):
        return (self.input + self.output) * self.hidden

class eta_calculator:
    def divide_eta(self, eta, global_iter, local_iter):
        return eta / ((global_iter + 1) * (local_iter + 1))

    def sqrt_eta(self, eta, global_iter, local_iter):
        return eta / sqrt(((global_iter + 1) * (local_iter + 1)))

    def same_eta(self, eta, global_iter, local_iter):
        return eta

    def choose(self, option):
        if option == 'divide':
            return self.divide_eta
        elif option == 'same':
            return self.same_eta
        elif option == 'sqrt':
            return self.sqrt_eta
        else:
            print('Can not calculate eta')
            exit(0)













