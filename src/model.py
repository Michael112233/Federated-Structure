from math import sqrt

import numpy as np
import meta_data as md

#######################
# for common function #
#######################
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def select_models(length):
    if md.model_name == 'svm':
        return svm(length)
    elif md.model_name == 'logistic':
        return logistic(length)
    elif md.model_name == 'neural':
        return NeuralNetwork(length)
    else:
        print("No Model")
        exit(0)

#################
# define models #
#################
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
    def __init__(self, length):
        super().__init__(length)
        self.model_name = 'logistic'

    def predict(self, w, x):
        y_hat = np.minimum(1 - 1e-15, np.maximum(1e-15, sigmoid(x.dot(w))))
        return y_hat

    def loss(self, w, x, y):
        y_hat = self.predict(w, x)
        y_hat_bar = 1 - y_hat
        # 计算损失函数
        loss = (-1 / len(y)) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(y_hat_bar))
        return loss

    def grad(self, w, x, y):
        y_hat = self.predict(w, x)
        grad = (1 / len(y)) * x.T.dot(y_hat - y)
        return grad


class svm(model):
    def __init__(self, length):
        super().__init__(length)
        self.model_name = 'svm'

    def predict(self, w, x):
        y_hat = x.dot(w)
        return y_hat

    def loss(self, weight, x, y):
        y_hat = self.predict(weight, x)
        loss = np.mean(np.maximum(0.0, 1 - y * y_hat) ** 2) / 2
        return loss

    def grad(self, w, x, y):
        y_hat = self.predict(w, x)
        dw = -(1 / len(w)) * x.T.dot(np.maximum(0.0, 1 - y * y_hat) * y)
        return dw

    def acc(self, w, x, y):
        y_hat = np.sign(self.predict(w, x))
        corrent_array = y_hat - y
        corrent_index = np.where(corrent_array == 0)
        accuracy = len(corrent_index[0]) / len(y)
        return accuracy * 100



# class neural_network(model):
#     def __init__(self, input_dim, output_dim):
#         super().__init__(0)
#         self.input = input_dim
#         self.hidden = md.hidden_dim
#         self.output = output_dim
#         self.model_name = 'neural'
#
#     # transform weight vector into two matrix
#     def weight_transform(self, weight):
#         weight1_tmp = weight[0: self.input * self.hidden]
#         weight2_tmp = weight[self.input * self.hidden:]
#         weight1 = weight1_tmp.reshape((self.input, self.hidden))
#         weight2 = weight2_tmp.reshape((self.hidden, self.output))
#         return weight1, weight2
#
#     def predict(self, w, x):
#         weight1, weight2 = self.weight_transform(w)
#         hidden_input = x.dot(weight1)
#         hidden_output = sigmoid(hidden_input)
#         output_input = hidden_output.dot(weight2)
#         output_output = sigmoid(output_input)
#         return output_output
#
#     def loss(self, w, x, y):
#         y_pred = self.predict(w, x)
#         y_pred_bar = np.maximum(1e-15, (1 - y_pred))
#         return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(y_pred_bar))
#
#     def len(self):
#         return (self.input + self.output) * self.hidden

class NeuralNetwork(model):
    def __init__(self, input_dim):
        super().__init__(md.hidden_dim * (input_dim + 1))
        self.input = input_dim
        self.hidden = md.hidden_dim
        self.output = 1
        self.model_name = 'neural'
        self.weight1 = None
        self.weight2 = None
        self.hidden_input = None
        self.hidden_output = None
        self.output_input = None
        self.output_output = None

    def weight_transform(self, weight):
        weight1_tmp = weight[0: self.input * self.hidden]
        weight2_tmp = weight[self.input * self.hidden:]
        weight1 = weight1_tmp.reshape((self.input, self.hidden))
        weight2 = weight2_tmp.reshape((self.hidden, self.output))
        return weight1, weight2

    def weight_flatten(self, weight1, weight2):
        weight1_flat = weight1.flatten()
        weight2_flat = weight2.flatten()
        weight = np.concatenate((weight1_flat, weight2_flat))
        weight = weight.reshape((weight.shape[0], 1))
        return weight

    def predict(self, w, x):
        self.weight1, self.weight2 = self.weight_transform(w)
        self.hidden_input = x.dot(self.weight1)
        self.hidden_output = sigmoid(self.hidden_input)
        self.output_input = self.hidden_output.dot(self.weight2)
        self.output_output = sigmoid(self.output_input)
        return self.output_output

    def grad(self, w, x, y):
        size = x.shape[0]
        self.output_output = self.predict(w, x)

        # grad of output_layer
        t1 = (self.output_output - y) / ((1 - self.output_output) * self.output_output)
        t2 = t1 * (1 - self.output_input) * self.output_input
        t3 = self.hidden_output.T.dot(t2)
        dw2 = 1 / size * self.hidden_output.T.dot((self.output_output - y) / ((1 - self.output_output) * self.output_output) * (1 - self.output_input) * self.output_input)
        # print(dw2.shape)
        # grad of hidden_layer
        # dz1 = dz2.dot(self.weight2.T) * self.hidden_output * (1 - self.hidden_output)
        # dw1 = x.T.dot(dz1)
        dw1 = 1 / size * x.T.dot(((self.output_output - y) / ((1 - self.output_output) * self.output_output) * (1 - self.output_input) * self.output_input).dot(self.weight2.T) * (1 - self.hidden_input) * self.hidden_input)

        # d = self.weight_flatten(dw1, dw2)

        return self.weight_flatten(dw1, dw2)

    def loss(self, w, x, y):
        y_pred = self.predict(w, x)
        y_pred_bar = np.maximum(1e-15, (1 - y_pred))
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(y_pred_bar))


###################
# eta calculation #
###################
class eta_calculator:
    def divide_eta(self, eta, global_iter):
        return eta / (global_iter + 1)

    def sqrt_eta(self, eta, global_iter):
        return eta / sqrt((global_iter + 1))

    def same_eta(self, eta, global_iter):
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
