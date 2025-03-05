from src import meta_data as md

from algorithms.FedAvg_SGD_ZO import FedAvg_SGD
from algorithms.FedProx import FedProx
from algorithms.Scaffold import Scaffold


# ********************
# * choose algorithm *
# ********************
def choose_algorithm(model, dataset, decay):
    if md.algorithm_name == 'FedAvg_SGD':
        return FedAvg_SGD(model, dataset, decay)
    elif md.algorithm_name == 'FedProx':
        return FedProx(model, dataset, decay)
    elif md.algorithm_name == 'Scaffold':
        return Scaffold(model, dataset, decay)
    else:
        print("The algorithm is not defined")
        exit(0)
