import meta_data as md

from algorithms.FedAvg_SGD_ZO import FedAvg_SGD_ZO


# ********************
# * choose algorithm *
# ********************
def choose_algorithm(model, dataset, decay):
    if md.algorithm_name == 'FedAvg_SGD':
        return FedAvg_SGD_ZO(model, dataset, decay)
    else:
        print("The algorithm is not defined")
        exit(0)
