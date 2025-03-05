import os
import time
import pandas as pd

import meta_data as md

def judge_whether_print(current_round):
    if current_round <= 10:
        return True
    elif current_round <= 200:
        return current_round % 10 == 0
    elif current_round <= 500:
        return current_round % 50 == 0
    elif current_round <= 2000:
        return current_round % 100 == 0
    elif current_round <= 4000:
        return current_round % 200 == 0
    else:
        return current_round % 500 == 0

# *****************
# * print results *
# *****************

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

class excel_solver:
    def __init__(self, file_path_import=""):
        file_path = ""
        mkdir("../performance")
        mkdir("../performance/excel/")
        file_name = str(time.strftime('%Y-%m-%d-%H-%M-%S')) + ".csv"
        if file_path_import == "":
            self.file_path = file_path + file_name
        else:
            self.file_path = file_path + file_path_import

    def save_excel(self, algorithm):
        current_time = algorithm.current_time
        current_loss = algorithm.current_loss
        current_round = algorithm.current_round
        dataframe = pd.DataFrame(
            {'current_round': current_round, 'current_time': current_time,
             'current_loss': current_loss})
        dataframe.to_csv(self.file_path, index=True)

def generate_filename():
    mkdir('../performance/excel/{}'.format(md.algorithm_name))
    mkdir('../performance/excel/{}/{}'.format(md.algorithm_name, md.dataset_name))
    mkdir('../performance/excel/{}/{}/{}'.format(md.algorithm_name, md.dataset_name, md.model_name))
    mkdir('../performance/excel/{}/{}/{}/{}'.format(md.algorithm_name, md.dataset_name, md.model_name, md.sampling_kind))
    mkdir('../performance/excel/{}/{}/{}/{}/eta={}'.format(md.algorithm_name, md.dataset_name, md.model_name, md.sampling_kind, md.eta))
    filepath = '../performance/excel/{}/{}/{}/{}/eta={}/'.format(md.algorithm_name, md.dataset_name, md.model_name, md.sampling_kind, md.eta)
    if md.algorithm_name == 'FedProx':
        filepath += 'prox_val={}/'.format(md.prox_val)
        mkdir(filepath)
    elif md.algorithm_name == 'Scaffold':
        filepath += 'Scaffold_kind={}/'.format(md.scaffold_kind)
        mkdir(filepath)
    filename = filepath + str(time.strftime('%Y-%m-%d-%H-%M-%S')) + ".csv"
    return filename