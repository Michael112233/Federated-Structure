import time

import util
from algorithms.utils import choose_algorithm
from data_processing import data
import model
import meta_data as md

# if __name__ == '__main__':
dataset_names = ['mnist', 'fashion_mnist', 'rcv1']
model_names = ['logistic', 'neural', 'svm']
algorithms_names = ['FedAvg_SGD', 'FedProx', 'Scaffold']
sampling_kinds = ['iid', 'non-iid']
eta_list = [0.01, 0.1, 1, 10]

for eta in eta_list:
    for dataset_name in dataset_names:
        for model_name in model_names:
            for algorithm_name in algorithms_names:
                for sampling_kind in sampling_kinds:
                    Scaffold_kind_list = [0]
                    if algorithm_name == 'Scaffold':
                        Scaffold_kind_list = [0, 1]
                    for Scaffold_kind in Scaffold_kind_list:
                        print(dataset_name, model_name, algorithm_name, sampling_kind, Scaffold_kind)
                        md.dataset_name = dataset_name
                        md.sampling_kind = sampling_kind
                        md.algorithm_name = algorithm_name
                        md.model_name = model_name
                        md.scaffold_kind = Scaffold_kind
                        md.eta = eta

                        start_time = time.time()
                        dataset = data()
                        eta_selector = model.eta_calculator()

                        dataset.get_dataset()
                        decay = eta_selector.choose(md.eta_option)
                        global_model = model.select_models(dataset.width())
                        max_grad_time = md.max_grad_time_mul * dataset.length()
                        algorithm = choose_algorithm(global_model, dataset, decay)

                        algorithm.alg_run(start_time)

                        el = util.excel_solver(util.generate_filename())
                        el.save_excel(algorithm)
