import time

import numpy as np

import util
from algorithms.utils import choose_algorithm
from data_processing import data
import model
import meta_data as md

if __name__ == '__main__':
    start_time = time.time()
    dataset = data()
    eta_selector = model.eta_calculator()

    dataset.get_dataset()
    decay = eta_selector.choose(md.eta_option)
    global_model = model.select_models(dataset.width())
    max_grad_time = md.max_grad_time_mul * dataset.length()
    algorithm = choose_algorithm(global_model, dataset, decay)

    algorithm.alg_run(start_time)

    el = util.excel_solver(md.algorithm_name + ", prox_val=" + str(md.prox_val))
    el.save_excel(algorithm)
