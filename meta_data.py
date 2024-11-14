# about dataset
dataset_name = 'mnist' # ''
sampling_kind = 'non-iid' # iid and non-iid
training_samples_ratio = 0.8

# about structure
algorithm_name = 'Scaffold' # FedAvg_SGD, FedProx, Scaffold
client_num = 100
participate_ratio = 0.5
batch_size = 64
global_iter = 1000
local_iter = 50

# about model (logistic, svm, neural)
model_name = 'logistic'
hidden_dim = 50

# training
eta_option = 'sqrt'
eta = 1
max_grad_time_mul = 300
radius = 1e-4

# non_iid parameter
non_iid_bar = 3

# whether print info
verbose = True

# FedProx value
prox_val = 0.1

#scaffold kind (0 or 1)
scaffold_kind = 1
