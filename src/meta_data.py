# about dataset
dataset_name = 'rcv1' # ''
sampling_kind = 'iid' # iid and non-iid
training_samples_ratio = 0.8

# about structure
algorithm_name = 'FedAvg_SGD' # FedAvg_SGD, FedProx, Scaffold
client_num = 100
participate_ratio = 0.5
batch_size = 1000
global_iter = 1000
local_iter = 50

# about model (logistic, svm, neural)
model_name = 'svm'
hidden_dim = 50

# training
eta_option = 'sqrt'
eta = 100 # logistic 10, svm 1
max_grad_time_mul = 300
radius = 1e-4

# non_iid parameter
non_iid_bar = 3

# whether print info
verbose = True

# FedProx value
prox_val = 0.01

#scaffold kind (0 or 1)
scaffold_kind = 0
